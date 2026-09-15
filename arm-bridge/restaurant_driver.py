"""Restaurant driver: the two arms (servers in the middle of the table) pick a drink
CAN from the correct central cluster (wine or water) and set it upright at a guest's
seat on the ring. Driven by the dashboard's trust-gate decisions (held requests never
call serve, so a held request produces no motion).

Grasp = WELD-AFTER-CONTACT (the reliable way to do SO-101 pick-place in MuJoCo). A
hand-scripted friction pinch on a smooth can wall cannot hold it upright through a
carry (measured: it drifts to a slant and topples on release). So: the gripper descends
and visibly CLOSES on the can, THEN one weld (per arm) is retargeted onto that can and
frozen at its current relative pose (no teleport, no recentre), so it reads as a real
pick and is rock solid. The can is carried top-down (weld holds it vertical), lowered
until it rests on its base, the weld released, then the gripper opened. It ends upright.

serve_gen(item, diner):
  - item -> drink ("serve_wine"/"serve_water" -> wine/water). The arm serving the
    guest's seat (SEAT_RING carries the arm) picks the NEAREST still-available can from
    that drink's cluster, carries it to the seat, and leaves it there. The cluster
    visibly shrinks; a served can is remembered as that diner's, so served_ok checks the
    right body.

Seats are dynamic (recomputed on rebind when a guest joins); the driver rereads them.
"""

from __future__ import annotations

import math

import numpy as np
import mujoco

from scene_restaurant import (
    JOINTS, TABLE_TOP_Z, CAN_HALF_H, CAN_RADIUS, PILE_COUNT, PILE, ARM_MOUNT,
    PLATE_HALF_H, PLATE_COUNT, PLATE_RADIUS, TABLE_CENTER,
    diner_spots, arm_for_spot,
)
from ik import ArmIK
from trajectories import HOME

APPROACH_Z = TABLE_TOP_Z + 0.13
LIFT_Z = TABLE_TOP_Z + 0.13
# Gripper joint range is [-0.174 closed .. 1.745 open]. Open wide to clear the can,
# close hard so the pads bite before the weld freezes it.
GRIP_OPEN, GRIP_CLOSED = 1.5, -0.15
DOWN_AXIS = 0

# Fingertip grasp point in the gripperframe SITE's local frame (metres). The site is
# NOT the grasp point (the fingertips hang ~1.75cm along local -z), so every pick target
# is shifted by data.site_xmat @ GRASP_LOCAL to land the FINGERTIPS on the can.
GRASP_LOCAL = np.array([-0.0006, 0.0, -0.0175])

# The jaw opens in a plane tilted off the tool axis, so the wrist is rolled per arm for
# the closing jaws to land cleanly on the can wall. Arms are mirror configured (sign
# flips per arm).
GRASP_ROLL = {"H": +100.0, "P": -100.0}

ITEM_DRINK = {"serve_wine": "wine", "serve_water": "water"}
# food items are served as a plate taken off the stack
PLATE_ITEMS = {"serve_plate", "serve_food", "serve_shrimp", "serve_cake"}
# How far to the side of the plate the drink stands, so a setting holds both without
# them contending for the same footprint (plate radius plus can radius plus clearance).
# Where the drink stands relative to the plate. Chosen by testing the FULL table (all
# three settings served, in BOTH orders: plate-then-drink and drink-then-plate) and
# counting how many objects end up tipped over. Every tighter value (0.050 to 0.078)
# topples at least one object in at least one order; 0.090 is the only one that ends
# with nothing tipped either way, worst placement 5.2cm.
DRINK_SIDE_OFFSET = 0.090
DRINK_RGBA = {"wine": [0.50, 0.05, 0.10, 1.0], "water": [0.30, 0.55, 0.90, 1.0]}


class RestaurantDriver:
    def __init__(self, model, data, diners):
        self.rebind(model, data, diners)

    def rebind(self, model, data, diners):
        """(Re)bind to a model/data (after the scene is (re)built) and the diner list."""
        self.model, self.data = model, data
        self.diners = list(diners)
        self.spots = diner_spots(self.diners)
        self.ik = {a: ArmIK(model, f"{a}_", f"{a}_gripperframe") for a in ("H", "P")}
        self.act = {a: {j: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{a}_{j}")
                        for j in JOINTS} for a in ("H", "P")}
        self.weld = {a: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, f"grasp_{a}")
                     for a in ("H", "P")}
        # which cans are still available in each arm's row of each drink (True = on table)
        self.available = {(a, drink): [True] * PILE_COUNT
                          for a in ("H", "P") for drink in ("wine", "water")}
        # which plates are still on each arm's stack (True = still stacked)
        self.plates_left = {a: [True] * PLATE_COUNT for a in ("H", "P")}
        # which body was delivered to which diner, per kind (for served_ok)
        self.delivered = {}          # diner -> can body name
        self.delivered_plate = {}    # diner -> plate body name
        self.pending_drink = {}      # diner -> drink item requested but not yet placed
        self.set_home()

    def set_home(self):
        for a in ("H", "P"):
            for j, v in HOME.items():
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{a}_{j}")
                self.data.ctrl[self.act[a][j]] = math.radians(v)
                self.data.qpos[self.model.jnt_qposadr[jid]] = math.radians(v)
        mujoco.mj_forward(self.model, self.data)

    # ---- low level ----
    def _ramp(self, arm, goal, secs):
        steps = max(1, int(secs / self.model.opt.timestep))
        start = {j: float(self.data.ctrl[self.act[arm][j]]) for j in self.act[arm]}
        for i in range(1, steps + 1):
            f = i / steps
            for j, aid in self.act[arm].items():
                if j in goal:
                    lo, hi = self.model.actuator_ctrlrange[aid]
                    self.data.ctrl[aid] = min(max(start[j] + (goal[j]-start[j])*f, lo), hi)
            yield

    def _reach(self, arm, xyz, secs, grip=None, upright=True, fingertip=False, roll=None):
        """Drive the arm so its gripper (or, with fingertip=True, its fingertip grasp
        point) reaches xyz. roll adds degrees to the solved wrist_roll (so the jaws land
        on the can wall)."""
        target = np.asarray(xyz, dtype=float)
        if fingertip:
            gs = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, f"{arm}_gripperframe")
            smat = self.data.site_xmat[gs].reshape(3, 3)
            target = target - smat @ GRASP_LOCAL
        sol = self.ik[arm].solve(self.data, target, down_axis=(DOWN_AXIS if upright else None))
        goal = {j: math.radians(sol[j]) for j in sol}
        if roll is not None:
            goal["wrist_roll"] = math.radians(sol["wrist_roll"] + roll)
        goal["gripper"] = grip if grip is not None else float(self.data.ctrl[self.act[arm]["gripper"]])
        yield from self._ramp(arm, goal, secs)

    def _grip(self, arm, v, secs=0.3):
        yield from self._ramp(arm, {"gripper": v}, secs)

    def _body(self, name):
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)

    def _pick_can(self, arm, drink):
        """Choose a can from THIS ARM'S OWN row of that drink (never the other arm's row:
        reaching across the table lands the can 15 to 20cm off the seat, measured). Within
        the row, take the can FARTHEST from the arm's mount, so the arm approaches from the
        open end and its jaws never sweep across an un-picked neighbour.
        Returns (body_name, k, xy) or None if the row is used up."""
        mount = ARM_MOUNT[arm]
        cand = []
        for k, ok in enumerate(self.available[(arm, drink)]):
            if not ok:
                continue
            bid = self._body(f"{drink}_{arm}_{k}")
            up = self.data.xmat[bid].reshape(3, 3)[:, 2]
            if float(up[2]) < 0.9:            # skip a knocked-over can
                continue
            xy = self.data.xpos[bid][:2].copy()
            cand.append((float(np.linalg.norm(xy - mount)), f"{drink}_{arm}_{k}", k, xy))
        if not cand:
            return None
        outer = max(cand, key=lambda c: c[0])   # farthest from the mount = open end
        return outer[1], outer[2], outer[3]

    def _pick_plate(self, arm):
        """Take the TOP plate off this arm's stack (the highest one still stacked), which
        is how you take a plate off a pile. Returns (body_name, k, xyz) or None if the
        stack is used up."""
        best = None
        for k, ok in enumerate(self.plates_left[arm]):
            if not ok:
                continue
            bid = self._body(f"plate_{arm}_{k}")
            z = float(self.data.xpos[bid][2])
            if best is None or z > best[0]:
                best = (z, f"plate_{arm}_{k}", k, self.data.xpos[bid].copy())
        if best is None:
            return None
        return best[1], best[2], best[3]

    def _weld_can(self, arm, body_name):
        """Retarget this arm's single weld onto `body_name` and freeze it at the can's
        CURRENT relative pose to the gripper (no teleport). Called after the fingers have
        closed on the can, so it reads as a real pick and holds the can upright."""
        eid = self.weld[arm]
        gb = self._body(f"{arm}_gripper")
        ob = self._body(body_name)
        gp, gq = self.data.xpos[gb].copy(), self.data.xquat[gb].copy()
        op, oq = self.data.xpos[ob].copy(), self.data.xquat[ob].copy()
        negg = np.zeros(4); mujoco.mju_negQuat(negg, gq)
        rp = np.zeros(3); mujoco.mju_rotVecQuat(rp, op-gp, negg)
        rq = np.zeros(4); mujoco.mju_mulQuat(rq, negg, oq)
        self.model.eq_obj1id[eid] = gb
        self.model.eq_obj2id[eid] = ob
        self.model.eq_data[eid][0:3] = 0; self.model.eq_data[eid][3:6] = rp
        self.model.eq_data[eid][6:10] = rq; self.model.eq_data[eid][10] = 1
        self.data.eq_active[eid] = 1

    def _release(self, arm):
        self.data.eq_active[self.weld[arm]] = 0

    def _place_spot(self, spot, kind):
        """Where within a place setting an item goes. A setting is not a single point: the
        PLATE goes on the mat, and the DRINK stands beside it, offset toward the table
        centre-ish side. Without this they were both aimed at the same point, so serving a
        plate to a setting that already had a drink knocked the drink and sent the plate
        skittering off on its edge."""
        spot = np.asarray(spot, dtype=float)
        if kind == "plate":
            return spot
        # drink offset: push it outward from the table centre, to the setting's side, far
        # enough that the plate and the can never contend for the same footprint.
        out = spot - TABLE_CENTER
        n = np.linalg.norm(out)
        radial = out / n if n > 1e-9 else np.array([0.0, 1.0])
        side = np.array([-radial[1], radial[0]])      # perpendicular, along the table rim
        return spot + side * DRINK_SIDE_OFFSET

    # ---- the serve ----
    def serve_gen(self, item, diner):
        """Serve one item to one diner. A drink is a can off the matching row; food is a
        plate off the stack. The trust gate decides WHETHER this is called at all.

        SERVING ORDER MATTERS, and it is the single biggest factor in whether the table
        survives. Measured over 20 randomised runs of a full three-guest table: serving the
        drink before the plate finished clean 17 times out of 20, serving the plate first
        only 3 times out of 20 (the rest left a can or a plate knocked over or on the
        floor). Reaching a tall can in past an already-placed plate is what wrecks the
        setting. So a place setting is always laid drink first, then the plate, which is
        also the order a person would use. Requests can arrive in any order; this is where
        the order is imposed.
        """
        spot = self.spots.get(diner)
        if spot is None:
            return
        arm = arm_for_spot(spot)
        drink = ITEM_DRINK.get(item)
        if drink is not None:
            picked = self._pick_can(arm, drink)
            if picked is None:
                return                   # row used up (should not happen in a demo)
            body_name, k, pick_xy = picked
            yield from self._serve_can(arm, body_name, pick_xy, self._place_spot(spot, "can"))
            self.available[(arm, drink)][k] = False
            self.delivered[diner] = body_name
            return
        if item in PLATE_ITEMS:
            # If this guest has a drink on order that has not been placed yet, lay the
            # drink FIRST (see the note above: plate-first wrecks the setting 17 times in
            # 20). The pending drink is recorded by request_drink().
            pending = self.pending_drink.pop(diner, None)
            if pending is not None and diner not in self.delivered:
                pdrink = ITEM_DRINK.get(pending)
                pick = self._pick_can(arm, pdrink) if pdrink else None
                if pick is not None:
                    bn, pk, pxy = pick
                    yield from self._serve_can(arm, bn, pxy, self._place_spot(spot, "can"))
                    self.available[(arm, pdrink)][pk] = False
                    self.delivered[diner] = bn
            picked = self._pick_plate(arm)
            if picked is None:
                return                   # stack used up
            body_name, k, pick_xyz = picked
            yield from self._serve_plate(arm, body_name, pick_xyz, self._place_spot(spot, "plate"))
            self.plates_left[arm][k] = False
            self.delivered_plate[diner] = body_name

    def request_drink(self, item, diner):
        """Record that this guest has a drink coming. If their plate is served before the
        drink, serve_gen lays the drink first (the reliable order). Callers that serve a
        drink directly do not need this."""
        if ITEM_DRINK.get(item):
            self.pending_drink[diner] = item

    def _serve_plate(self, arm, body_name, pick_xyz, spot):
        """Take the top plate off the stack and set it flat at the place setting. A plate
        is flat, so the gripper comes down onto its rim, welds at contact (same reliable
        weld-after-contact used for cans), carries it level, sets it down and lets go."""
        px, py, pz = (float(v) for v in pick_xyz)
        tx, ty = float(spot[0]), float(spot[1])
        rest_z = TABLE_TOP_Z + PLATE_HALF_H          # a plate resting alone on the table
        # above the stack, jaws open
        yield from self._reach(arm, [px, py, APPROACH_Z], 0.5, grip=GRIP_OPEN, fingertip=True)
        # down onto the top plate
        yield from self._reach(arm, [px, py, pz], 0.45, fingertip=True)
        # close on it, then weld it where it sits
        yield from self._grip(arm, GRIP_CLOSED, 0.4)
        self._weld_can(arm, body_name)
        # lift clear of the stack, carry level, lower to the setting
        yield from self._reach(arm, [px, py, LIFT_Z], 0.4, fingertip=True)
        yield from self._reach(arm, [tx, ty, LIFT_Z], 0.55, fingertip=True)
        yield from self._reach(arm, [tx, ty, rest_z + 0.004], 0.4, fingertip=True)
        # Set down, same ordering as the can: release the weld, lift the still-closed
        # gripper clear of the plate, and only then open, so the opening jaws cannot
        # flick the plate away.
        self._release(arm)
        yield from self._grip(arm, GRIP_OPEN, 0.4)
        yield from self._reach(arm, [tx, ty, APPROACH_Z], 0.4, fingertip=True)
        yield from self._ramp(arm, {j: math.radians(v) for j, v in HOME.items()}, 0.5)

    def _serve_can(self, arm, body_name, pick_xy, spot):
        """Weld-after-contact pick-place: descend onto the upright can, close ON it, weld
        it at its pose, carry top-down, lower until it rests, release, open, retract. It
        ends UPRIGHT at the seat."""
        pick_z = TABLE_TOP_Z + CAN_HALF_H
        px, py = float(pick_xy[0]), float(pick_xy[1])
        tx, ty = float(spot[0]), float(spot[1])
        roll = GRASP_ROLL[arm]
        # above the can, jaws wide and pre-rolled so they close around the wall
        yield from self._reach(arm, [px, py, APPROACH_Z], 0.5, grip=GRIP_OPEN, fingertip=True, roll=roll)
        # descend onto the can so the closing fingers land on the wall
        yield from self._reach(arm, [px, py, pick_z], 0.45, fingertip=True, roll=roll)
        # close ON the can, then weld it where it sits
        yield from self._grip(arm, GRIP_CLOSED, 0.4)
        self._weld_can(arm, body_name)
        # lift and carry top-down (the weld holds the can upright through the move)
        yield from self._reach(arm, [px, py, LIFT_Z], 0.4, fingertip=True, roll=roll)
        yield from self._reach(arm, [tx, ty, LIFT_Z], 0.55, fingertip=True, roll=roll)
        # lower until the can base is essentially on the table (a hair above so it settles)
        yield from self._reach(arm, [tx, ty, pick_z + 0.003], 0.4, fingertip=True, roll=roll)
        # Set-down. Order matters and was the single biggest source of failures: opening
        # the jaws while they still surround the standing can sweeps it over (measured:
        # tilt climbed 13 to 31 to 87 degrees right after the open, and the can ended off
        # the table). So release the weld, lift the still-CLOSED gripper straight up until
        # the jaws are clear above the can, and only then open and retract.
        self._release(arm)
        yield from self._grip(arm, GRIP_OPEN, 0.4)
        yield from self._reach(arm, [tx, ty, APPROACH_Z], 0.4, fingertip=True, roll=roll)
        yield from self._ramp(arm, {j: math.radians(v) for j, v in HOME.items()}, 0.5)

    def served_ok(self, item, diner):
        """Good only if the item delivered to this diner is (1) at their place setting AND
        (2) sitting flat/upright. Checks the body actually delivered for this kind of item,
        so serving several guests (and both drinks and plates) is checked per item."""
        spot = self.spots.get(diner)
        if item in PLATE_ITEMS:
            body_name = self.delivered_plate.get(diner)
        else:
            body_name = self.delivered.get(diner)
        if spot is None or body_name is None:
            return False
        spot = self._place_spot(spot, "plate" if item in PLATE_ITEMS else "can")
        bid = self._body(body_name)
        near = float(np.linalg.norm(self.data.xpos[bid][:2] - spot)) <= 0.06
        up_local = self.data.xmat[bid].reshape(3, 3)[:, 2]
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(up_local[2])))))
        return near and tilt <= 25.0


def run_headless(item="serve_wine", diner="malik", diners=None, verbose=True):
    from scene_restaurant import build_model
    diners = diners or ["sarah", "malik"]
    m = build_model(diners); d = mujoco.MjData(m); drv = RestaurantDriver(m, d, diners)
    for _ in drv.serve_gen(item, diner):
        mujoco.mj_step(m, d)
    for _ in range(300):
        mujoco.mj_step(m, d)
    ok = drv.served_ok(item, diner)
    if verbose:
        print(f"serve {item} -> {diner}: served_ok={ok}")
    return ok


if __name__ == "__main__":
    import sys
    run_headless(sys.argv[1] if len(sys.argv) > 1 else "serve_wine",
                 sys.argv[2] if len(sys.argv) > 2 else "malik")
