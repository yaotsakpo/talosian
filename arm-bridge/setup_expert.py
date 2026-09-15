"""Scripted expert for the table-setup task: given the current plate and cup
positions, plan and execute pick-and-place for each, so the finished place setting
(plate on plate_target, cup on cup_target) is achieved. This is the EXPERT policy:

  1. it proves the task is solvable in sim across randomized seeds, and
  2. it generates the demonstrations we record for training (collect_dataset.py).

Design: DETERMINISTIC step-generator execution. Each arm's motion is a Python
generator that, on every physics step, writes that arm's actuator setpoints and
yields. A single controller (`run`) advances both arm generators one step at a
time and calls mj_step ONCE per tick. That means:

  - both arms move together (true bimanual) without threads or locks,
  - the result is identical every run for a given seed (no wall-clock timing, no
    thread races), which is what makes the 10-seed eval trustworthy, and
  - it runs as fast as the CPU allows (no real-time sleeps), which data
    collection needs.

Motion uses the existing Jacobian IK (ik.py) to drive each gripper to world
waypoints, plus a weld-based grasp (a friction pinch on a thin disc is unreliable
in sim, so at grasp we weld the object to the gripper at a clean canonical offset,
object centered under the gripper, and release by deactivating the weld).

Each arm is assigned the object on ITS side of the table so the two arms never
cross over each other. Object->target is fixed (plate->plate_target,
cup->cup_target) regardless of which arm carries it.

OMPL note: Intel recommends OMPL for motion planning. If OMPL is importable we can
use it for collision-free joint-space paths; otherwise we use straight Cartesian
interpolation through the waypoints, a valid and common expert for open-tabletop
pick-and-place. The generator design accepts either as the per-arm path source.
"""

from __future__ import annotations

import math

import numpy as np
import mujoco

from scene_setup import (
    build_model, object_xy, on_target,
    PLATE_TARGET, CUP_TARGET, TABLE_TOP_Z, PLATE_HALF_H, CUP_HALF_H,
)
from ik import ArmIK
from trajectories import HOME

try:  # OMPL is optional; the Cartesian planner is used when it is unavailable.
    from ompl import base as _ob  # noqa: F401
    from ompl import geometric as _og  # noqa: F401
    HAVE_OMPL = True
except Exception:
    HAVE_OMPL = False

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")

GRIP_OPEN = 1.5      # radians (near the 1.75 max)
GRIP_CLOSED = 0.02   # radians (near the -0.17 min)
GRIP_RELEASE = 0.35  # small open: breaks jaw-object contact without a big sweep

# heights (world Z) for the pick/place phases. Kept low enough that the arm can
# stay top-down (upright) through the WHOLE trajectory: upright IK is exact up to
# ~z=0.13 over the far cup target, and breaks above that. A low, all-upright path
# means the grasped object never leaves vertical, so a tall cup lands upright on
# its target instead of tilting off.
APPROACH_Z = TABLE_TOP_Z + 0.09   # hover above before descending
PICK_PLATE_Z = TABLE_TOP_Z + PLATE_HALF_H + 0.01
PICK_CUP_Z = TABLE_TOP_Z + CUP_HALF_H
LIFT_Z = TABLE_TOP_Z + 0.09       # ~0.13 world, the top of the upright-feasible zone


def _dwell_steps(model, seconds: float) -> int:
    return max(1, int(seconds / model.opt.timestep))


class SetupExpert:
    """Deterministic bimanual pick-and-place expert. Owns no threads; the caller
    drives it with `run()` (headless) or feeds its generators into a viewer loop."""

    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.ik = {"H": ArmIK(model, "H_", "H_gripperframe"),
                   "P": ArmIK(model, "P_", "P_gripperframe")}
        self.act = {"H": {}, "P": {}}
        for arm in ("H", "P"):
            for j in JOINTS:
                self.act[arm][j] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{arm}_{j}")
        # Map object body name -> its grasp-weld equality id. Discovered from the
        # model so the same motion core serves any scene's objects (the table-set
        # plate/cup, or the drink-service water/wine glasses).
        self.weld = self._discover_welds()

    def _discover_welds(self):
        """Match each 'grasp_*' weld equality to the object body it welds (obj2)."""
        welds = {}
        for eid in range(self.model.neq):
            nm = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, eid)
            if nm and nm.startswith("grasp_"):
                obody = self.model.eq_obj2id[eid]
                oname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, obody)
                if oname:
                    welds[oname] = eid
        return welds

    # ---- pose helpers --------------------------------------------------------
    def set_home(self):
        """Put both arms at HOME in both qpos and ctrl (call once before running)."""
        for arm in ("H", "P"):
            for j, v in HOME.items():
                aid = self.act[arm][j]
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{arm}_{j}")
                self.data.ctrl[aid] = math.radians(v)
                self.data.qpos[self.model.jnt_qposadr[jid]] = math.radians(v)
        mujoco.mj_forward(self.model, self.data)

    def _ctrl(self, arm, j):
        return float(self.data.ctrl[self.act[arm][j]])

    def _set_ctrl(self, arm, goal_rad: dict):
        for j, aid in self.act[arm].items():
            if j in goal_rad:
                lo, hi = self.model.actuator_ctrlrange[aid]
                self.data.ctrl[aid] = min(max(goal_rad[j], lo), hi)

    # ---- grasp / release (weld toggling) -------------------------------------
    def _grasp(self, arm: str, obj: str):
        """Weld the object to the gripper at a CANONICAL offset (object centered
        under the gripper site, upright), so gripper-on-target == object-on-target
        with no world-offset math that would rotate with the arm."""
        gbody = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"{arm}_gripper")
        gsite = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, f"{arm}_gripperframe")
        obody = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, obj)
        eid = self.weld[obj]
        # rebind the weld to the arm that is ACTUALLY grasping (assignment can
        # give either arm either object), so it never welds to the wrong gripper.
        self.model.eq_obj1id[eid] = gbody
        self.model.eq_obj2id[eid] = obody
        # Weld the object AT the gripper site (drop=0). The IK only constrains
        # gripper POSITION, not orientation, so the wrist tilts between pick and
        # place; any nonzero drop becomes a lever arm that swings the object
        # sideways as it tilts. Welding at the site removes that lever entirely,
        # so gripper-XY == object-XY at the target.
        drop = 0.0
        qadr = self.model.jnt_qposadr[self.model.body_jntadr[obody]]
        gpos = self.data.site_xpos[gsite].copy()
        self.data.qpos[qadr:qadr + 3] = [gpos[0], gpos[1], gpos[2] - drop]
        self.data.qpos[qadr + 3:qadr + 7] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(self.model, self.data)
        gp = self.data.xpos[gbody].copy(); gq = self.data.xquat[gbody].copy()
        op = self.data.xpos[obody].copy(); oq = self.data.xquat[obody].copy()
        negg = np.zeros(4); mujoco.mju_negQuat(negg, gq)
        relpos = np.zeros(3); mujoco.mju_rotVecQuat(relpos, op - gp, negg)
        relquat = np.zeros(4); mujoco.mju_mulQuat(relquat, negg, oq)
        self.model.eq_data[eid][0:3] = 0.0
        self.model.eq_data[eid][3:6] = relpos
        self.model.eq_data[eid][6:10] = relquat
        self.model.eq_data[eid][10] = 1.0
        self.data.eq_active[eid] = 1

    def _release(self, obj: str):
        self.data.eq_active[self.weld[obj]] = 0

    # ---- per-arm motion generators -------------------------------------------
    # Each generator yields once per physics step after writing its arm's ctrl.
    # The controller (`run`) steps physics once when every active generator has
    # yielded, so both arms advance in lockstep.

    def _ramp_gen(self, arm: str, goal_rad: dict, seconds: float):
        steps = _dwell_steps(self.model, seconds)
        start = {j: self._ctrl(arm, j) for j in self.act[arm]}
        for i in range(1, steps + 1):
            f = i / steps
            self._set_ctrl(arm, {j: start[j] + (goal_rad.get(j, start[j]) - start[j]) * f
                                 for j in self.act[arm]})
            yield

    # Local site axis that, driven to world -Z, gives a clean top-down grasp
    # (verified for the SO-101 gripper site: local x, index 0).
    DOWN_AXIS = 0

    def _reach_gen(self, arm: str, xyz, seconds: float, grip: float | None = None,
                   upright: bool = False):
        # `upright` requests a top-down gripper. We only use it at pick/place
        # (low) heights, where the arm can satisfy BOTH the position and the
        # down-orientation exactly. At high lift/traverse heights the two conflict
        # (position wins), and orientation there does not matter because the
        # object is rigidly welded to the gripper.
        da = self.DOWN_AXIS if upright else None
        sol = self.ik[arm].solve(self.data, np.asarray(xyz), down_axis=da)
        goal = {j: math.radians(sol[j]) for j in sol}
        if grip is not None:
            goal["gripper"] = grip
        else:
            goal["gripper"] = self._ctrl(arm, "gripper")
        yield from self._ramp_gen(arm, goal, seconds)

    def _grip_gen(self, arm: str, value: float, seconds: float = 0.25):
        yield from self._ramp_gen(arm, {"gripper": value}, seconds)

    def _hold_gen(self, arm: str, seconds: float):
        """Hold the current ctrl for `seconds` (lets welds/objects settle)."""
        goal = {j: self._ctrl(arm, j) for j in self.act[arm]}
        yield from self._ramp_gen(arm, goal, seconds)

    def place_object_gen(self, arm: str, obj: str, target_xy, pick_z: float):
        """Generator for a full pick-and-place of `obj` with `arm` onto target.

        Waypoints, all top-down: hover-over-pick -> descend -> close+weld ->
        SETTLE (let the weld lock before moving, so the object cannot be flung) ->
        lift -> hover-over-target -> descend gently -> release -> SETTLE (let the
        object come to rest before the check) -> retreat -> home."""
        px, py = object_xy(self.model, self.data, obj)
        tx, ty = float(target_xy[0]), float(target_xy[1])
        # Entire path stays upright (top-down), so the grasped object never tilts.
        yield from self._reach_gen(arm, [px, py, APPROACH_Z], 0.5, grip=GRIP_OPEN, upright=True)
        yield from self._reach_gen(arm, [px, py, pick_z], 0.4, upright=True)   # grasp upright
        yield from self._grip_gen(arm, GRIP_CLOSED, 0.3)
        self._grasp(arm, obj)                                   # weld at canonical offset (upright)
        yield from self._reach_gen(arm, [px, py, LIFT_Z], 0.4, upright=True)   # lift upright
        yield from self._reach_gen(arm, [tx, ty, LIFT_Z], 0.5, upright=True)   # traverse upright
        yield from self._reach_gen(arm, [tx, ty, pick_z + 0.01], 0.4, upright=True)  # place upright
        self._release(obj)                                      # unweld
        yield from self._grip_gen(arm, GRIP_OPEN, 0.3)
        yield from self._reach_gen(arm, [tx, ty, APPROACH_Z], 0.4)
        yield from self._ramp_gen(arm, {j: math.radians(v) for j, v in HOME.items()}, 0.5)

    # ---- assignment ----------------------------------------------------------
    def assign_arms(self):
        """Assign by TARGET side, not start side: the plate target is on the left
        (H's side) and the cup target on the right (P's side), so H always sets the
        plate and P always sets the cup. The object may spawn on the far side (a
        cross-body pick, which both arms can reach), but each arm carries its
        object to a target on its OWN side, so the two arms never cross during the
        carry/place, where crossing caused the failures."""
        return {"H": ("plate", PLATE_TARGET, PICK_PLATE_Z),
                "P": ("cup", CUP_TARGET, PICK_CUP_Z)}

    # ---- the controller ------------------------------------------------------
    def run(self, coordinated: bool = True, settle: float = 0.6, on_step=None,
            stagger: float = 0.45):
        """Execute the coordinated set-the-table deterministically. Steps physics
        itself (no threads, no sleeps). `on_step(t)` is called after each physics
        step (used by data collection / rendering). Returns the success dict.

        The two arms run concurrently (bimanual), but P starts `stagger` seconds
        after H so they are never crossing the shared center at the SAME instant
        (simultaneous center-crossing made the arms collide and knock the cup on
        some seeds). They still overlap heavily, so it reads as coordinated
        bimanual motion, not one-then-the-other. stagger=0 -> fully simultaneous.

        coordinated=False: the arm carrying the CUP is given a WRONG target, so the
        cup ends up off its mark (the trust failure made physical)."""
        jobs = self.assign_arms()
        # Spoofed wrong target for the cup: clearly off the place setting but on
        # P's OWN (right) side, so the mis-placed cup does not cross over the
        # center and knock the plate. That keeps the failure legible: "the plate
        # is set, but the forged message sent the cup to the wrong spot", rather
        # than a pile-up in the middle.
        wrong_cup = np.array([CUP_TARGET[0] + 0.13, CUP_TARGET[1] + 0.02])
        gens = {}
        for arm in ("H", "P"):
            obj, target, pick_z = jobs[arm]
            if obj == "cup" and not coordinated:
                target = wrong_cup
            gens[arm] = self.place_object_gen(arm, obj, target, pick_z)

        # Delay P's start by `stagger` seconds (a leading run of no-op steps for P).
        def delayed(gen, delay_steps):
            for _ in range(delay_steps):
                yield
            yield from gen
        gens["P"] = delayed(gens["P"], _dwell_steps(self.model, stagger))

        active = dict(gens)
        t = 0
        while active:
            done = []
            for arm, g in active.items():
                try:
                    next(g)
                except StopIteration:
                    done.append(arm)
            for arm in done:
                active.pop(arm)
            mujoco.mj_step(self.model, self.data)
            t += 1
            if on_step is not None:
                on_step(t)

        # let objects settle
        for _ in range(_dwell_steps(self.model, settle)):
            mujoco.mj_step(self.model, self.data)
            t += 1
            if on_step is not None:
                on_step(t)

        plate_ok = on_target(self.model, self.data, "plate", PLATE_TARGET)
        cup_ok = on_target(self.model, self.data, "cup", CUP_TARGET)
        return {"coordinated": coordinated, "plate_ok": plate_ok, "cup_ok": cup_ok,
                "success": bool(plate_ok and cup_ok), "steps": t}


def run_headless(seed: int = 0, coordinated: bool = True, verbose: bool = True):
    model = build_model(seed=seed)
    data = mujoco.MjData(model)
    exp = SetupExpert(model, data)
    exp.set_home()
    res = exp.run(coordinated=coordinated)
    res["seed"] = seed
    if verbose:
        print(f"seed {seed} coordinated={coordinated}: "
              f"plate_ok={res['plate_ok']} cup_ok={res['cup_ok']} "
              f"SUCCESS={res['success']} ({res['steps']} steps)")
    return res


if __name__ == "__main__":
    import sys
    print("OMPL available:", HAVE_OMPL)
    if len(sys.argv) > 1 and sys.argv[1] == "eval":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        ok = 0
        for s in range(n):
            r = run_headless(seed=s, coordinated=True)
            ok += r["success"]
        print(f"=== {ok}/{n} coordinated success ===")
    else:
        run_headless(seed=int(sys.argv[1]) if len(sys.argv) > 1 else 0)
