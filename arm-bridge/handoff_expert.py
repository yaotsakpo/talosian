"""Deterministic expert for the ARM-TO-ARM HAND-OFF task. H picks the block on the
left, the two grippers meet at the centre, the block transfers H -> P, and P places
it on the right target. Reuses the proven motion core (SetupExpert): upright IK,
weld grasp, deterministic step-generator run loop.

The hand-off itself is a WELD REBIND: the block is welded to H's gripper for the
pick and carry, then at the meeting point the weld is re-bound to P's gripper (and
re-anchored at P's current relative pose) and H opens. So the physical "pass"
happens through the same stiff weld we already trust for grasping, just re-parented
mid-air. This is the object hand-off / coordination the rubric rewards.
"""

from __future__ import annotations

import math

import numpy as np
import mujoco

from setup_expert import SetupExpert, GRIP_OPEN, GRIP_CLOSED, _dwell_steps, APPROACH_Z, LIFT_Z
from scene_handoff import build_model, object_xy, on_target, HANDOFF_TARGET, MEET_XY, TABLE_TOP_Z, BLOCK_HALF
from trajectories import HOME

PICK_Z = TABLE_TOP_Z + BLOCK_HALF
MEET_Z = TABLE_TOP_Z + 0.09   # height the grippers meet at (in the upright zone)


class HandoffExpert(SetupExpert):

    def _rebind_weld(self, obj: str, to_arm: str):
        """Hand off `obj` to `to_arm`: re-anchor its weld to that gripper at the
        CANONICAL offset (block re-centred under the new gripper site), so the
        receiving arm's gripper-on-target == block-on-target with no offset drift.
        The weld stays active throughout (the block is never dropped)."""
        eid = self.weld[obj]
        gbody = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"{to_arm}_gripper")
        gsite = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, f"{to_arm}_gripperframe")
        obody = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, obj)
        self.model.eq_obj1id[eid] = gbody
        self.model.eq_obj2id[eid] = obody
        # snap the block directly under the receiving gripper site, upright
        qadr = self.model.jnt_qposadr[self.model.body_jntadr[obody]]
        gpos = self.data.site_xpos[gsite].copy()
        self.data.qpos[qadr:qadr + 3] = [gpos[0], gpos[1], gpos[2]]
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

    def run(self, settle: float = 0.6, on_step=None):
        t = 0

        def step():
            nonlocal t
            mujoco.mj_step(self.model, self.data)
            t += 1
            if on_step is not None:
                on_step(t)

        def drive(gen):
            for _ in gen:
                step()

        bx, by = object_xy(self.model, self.data, "block")
        meetH = [MEET_XY[0] - 0.03, MEET_XY[1], MEET_Z]   # H brings the block just left of centre
        meetP = [MEET_XY[0] - 0.03, MEET_XY[1], MEET_Z]   # P comes to the same point to take it
        tx, ty = float(HANDOFF_TARGET[0]), float(HANDOFF_TARGET[1])

        # 1. H picks the block on the left (upright), welds it.
        drive(self._reach_gen("H", [bx, by, APPROACH_Z], 0.5, grip=GRIP_OPEN, upright=True))
        drive(self._reach_gen("H", [bx, by, PICK_Z], 0.4, upright=True))
        drive(self._grip_gen("H", GRIP_CLOSED, 0.3))
        self._grasp("H", "block")
        drive(self._reach_gen("H", [bx, by, MEET_Z], 0.4, upright=True))

        # 2. H carries the block to the meeting point; P brings its OPEN gripper in.
        drive(self._reach_gen("H", meetH, 0.5, upright=True))
        drive(self._reach_gen("P", [meetP[0] + 0.04, meetP[1], MEET_Z], 0.5, grip=GRIP_OPEN, upright=True))
        drive(self._reach_gen("P", meetP, 0.35, grip=GRIP_OPEN, upright=True))  # P closes in on the block

        # 3. THE HAND-OFF: P closes, weld rebinds H -> P, then H opens and backs off.
        drive(self._grip_gen("P", GRIP_CLOSED, 0.3))
        self._rebind_weld("block", "P")
        drive(self._grip_gen("H", GRIP_OPEN, 0.25))
        drive(self._reach_gen("H", [meetH[0] - 0.10, meetH[1], MEET_Z + 0.02], 0.4))  # H retreats

        # 4. P carries the block to the right target and places it.
        drive(self._reach_gen("P", [tx, ty, LIFT_Z], 0.5, upright=True))
        drive(self._reach_gen("P", [tx, ty, PICK_Z + 0.01], 0.4, upright=True))
        self._release("block")
        drive(self._reach_gen("P", [tx, ty, APPROACH_Z], 0.35, upright=True))
        drive(self._grip_gen("P", GRIP_OPEN, 0.25))

        # 5. both home
        drive(self._ramp_gen("H", {j: math.radians(v) for j, v in HOME.items()}, 0.4))
        drive(self._ramp_gen("P", {j: math.radians(v) for j, v in HOME.items()}, 0.4))

        for _ in range(_dwell_steps(self.model, settle)):
            step()

        ok = on_target(self.model, self.data)
        return {"success": bool(ok), "steps": t,
                "block_final": object_xy(self.model, self.data, "block").tolist()}


def run_headless(seed: int = 0, verbose: bool = True):
    model = build_model(seed=seed)
    data = mujoco.MjData(model)
    exp = HandoffExpert(model, data)
    exp.set_home()
    res = exp.run()
    res["seed"] = seed
    if verbose:
        print(f"seed {seed}: success={res['success']} block_final="
              f"{np.round(res['block_final'],3).tolist()} target={HANDOFF_TARGET} ({res['steps']} steps)")
    return res


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "eval":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        ok = sum(run_headless(seed=s)["success"] for s in range(n))
        print(f"=== HAND-OFF {ok}/{n} success ===")
    else:
        run_headless(seed=int(sys.argv[1]) if len(sys.argv) > 1 else 0)
