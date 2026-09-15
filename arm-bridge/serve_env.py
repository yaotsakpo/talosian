"""Closed-loop serve environment for the TRAINED policy. The policy outputs joint
targets AND a per-arm grasp signal; this env turns the grasp signal into the weld
(a real gripper command: when the arm signals grasp and its gripper is near a
graspable object, the object is grasped; when the signal drops, it is released).

This is the piece that lets a trained VLA actually pick things up: in the expert
demonstrations the grasp is a weld toggled by Python, so a joint-only policy could
never reproduce it. Here the policy's learned grasp bit drives the weld, closing
the loop.

Grasp semantics (per arm):
  signal 1 and not holding: if the gripper site is within GRASP_RADIUS of a
    graspable object, weld that object to the gripper (canonical centred offset).
  signal 0 and holding: release (deactivate the weld).

Graspable objects: the free bodies with a grasp_* weld. Each arm can hold one.
"""

from __future__ import annotations

import numpy as np
import mujoco

from scene_serve import build_model, served, SARAH_SPOT, MALIK_SPOT, JOINTS
from setup_expert import SetupExpert

GRASP_RADIUS = 0.06   # m: how close the gripper must be to grab an object
GRASP_THRESH = 0.5    # policy grasp-signal threshold


class ServeEnv:
    def __init__(self, seed=0, drink="water"):
        self.model = build_model(seed=seed, sarah_drink=drink)
        self.data = mujoco.MjData(self.model)
        # reuse SetupExpert only for its grasp/weld mechanics + home
        self._m = SetupExpert(self.model, self.data)
        self._m.set_home()
        self.saddr = np.array([self.model.jnt_qposadr[mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{a}_{j}")]
            for a in ("H", "P") for j in JOINTS])
        self.act_ids = np.array([mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{a}_{j}")
            for a in ("H", "P") for j in JOINTS])
        # graspable objects and their welds (from the scene)
        self.objects = list(self._m.weld.keys())
        self.holding = {"H": None, "P": None}   # obj each arm currently holds

    def _gripper_xy(self, arm):
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, f"{arm}_gripperframe")
        return self.data.site_xpos[sid][:2].copy()

    def _obj_xy(self, obj):
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, obj)
        return self.data.xpos[bid][:2].copy()

    def _apply_grasp(self, arm, want_grasp):
        """Turn the policy's grasp signal into the weld for `arm`."""
        if want_grasp and self.holding[arm] is None:
            # grab the nearest graspable object within reach not already held
            gxy = self._gripper_xy(arm)
            held = set(v for v in self.holding.values() if v)
            cands = [(np.linalg.norm(self._obj_xy(o) - gxy), o)
                     for o in self.objects if o not in held]
            cands = [c for c in cands if c[0] <= GRASP_RADIUS]
            if cands:
                obj = min(cands)[1]
                self._m._grasp(arm, obj)   # reuse the canonical-offset weld
                self.holding[arm] = obj
        elif not want_grasp and self.holding[arm] is not None:
            self._m._release(self.holding[arm])
            self.holding[arm] = None

    def step_ctrl(self, joint_ctrl, grasp_bits):
        """Set joint targets and apply grasp signals, then step once."""
        for aid, v in zip(self.act_ids, joint_ctrl):
            lo, hi = self.model.actuator_ctrlrange[aid]
            self.data.ctrl[aid] = min(max(float(v), lo), hi)
        self._apply_grasp("H", grasp_bits[0] > GRASP_THRESH)
        self._apply_grasp("P", grasp_bits[1] > GRASP_THRESH)
        mujoco.mj_step(self.model, self.data)

    def state(self):
        return self.data.qpos[self.saddr].copy()

    def outcome(self):
        sarah = served(self.model, self.data, "sarah_glass", SARAH_SPOT)
        malik = served(self.model, self.data, "malik_wine_glass", MALIK_SPOT)
        return {"sarah_served": bool(sarah), "malik_served": bool(malik),
                "success": bool(sarah and malik)}
