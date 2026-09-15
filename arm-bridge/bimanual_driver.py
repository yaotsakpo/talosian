"""Bimanual sim driver: two SO-101 arms doing the SAME simple gestures the single
arm does (wave, nod, bow, point, collect, dance), one per arm. No grasping, no
manipulation, just the gestures we already have, on two arms.

The two arms are principals in the coordination story (H = one, P = the other),
but physically they only ever perform the same safe, legible gestures. The trust
demo is about which arm's MESSAGE is trusted, not about dexterity.
"""

from __future__ import annotations

import math
import threading
import time

import mujoco

import numpy as np

from scene_bimanual import build_model
# reuse the single-arm gesture trajectories, they already work and look good
from trajectories import JOINTS, HOME, get_trajectory, KNOWN_ACTIONS
from ik import ArmIK


class BimanualDriver:
    def __init__(self):
        self.model = build_model()
        self.data = mujoco.MjData(self.model)
        self.act = {"H": {}, "P": {}}
        for arm in ("H", "P"):
            for j in JOINTS:
                aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{arm}_{j}")
                if aid < 0:
                    raise RuntimeError(f"actuator {arm}_{j} missing")
                self.act[arm][j] = aid
        self._lock = threading.Lock()
        # IK per arm (to reach the clap meeting point).
        self.ik = {"H": ArmIK(self.model, "H_", "H_gripperframe"),
                   "P": ArmIK(self.model, "P_", "P_gripperframe")}
        self._home()

    def _home(self):
        for arm in ("H", "P"):
            for j, deg in HOME.items():
                self.data.ctrl[self.act[arm][j]] = math.radians(deg)

    def _ramp(self, arm: str, pose: dict, dwell: float):
        model = self.model
        steps = max(1, int(dwell / model.opt.timestep))
        start = {j: self.data.ctrl[self.act[arm][j]] for j in JOINTS}
        goal = {}
        for j, aid in self.act[arm].items():
            if j in pose:
                lo, hi = model.actuator_ctrlrange[aid]
                goal[j] = min(max(math.radians(pose[j]), lo), hi)
            else:
                goal[j] = start[j]
        for i in range(1, steps + 1):
            f = i / steps
            with self._lock:
                for j, aid in self.act[arm].items():
                    self.data.ctrl[aid] = start[j] + (goal[j] - start[j]) * f
            time.sleep(model.opt.timestep)

    def perform(self, action: str, arm: str = "H", log=print):
        """One arm performs one gesture (the same gestures the single arm does)."""
        if action not in KNOWN_ACTIONS:
            raise ValueError(f"{action} not on the allowlist {sorted(KNOWN_ACTIONS)}")
        if arm not in self.act:
            raise ValueError(f"unknown arm {arm}")
        log(f"[{arm}] {action}")
        for pose, dwell in get_trajectory(action).resolved():
            self._ramp(arm, pose, dwell)

    def both(self, action: str, log=print):
        """Both arms do the gesture together (e.g. both wave)."""
        tH = threading.Thread(target=self.perform, args=(action, "H", lambda *a: None))
        tP = threading.Thread(target=self.perform, args=(action, "P", lambda *a: None))
        log(f"[H+P] {action}")
        tH.start(); tP.start(); tH.join(); tP.join()

    # ---- the coordinated CLAP -------------------------------------------
    # Clapping needs BOTH arms: the two grippers swing in to meet at the center,
    # then apart, together. A successful clap = coordination held. A MISSED clap
    # (one arm sent to the wrong place by a spoof) = coordination broken, the
    # hands do not meet.
    def clap(self, coordinated: bool, times: int = 3, log=print):
        """coordinated=True: both arms meet at center and clap. coordinated=False:
        Agent B was spoofed to the wrong spot, so the clap misses (they don't meet)."""
        cx, cy, cz = 0.0, -0.02, 0.24  # the meeting point (a bit lower, in front)
        meet_H = np.array([cx - 0.02, cy, cz])   # H's hand just left of center
        meet_P = np.array([cx + 0.02, cy, cz])   # P's hand just right of center
        # where each hand goes when apart (back toward its own side)
        apart_H = np.array([-0.12, cy, cz + 0.02])
        apart_P = np.array([0.12, cy, cz + 0.02])
        # if not coordinated, B (P) is sent OFF target so the hands miss
        miss_P = np.array([0.22, 0.10, cz + 0.05])
        log(f"[clap] coordinated={coordinated}")
        for n in range(times):
            tH = threading.Thread(target=self._move_to, args=("H", meet_H, 0.35))
            tP = threading.Thread(target=self._move_to,
                                  args=("P", meet_P if coordinated else miss_P, 0.35))
            tH.start(); tP.start(); tH.join(); tP.join()
            tH = threading.Thread(target=self._move_to, args=("H", apart_H, 0.35))
            tP = threading.Thread(target=self._move_to, args=("P", apart_P, 0.35))
            tH.start(); tP.start(); tH.join(); tP.join()
        # return both home
        tH = threading.Thread(target=self._ramp, args=("H", dict(HOME), 0.5))
        tP = threading.Thread(target=self._ramp, args=("P", dict(HOME), 0.5))
        tH.start(); tP.start(); tH.join(); tP.join()

    def _move_to(self, arm, xyz, dwell=1.1, wrist_flex=None):
        with self._lock:
            sol = self.ik[arm].solve(self.data, np.array(xyz))
        if wrist_flex is not None:
            sol["wrist_flex"] = wrist_flex
        self._ramp(arm, sol, dwell)
