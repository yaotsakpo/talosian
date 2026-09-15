"""SimDriver: drive a REAL simulated SO-101 in MuJoCo.

Loads the robotstudio_so101 model (mujoco_menagerie) and moves it through the
same vetted trajectories the physical So101Driver would use. This lets the whole
Talosian pipeline run end-to-end, browser -> bridge -> a 3D arm that actually
moves, with no hardware. It is real physics on the real robot model, not a log.

The model's actuators are position-controlled (ctrl = target radians). perform()
walks the trajectory frames, ramping each actuator toward the frame's target and
stepping the physics, so motion is smooth rather than teleporting.

A viewer window (mujoco.viewer) shows the arm when SIM_VIEWER=1 (default on).
Set SIM_VIEWER=0 for headless (CI / no display), where it still steps physics so
the pipeline is exercised.
"""

from __future__ import annotations

import math
import os
import threading
import time

import mujoco

from driver import ExecutorDriver, DriveResult
from trajectories import JOINTS

# Model path, resolved relative to this file so it works from any CWD.
_HERE = os.path.dirname(os.path.abspath(__file__))
_MODEL = os.path.join(_HERE, "menagerie", "robotstudio_so101", "scene_box.xml")
_MODEL_FALLBACK = os.path.join(_HERE, "menagerie", "robotstudio_so101", "scene.xml")


class SimDriver(ExecutorDriver):
    def __init__(self, viewer: bool | None = None):
        path = _MODEL if os.path.exists(_MODEL) else _MODEL_FALLBACK
        self.model = mujoco.MjModel.from_xml_path(path)
        self.data = mujoco.MjData(self.model)
        # Map our joint names -> actuator ids (same names in this model).
        self._act = {}
        for name in JOINTS:
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if aid < 0:
                raise RuntimeError(f"actuator {name!r} missing from the SO-101 model")
            self._act[name] = aid
        self._lock = threading.Lock()
        self._want_viewer = (
            viewer if viewer is not None else os.environ.get("SIM_VIEWER", "1") == "1"
        )
        self._viewer = None
        self._running = True
        # Settle at home so the first command starts from a known pose.
        self._apply_home()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    # ---- physics loop (background) --------------------------------------
    def _apply_home(self):
        from trajectories import HOME
        for name, deg in HOME.items():
            self.data.ctrl[self._act[name]] = math.radians(deg)

    def _loop(self):
        if self._want_viewer:
            try:
                from mujoco import viewer as _mj_viewer
                self._viewer = _mj_viewer.launch_passive(self.model, self.data)
            except Exception as e:  # no display / headless
                print(f"[sim] viewer unavailable ({e}); running headless")
                self._viewer = None
        dt = self.model.opt.timestep
        while self._running:
            with self._lock:
                mujoco.mj_step(self.model, self.data)
            if self._viewer is not None:
                if not self._viewer.is_running():
                    self._running = False
                    break
                self._viewer.sync()
            time.sleep(dt)

    # ---- driver interface ------------------------------------------------
    def perform(self, action: str) -> DriveResult:
        traj = self._resolve(action)  # allowlist gate (raises if unknown)
        frames = traj.resolved()
        issued = 0
        for target_deg, dwell in frames:
            self._ramp_to(target_deg, dwell)
            issued += 1
        return DriveResult(
            action=action, performed=True,
            detail=f"sim SO-101 performed {action} ({issued} frames)", frames=issued,
        )

    def _ramp_to(self, target_deg: dict, dwell_s: float):
        """Ramp actuators from current ctrl to the target over ~dwell, so the
        motion is smooth. Targets are clamped to each actuator's ctrl range."""
        steps = max(1, int(dwell_s / max(self.model.opt.timestep, 1e-3)))
        start = {n: self.data.ctrl[a] for n, a in self._act.items()}
        goal = {}
        for name, aid in self._act.items():
            if name in target_deg:
                rad = math.radians(target_deg[name])
                lo, hi = self.model.actuator_ctrlrange[aid]
                goal[name] = min(max(rad, lo), hi)
            else:
                goal[name] = start[name]
        for i in range(1, steps + 1):
            f = i / steps
            with self._lock:
                for name, aid in self._act.items():
                    self.data.ctrl[aid] = start[name] + (goal[name] - start[name]) * f
            time.sleep(self.model.opt.timestep)

    def close(self):
        self._running = False
        if self._viewer is not None:
            try:
                self._viewer.close()
            except Exception:
                pass
