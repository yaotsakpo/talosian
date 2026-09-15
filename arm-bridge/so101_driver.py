"""So101Driver: the REAL SO-101 over LeRobot. On-site swap.

Imported and selected only when a physical arm is connected (it needs the
`lerobot` package and hardware). Structured as a thin translation from our vetted
trajectories to LeRobot's send_action, so bringing it online on-site is a small,
low-risk step: install lerobot, connect the arm, set ARM_DRIVER=so101.

The trajectory joint NAMES (shoulder_pan, shoulder_lift, elbow_flex, wrist_flex,
wrist_roll, gripper) match both the SO-101 MuJoCo model and LeRobot's SO-101
motor bus, so the mapping is by name with no re-indexing.

Verified on-site (untestable without hardware), which is exactly why the rest of
the system is fully exercised through MockDriver and SimDriver ahead of time.
"""

from __future__ import annotations

import math
import os
import time

from driver import ExecutorDriver, DriveResult
from trajectories import JOINTS


class So101Driver(ExecutorDriver):
    def __init__(self, port: str | None = None):
        # Import lazily so the module only requires lerobot when actually used.
        from lerobot.common.robot_devices.motors.feetech import FeetechMotorsBus

        port = port or os.environ.get("SO101_PORT", "/dev/ttyACM0")
        # Motor ids follow the SO-101 default chain, indexed by JOINTS order.
        motors = {name: (i + 1, "sts3215") for i, name in enumerate(JOINTS)}
        self.bus = FeetechMotorsBus(port=port, motors=motors)
        self.bus.connect()
        # Position control; leave torque enabled so the arm holds its pose.
        self._apply_home()

    def _apply_home(self):
        from trajectories import HOME
        self._write({name: HOME[name] for name in JOINTS})

    def _write(self, target_deg: dict):
        # LeRobot's Feetech bus takes degrees directly on the "Goal_Position"
        # register via write("Goal_Position", values, motor_names).
        names = [n for n in JOINTS if n in target_deg]
        values = [float(target_deg[n]) for n in names]
        self.bus.write("Goal_Position", values, names)

    def perform(self, action: str) -> DriveResult:
        traj = self._resolve(action)  # allowlist gate
        frames = traj.resolved()
        for target_deg, dwell in frames:
            self._write(target_deg)
            time.sleep(dwell)
        return DriveResult(
            action=action, performed=True,
            detail=f"SO-101 performed {action}", frames=len(frames),
        )

    def close(self):
        try:
            self.bus.disconnect()
        except Exception:
            pass
