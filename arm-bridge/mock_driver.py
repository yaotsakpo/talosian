"""MockDriver: no hardware, no physics. Logs each action and the trajectory it
would issue. Used for tests/CI and as the always-available fallback. Exercises
the same allowlist gate as the real drivers, so a refused action is refused
here too."""

from __future__ import annotations

from driver import ExecutorDriver, DriveResult


class MockDriver(ExecutorDriver):
    def __init__(self, log=print):
        self._log = log
        self.performed = []  # transcript for tests

    def perform(self, action: str) -> DriveResult:
        traj = self._resolve(action)  # allowlist gate
        frames = traj.resolved()
        self.performed.append(action)
        self._log(f"[mock] {action}: {len(frames)} frames (home -> ... -> home)")
        return DriveResult(
            action=action, performed=True,
            detail=f"mock performed {action}", frames=len(frames),
        )
