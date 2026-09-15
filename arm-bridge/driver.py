"""Executor driver interface.

The one boundary between Talosian's trust decision (which happens in the
browser, with real continuity crypto) and the physical SO-101 arm. The gate
decides ACT vs HOLD; only an ACT ever reaches a driver. A driver's single job
is to move the arm through a named action's trajectory, or refuse if the action
is not on the physical allowlist.

Two implementations:
  - MockDriver   : logs the action + trajectory, no hardware. Runs anywhere,
                   used for development and the tests. This is what runs until
                   the arm is physically present.
  - So101Driver  : real LeRobot send_action against an SO-101. Imported and
                   selected only on-site (it needs the `lerobot` package and a
                   connected arm). Structured as a thin swap so the rest of the
                   system is fully exercised ahead of time.

SAFETY (non-negotiable, mirrors the design spec):
  - The action registry is an ALLOWLIST. A driver executes ONLY named actions
    with a vetted trajectory. An unknown action is refused, never improvised.
    This is a third safety layer independent of continuity and scope.
  - The Talosian protection toggle gates DELIVERY of commands, never physical
    safety. It must never disable an emergency stop or a motion-safety limit.
  - Trajectories are bounded, slow, and collision-safe by construction (small
    joint deltas, a return-to-home after each action).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from trajectories import Trajectory, get_trajectory, KNOWN_ACTIONS


@dataclass
class DriveResult:
    """Outcome of asking a driver to perform an action."""

    action: str
    performed: bool
    detail: str
    # The trajectory frames actually issued (for logging / the mock's transcript).
    frames: int


class ExecutorDriver(ABC):
    """A physical (or simulated) arm the executor drives. One method that
    matters: perform(action). Refuses anything off the allowlist."""

    @abstractmethod
    def perform(self, action: str) -> DriveResult:  # pragma: no cover - interface
        ...

    def _resolve(self, action: str) -> Trajectory:
        """Shared allowlist gate. Raises for an unknown action so no driver can
        move the arm for something that was never vetted."""
        if action not in KNOWN_ACTIONS:
            raise ValueError(
                f"action {action!r} is not on the physical allowlist "
                f"{sorted(KNOWN_ACTIONS)!r}; refusing to move the arm"
            )
        return get_trajectory(action)
