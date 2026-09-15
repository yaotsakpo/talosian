"""Tests for the gate as it attaches to Studio's runtime.

Mirrors Studio's own test style (application/backend/tests/runtime/test_action_source.py):
a fake source stands in for the loaded PolicySource, and the contract under test is what
`update(robot_state, camera_frames, step)` returns.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from gated_source import GatedSource

SORT = [9.0, 9.0, 9.0, 9.0, 9.0, 9.0]        # the placement motion


class FakeSource:
    """Stands in for the PolicySource Studio loaded."""

    def __init__(self, action=SORT):
        self.action = action
        self.calls = 0

    def update(self, robot_state, camera_frames, step):
        self.calls += 1
        return self.action

    def stop(self):
        return "stopped"


def ev(confidence=0.95, verdict="good", calibrated=True):
    return {"verdict": verdict, "confidence": confidence, "calibrated": calibrated}


def test_a_supported_action_reaches_the_arm():
    g = GatedSource(FakeSource(), evidence=lambda: ev(0.97))
    assert g.update({}, {}, 0) == SORT


def test_a_held_part_returns_none_so_the_runtime_holds():
    """Studio's runtime commands its latched hold target when the action is None, so
    returning None IS the way to say 'skip this placement'."""
    g = GatedSource(FakeSource(), evidence=lambda: ev(0.60))
    assert g.update({}, {}, 0) is None


def test_the_policy_still_runs_when_held():
    """The gate governs whether the action is sent, not whether the policy thinks. The
    policy is not disabled; its output is simply not granted authority."""
    inner = FakeSource()
    g = GatedSource(inner, evidence=lambda: ev(0.60))
    g.update({}, {}, 0)
    assert inner.calls == 1


def test_protection_off_sends_the_same_action():
    spoof = ev(0.62, calibrated=False)
    off = GatedSource(FakeSource(), evidence=lambda: spoof, protection_on=False)
    on = GatedSource(FakeSource(), evidence=lambda: spoof, protection_on=True)
    assert off.update({}, {}, 0) == SORT
    assert on.update({}, {}, 0) is None


def test_a_policy_with_nothing_to_say_passes_through():
    g = GatedSource(FakeSource(action=None), evidence=lambda: ev(0.99))
    assert g.update({}, {}, 0) is None


def test_every_decision_is_recorded():
    seen = []
    g = GatedSource(FakeSource(), evidence=lambda: ev(0.60), on_decision=seen.append)
    g.update({}, {}, 0)
    assert len(seen) == 1 and seen[0]["allowed"] is False


def test_the_wrapper_is_transparent():
    """Studio calls other methods on the source; they must still work."""
    assert GatedSource(FakeSource(), evidence=lambda: ev()).stop() == "stopped"
