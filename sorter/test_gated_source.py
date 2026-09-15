"""Tests for the gate as it wraps Studio's policy.

The contract: the grasp is never governed, the placement is, and a held cube goes back
to the input area rather than into either bin.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from gated_source import RETURN_TASK, GatedSource, is_placement

MOTION = [9.0] * 6


class FakeSource:
    """Stands in for the PolicySource Studio loaded."""

    def __init__(self, action=MOTION):
        self.action = action
        self.task = None
        self.tasks_set = []
        self.updates = 0

    def set_task(self, task):
        self.task = task
        self.tasks_set.append(task)

    def update(self, robot_state, camera_frames, step):
        self.updates += 1
        return self.action

    def stop(self):
        return "stopped"


def ev(confidence=0.95, verdict="good", calibrated=True):
    return {"verdict": verdict, "confidence": confidence, "calibrated": calibrated}


def gated(src=None, conf=0.95, **kw):
    return GatedSource(src or FakeSource(), evidence=lambda: ev(conf), **kw)


# --- which phase is which -------------------------------------------------------

def test_placement_tasks_are_recognised():
    for t in ("place the cube in the blue bin", "put it in the pink bin",
              "drop the cube", "sort the cube"):
        assert is_placement(t)


def test_grasp_tasks_are_not_placements():
    for t in ("pick up the cube", "approach the cube", "go home", None, ""):
        assert not is_placement(t)


# --- the grasp is never governed ------------------------------------------------

def test_a_grasp_is_never_gated_even_when_uncertain():
    """Picking a cube up to look at it should happen regardless of confidence."""
    g = gated(conf=0.10)
    g.set_task("pick up the cube")
    assert g.update({}, {}, 0) == MOTION
    assert g.last_decision is None           # no decision was even required


# --- the placement is governed --------------------------------------------------

def test_a_confident_placement_proceeds():
    g = gated(conf=0.97)
    g.set_task("place the cube in the blue bin")
    assert g.update({}, {}, 0) == MOTION
    assert g.last_decision["allowed"]


def test_an_uncertain_placement_is_redirected_to_the_input_area():
    src = FakeSource()
    g = gated(src, conf=0.55)
    g.set_task("place the cube in the blue bin")
    g.update({}, {}, 0)
    assert g.last_decision["allowed"] is False
    assert src.tasks_set[-1] == RETURN_TASK   # sent back, not to either bin


def test_the_return_motion_is_not_re_decided_every_tick():
    """Once redirected, the cube goes back without the gate second-guessing itself."""
    src = FakeSource()
    g = gated(src, conf=0.55)
    g.set_task("place the cube in the blue bin")
    g.update({}, {}, 0)
    before = len(src.tasks_set)
    g.update({}, {}, 1)
    g.update({}, {}, 2)
    assert len(src.tasks_set) == before       # not redirected again


def test_a_new_cube_gets_a_fresh_decision():
    src = FakeSource()
    g = gated(src, conf=0.55)
    g.set_task("place the cube in the blue bin")
    g.update({}, {}, 0)
    g.set_task("pick up the cube")            # next cube
    g.set_task("place the cube in the pink bin")
    g.update({}, {}, 1)
    assert src.tasks_set.count(RETURN_TASK) == 2


def test_protection_off_places_the_same_cube():
    """The comparison the demo rests on."""
    spoof = {"verdict": "good", "confidence": 0.62, "calibrated": False}
    for on, expect_return in ((False, False), (True, True)):
        src = FakeSource()
        g = GatedSource(src, evidence=lambda: spoof, protection_on=on)
        g.set_task("place the cube in the blue bin")
        g.update({}, {}, 0)
        assert (RETURN_TASK in src.tasks_set) is expect_return


# --- plumbing -------------------------------------------------------------------

def test_a_policy_with_nothing_to_say_passes_through():
    g = gated(FakeSource(action=None))
    g.set_task("place the cube in the blue bin")
    assert g.update({}, {}, 0) is None


def test_every_placement_decision_is_recorded():
    seen = []
    g = gated(conf=0.55, on_decision=seen.append)
    g.set_task("place the cube in the blue bin")
    g.update({}, {}, 0)
    assert len(seen) == 1 and seen[0]["allowed"] is False


def test_the_wrapper_is_transparent():
    assert gated().stop() == "stopped"
