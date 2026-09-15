"""The trust gate, wrapping the policy that drives the arm.

THE PIPELINE THIS GOVERNS
    ACT        picks a cube up
    Anomalib   judges it good or defective, with a score
    SmolVLA    places it in a bin

Only the third step is governed. Picking a cube up to look at it is exactly what should
happen regardless of confidence, so the grasp passes through untouched. What needs
authority is the PLACEMENT: moving a cube into a customer bin on the strength of a
judgement.

HOW IT KNOWS WHICH IS WHICH
    Studio holds ONE policy at a time (`self._policy` in StudioActionSource, replaced on
    each load) and switches what it is doing with a task string: StartTaskCommand ->
    policy.set_task(task). So the phase is readable from the task, and this wrapper
    watches set_task to know whether the arm is currently picking or placing.

WHAT A HELD CUBE DOES
    Anomalib decides after ACT already has the cube, so "hold" cannot mean "freeze": the
    arm is holding something and has to put it somewhere. There are two bins, blue for
    good and pink for bad, and neither is right for a cube nobody is sure about. So a
    held cube is PUT BACK where it came from.

    That has a useful property. The arm works through the pile taking only what it is
    sure about. What remains in the input area is exactly the set the model could not
    commit to: the gate's output is a physical pile, and its size measures where the
    model lacks competence on this batch.

    (It measures uncertainty, not defectiveness. The residue mixes genuinely ambiguous
    defects with good cubes that photographed badly, and claiming otherwise would be
    wrong.)
"""

from __future__ import annotations

from sort_gate import decide

# Task substrings that mean "this action puts the cube somewhere". Only these are
# governed. Anything else (approaching, grasping, lifting, going home) passes through.
PLACEMENT_HINTS = ("place", "put", "bin", "drop", "sort")

# The task handed to the policy to return a held cube to the input area.
RETURN_TASK = "put the cube back where it came from"


def is_placement(task: str | None) -> bool:
    """Is the policy currently being asked to put the cube somewhere?"""
    if not task:
        return False
    low = task.lower()
    return any(h in low for h in PLACEMENT_HINTS)


class GatedSource:
    """Wraps Studio's PolicySource so placements require evidence, grasps do not.

    inner:         the PolicySource Studio loaded.
    evidence:      callable returning the current cube's reading:
                   {verdict, confidence, calibrated, seen_before}.
    on_decision:   optional callback, for the dashboard and the audit record.
    protection_on: False reproduces today's pipeline, the claim reaching the motors
                   unchecked, so the difference can be shown rather than asserted.
    """

    def __init__(self, inner, evidence, on_decision=None, protection_on=True):
        self._inner = inner
        self._evidence = evidence
        self._on_decision = on_decision
        self.protection_on = protection_on
        self.last_decision = None
        self._task = None
        self._redirected = False        # this cube is already on its way back

    # Studio calls this when the phase changes; it is how we know pick from place.
    def set_task(self, task, *a, **kw):
        if task != self._task:
            self._redirected = False    # a new instruction means a new cube
        self._task = task
        return self._inner.set_task(task, *a, **kw)

    def update(self, robot_state, camera_frames, step):
        action = self._inner.update(robot_state, camera_frames, step)
        if action is None:
            return None

        # Not a placement: the arm is approaching, grasping or lifting. Never gated.
        if not is_placement(self._task):
            return action

        # Already redirected this cube: let the return motion finish uninterrupted.
        if self._redirected:
            return action

        decision = decide(self._evidence() or {}, protection_on=self.protection_on)
        self.last_decision = decision
        if self._on_decision is not None:
            self._on_decision(decision)

        if decision["allowed"]:
            return action

        # Held. Send the cube back to the input area instead of to a bin, and let that
        # motion run to completion rather than re-deciding every tick.
        self._redirected = True
        self._inner.set_task(RETURN_TASK)
        self._task = RETURN_TASK
        return self._inner.update(robot_state, camera_frames, step)

    def __getattr__(self, name):
        return getattr(self._inner, name)
