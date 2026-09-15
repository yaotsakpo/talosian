"""The trust gate as a wrapper around Studio's PolicySource.

Read from Studio's own backend (application/backend/src/runtime/action_source.py),
rather than from documentation. StudioActionSource.update() ends like this:

    policy_action = self._policy_action(robot_state, camera_frames, step)

    if self._follower_source == "teleop" and leader_action is not None:
        return leader_action
    if self._follower_source == "policy" and policy_action is not None:
        return policy_action
    return self._hold_target.copy()

Two things follow, and they shape this whole file:

1. The seam is `PolicySource.update(robot_state, camera_frames, step)`. Whatever it
   returns becomes the joint command for that tick.

2. **Studio already knows how to hold.** If the policy action is None, the runtime
   commands `self._hold_target`, the pose captured from the robot's own state, so the
   arm holds station. Studio does this when a policy errors (`_drop_policy_to_hold`).

So "skip the placement" does not need inventing: returning None is already the way to
say it, and the runtime does the safe thing. What is missing is anything that decides
WHETHER the action has earned the right to be sent. That is this gate.
"""

from __future__ import annotations

from sort_gate import decide


class GatedSource:
    """Wraps a PolicySource so its actions pass the trust gate before reaching the arm.

    inner:         the PolicySource being governed (what Studio loaded).
    evidence:      callable returning the current part's evidence:
                   {verdict, confidence, calibrated, seen_before}.
    on_decision:   optional callback, for the dashboard and the audit record.
    protection_on: False reproduces today's behaviour, the policy's action reaching the
                   motors unchecked, so the difference can be shown rather than asserted.

    Use it where Studio builds the policy source:

        source = GatedSource(inner=loaded_policy_source, evidence=anomaly_reading)

    Returning None on a held part is deliberate: the runtime already turns that into
    holding station, which is exactly the third outcome. The part is not carried to
    either bin.
    """

    def __init__(self, inner, evidence, on_decision=None, protection_on=True):
        self._inner = inner
        self._evidence = evidence
        self._on_decision = on_decision
        self.protection_on = protection_on
        self.last_decision = None

    def update(self, robot_state, camera_frames, step):
        action = self._inner.update(robot_state, camera_frames, step)
        if action is None:
            return None                      # the policy had nothing; runtime holds

        decision = decide(self._evidence() or {}, protection_on=self.protection_on)
        self.last_decision = decision
        if self._on_decision is not None:
            self._on_decision(decision)

        if decision["allowed"]:
            return action

        # Held. Returning None makes the runtime command the hold target, so the arm
        # stays where it is instead of carrying the part to a bin.
        return None

    def __getattr__(self, name):
        # stop(), close(), anything else Studio calls on a policy source
        return getattr(self._inner, name)
