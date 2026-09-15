"""The trust gate as a drop-in wrapper around a Physical AI Studio policy.

Studio runs a policy with a plain loop:

    from physicalai.inference import InferenceModel
    policy = InferenceModel("./policy")
    obs, info = env.reset()
    while not done:
        action = policy.select_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)

`select_action` is the boundary. What it returns becomes what the motors do. Nothing
between the two asks whether the claim behind that action has earned the authority to
move a part into a customer's bin.

GatedPolicy wraps the policy and answers that question at the boundary. It exposes the
same interface, so the loop above does not change at all:

    policy = GatedPolicy(InferenceModel("./policy"), evidence=anomaly_reading)

When the evidence does not support sorting, select_action returns a HOLD action instead
of the sorting action, so the arm does not carry the part to a bin. Nothing about the
trained policy changes; what changes is whether its output reaches the motors.
"""

from __future__ import annotations

from sort_gate import decide, BIN_REVIEW


class GatedPolicy:
    """Same interface as an InferenceModel, with authority established at the boundary.

    policy:        the wrapped policy (InferenceModel or anything with select_action).
    evidence:      callable returning the current evidence for the part in hand:
                   {verdict, confidence, calibrated, seen_before}.
    hold_action:   what to command when an action is withheld. None means "command
                   nothing", which the caller must treat as "do not move". A caller that
                   needs a concrete vector (some loops require one) can pass the arm's
                   current position, so the arm holds station rather than sorting.
    on_decision:   optional callback, for the dashboard and the audit record.
    protection_on: False reproduces today's behaviour (the policy's action reaching the
                   motors unchecked) so the difference can be shown, not asserted.
    """

    def __init__(self, policy, evidence, hold_action=None, on_decision=None,
                 protection_on=True):
        self._policy = policy
        self._evidence = evidence
        self._hold_action = hold_action
        self._on_decision = on_decision
        self.protection_on = protection_on
        self.last_decision = None

    def reset(self, *a, **kw):
        self.last_decision = None
        if hasattr(self._policy, "reset"):
            return self._policy.reset(*a, **kw)

    def select_action(self, observation, *a, **kw):
        action = self._policy.select_action(observation, *a, **kw)

        decision = decide(self._evidence() or {}, protection_on=self.protection_on)
        self.last_decision = decision
        if self._on_decision is not None:
            self._on_decision(decision)

        if decision["allowed"]:
            return action

        # Withheld. The part is not carried to a customer bin; it is held for review.
        return self._hold_action

    def __getattr__(self, name):
        # anything else this policy exposes stays reachable
        return getattr(self._policy, name)
