"""A gated ActionSource for the OpenVINO Physical AI runtime.

The runtime is assembled like this:

    runtime = RobotRuntime(
        fps=30,
        robot=SO101(port="/dev/ttyACM0"),
        action_source=PolicySource(model=InferenceModel("./exports/act_policy")),
        cameras={...},
    )

`action_source` is a documented extension point: it takes anything implementing the
ActionSource protocol, and the runtime pulls an action from it every tick and hands it
to the robot's send_action. So the boundary between "what the policy wants" and "what
the motors do" is a seam we can stand in, without forking the framework.

Intel describes the framework as having "future-ready hooks for action clamps and
emergency stops, guarding against unsafe movements caused by model errors or unexpected
inputs". The seam exists. The policy that belongs in it is not supplied. This is that
policy: a claim proposes, only a proof establishes.

    GatedSource(inner=PolicySource(...), evidence=lambda: anomaly_reading())

wraps any existing source. When the evidence behind the current part does not support
moving it to a customer bin, the sorting action is withheld and the arm holds instead.
Nothing about the underlying policy changes; what changes is whether its output has the
authority to reach the motors.
"""

from __future__ import annotations

from sort_gate import decide, BIN_REVIEW


class GatedSource:
    """Wraps an ActionSource so its actions pass through the trust gate first.

    inner:    the ActionSource being governed (e.g. PolicySource(model=...)).
    evidence: a callable returning the current evidence dict for the part in hand
              (verdict, confidence, calibrated, seen_before). Called per decision, not
              per tick, so perception can run at its own rate.
    on_hold:  optional callable invoked when an action is withheld, for the dashboard
              and the audit record.
    protection_on: False reproduces today's behaviour, the policy's output reaching the
              motors unchecked, so the difference can be demonstrated rather than claimed.
    """

    def __init__(self, inner, evidence, on_hold=None, protection_on=True):
        self._inner = inner
        self._evidence = evidence
        self._on_hold = on_hold
        self.protection_on = protection_on
        self.last_decision = None

    def reset(self):
        self.last_decision = None
        if hasattr(self._inner, "reset"):
            self._inner.reset()

    def get_action(self, observation):
        """Return an action for this tick, or None to withhold motion.

        The runtime asks the inner source what it wants to do. Before that reaches the
        robot, the gate asks whether the evidence supports doing it. A withheld action
        is not a modified action: the arm is simply not commanded to sort, which is the
        same property the restaurant bridge has, where a held request queues no motion.
        """
        action = self._pull(observation)
        if action is None:
            return None

        decision = decide(self._evidence() or {}, protection_on=self.protection_on)
        self.last_decision = decision

        if decision["allowed"]:
            return action

        if self._on_hold is not None:
            self._on_hold(decision)
        return None                      # withhold: no sorting motion at all

    def _pull(self, observation):
        """Get an action from the wrapped source, whatever method it exposes."""
        for name in ("get_action", "select_action", "__call__"):
            fn = getattr(self._inner, name, None)
            if callable(fn):
                return fn(observation)
        raise TypeError(f"{self._inner!r} does not look like an ActionSource")
