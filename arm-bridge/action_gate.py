"""Action gate (trust boundary 3): the LAST verification before an action reaches
the arm executor. Every joint command is checked against the authorized task's
SAFETY ENVELOPE; anything outside it is blocked and the arm holds its last safe
pose instead. So a drifted policy action, or a command that slipped past the input
gate, cannot produce unauthorized physical motion.

This is our principle at the ACTION boundary: an action is not trusted because of
WHERE it came from (the policy, the expert, the network); it is trusted only if it
is within the authorized envelope, checked right before the executor.

Two cheap, model-free checks (no privileged oracle):
  1. RANGE: every joint target within the actuator ctrl range (a NaN / wild value
     is blocked).
  2. RATE: the per-step change of each joint is bounded (MAX_DELTA). A sudden jump
     (a spoofed or diverged action) is blocked, so the arm cannot lurch. This is
     the physical-safety analogue of "no unproven command executes".

Reports how many actions were blocked, so the demo can show the gate catching
injected bad actions.
"""

from __future__ import annotations

import numpy as np


class ActionGate:
    def __init__(self, model, act_ids, max_delta=0.6):
        self.model = model
        self.act_ids = np.asarray(act_ids)
        self.max_delta = max_delta   # rad, max allowed per-control-step joint change
        self.lo = np.array([model.actuator_ctrlrange[a][0] for a in self.act_ids])
        self.hi = np.array([model.actuator_ctrlrange[a][1] for a in self.act_ids])
        self._last = None
        self.checked = 0
        self.blocked = 0

    def filter(self, ctrl):
        """Return an action safe to execute. If `ctrl` is out of range or changes
        too fast, block it (hold the last safe action) and count the block."""
        ctrl = np.asarray(ctrl, dtype=float)
        self.checked += 1
        bad = False
        # 1. finite + in range
        if not np.all(np.isfinite(ctrl)):
            bad = True
        else:
            clamped = np.clip(ctrl, self.lo, self.hi)
            if np.any(np.abs(clamped - ctrl) > 1e-6):
                bad = True          # out of actuator range
            ctrl = clamped
        # 2. rate limit vs last authorized action
        if not bad and self._last is not None:
            if np.any(np.abs(ctrl - self._last) > self.max_delta):
                bad = True          # too large a jump -> block
        if bad:
            self.blocked += 1
            return self._last if self._last is not None else np.clip(np.zeros_like(ctrl), self.lo, self.hi)
        self._last = ctrl
        return ctrl

    def report(self):
        return {"checked": self.checked, "blocked": self.blocked,
                "block_rate": round(self.blocked / max(self.checked, 1), 3)}


if __name__ == "__main__":
    # demo: the gate blocks an injected wild action while passing safe ones
    import mujoco
    from scene_serve import build_model, JOINTS
    m = build_model(seed=0, sarah_drink="water")
    act_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{a}_{j}")
               for a in ("H", "P") for j in JOINTS]
    g = ActionGate(m, act_ids, max_delta=0.6)
    safe = np.zeros(12)
    print("safe action     ->", "passed" if np.allclose(g.filter(safe), safe) else "blocked")
    small = safe + 0.1
    print("small step      ->", "passed" if np.allclose(g.filter(small), small) else "blocked")
    wild = safe + 5.0          # a diverged/spoofed action
    out = g.filter(wild)
    print("wild jump       ->", "blocked (held last safe)" if not np.allclose(out, wild) else "passed")
    nan = np.full(12, np.nan)
    out = g.filter(nan)
    print("NaN action      ->", "blocked" if np.all(np.isfinite(out)) else "passed")
    print("report:", g.report())
