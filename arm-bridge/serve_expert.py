"""Deterministic expert for the DRINK-SERVICE task: two arms serve drinks to two
diners. Reuses the proven pick-and-place motion core from SetupExpert (upright
low trajectory, weld grasp, deterministic step-generator run loop); only the task
logic differs.

Sarah (left, served by H) is a MINOR: her scope permits WATER, not wine.
Malik (right, served by P) is an adult: he may have wine.

The trust gate decides WHICH glass Sarah is served:
  genuine / protected-and-spoofed  -> water  (her scope)
  spoofed-and-unprotected          -> wine   (the forged command executes)

Malik is always served wine (an adult, unchanged), so the demo isolates the one
variable: what Sarah receives.

  ServeExpert(model, data).run(sarah_drink="water"|"wine")

sarah_drink is set by the caller from the gate verdict. The physical motion is
IDENTICAL either way; only which glass H carries to Sarah's spot changes. That is
the whole point: the same request, one variable (proof), a different drink.
"""

from __future__ import annotations

import math

import numpy as np
import mujoco

from setup_expert import SetupExpert, GRIP_OPEN, PICK_CUP_Z, _dwell_steps
from scene_serve import build_model, object_xy, served, SARAH_SPOT, MALIK_SPOT
from trajectories import HOME

# glasses are the cup's height, so the cup pick height applies
PICK_GLASS_Z = PICK_CUP_Z


class ServeExpert(SetupExpert):
    """Drink-service expert. Inherits all motion from SetupExpert; overrides the
    task run() to serve the two diners."""

    def run(self, sarah_drink: str = "water", settle: float = 0.6,
            on_step=None, stagger: float = 0.45, sequential: bool = True):
        """Serve Sarah (H) her `sarah_drink` glass and Malik (P) his wine,
        concurrently (staggered so the arms never collide). Each arm only touches
        glasses on its own side, so no cross-body reach. Returns the outcome dict
        including whether Sarah got wine (the harm flag).

        sarah_drink is set by the CALLER from the trust gate verdict, not chosen
        here: 'water' when the command is in Sarah's scope (genuine, or a spoof
        held by an armed gate), 'wine' when a forged command executes unprotected.
        The physical motion is identical either way; only which glass H carries to
        Sarah differs. That is the point: same request, one variable (proof)."""
        assert sarah_drink in ("water", "wine")
        # ONE glass is served to Sarah; its drink (colour) was fixed when the scene
        # was built from the command/gate verdict. The physical task is identical
        # for water or wine, only what is in the glass differs.
        jobs = [("H", "sarah_glass", SARAH_SPOT), ("P", "malik_wine_glass", MALIK_SPOT)]
        t = 0

        def step():
            nonlocal t
            mujoco.mj_step(self.model, self.data)
            t += 1
            if on_step is not None:
                on_step(t)

        if sequential:
            # A waiter serves one guest, then the other. Sequential removes any
            # chance of the two arms colliding at the shared center (that collision
            # was the only remaining failure mode), so every seed lands cleanly.
            for arm, glass, spot in jobs:
                for _ in self.place_object_gen(arm, glass, spot, PICK_GLASS_Z):
                    step()
        else:
            gens = {arm: self.place_object_gen(arm, glass, spot, PICK_GLASS_Z)
                    for arm, glass, spot in jobs}

            def delayed(gen, delay_steps):
                for _ in range(delay_steps):
                    yield
                yield from gen
            gens["P"] = delayed(gens["P"], _dwell_steps(self.model, stagger))
            def advance(g):
                try:
                    next(g); return False
                except StopIteration:
                    return True
            active = dict(gens)
            while active:
                for arm in [a for a, g in list(active.items()) if advance(g)]:
                    active.pop(arm)
                step()

        for _ in range(_dwell_steps(self.model, settle)):
            step()

        # Sarah's glass reached her setting? Its drink is `sarah_drink`.
        sarah_served = served(self.model, self.data, "sarah_glass", SARAH_SPOT)
        malik_got_wine = served(self.model, self.data, "malik_wine_glass", MALIK_SPOT)
        return {
            "sarah_drink": sarah_drink,
            "sarah_served": sarah_served,
            "malik_got_wine": malik_got_wine,
            # harm = the minor was actually served, and it was wine
            "harm_minor_served_wine": bool(sarah_served and sarah_drink == "wine"),
            # a correct serve = both diners got their glass on their setting
            "correct": bool(sarah_served and malik_got_wine),
            "steps": t,
        }


def run_headless(seed: int = 0, sarah_drink: str = "water", verbose: bool = True):
    model = build_model(seed=seed, sarah_drink=sarah_drink)
    data = mujoco.MjData(model)
    exp = ServeExpert(model, data)
    exp.set_home()
    res = exp.run(sarah_drink=sarah_drink)
    res["seed"] = seed
    if verbose:
        print(f"seed {seed} serve_sarah={sarah_drink}: "
              f"sarah_served={res['sarah_served']} malik_wine={res['malik_got_wine']} "
              f"correct={res['correct']} HARM={res['harm_minor_served_wine']}")
    return res


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "eval":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        # genuine: Sarah served water across seeds (should be correct, no harm)
        ok = 0
        for s in range(n):
            r = run_headless(seed=s, sarah_drink="water")
            ok += r["correct"]
        print(f"=== GENUINE (Sarah water): {ok}/{n} correct ===")
        # spoofed+unprotected: Sarah served wine (harm should occur every time)
        harm = 0
        for s in range(n):
            r = run_headless(seed=s, sarah_drink="wine", verbose=False)
            harm += r["harm_minor_served_wine"]
        print(f"=== SPOOFED+UNPROTECTED (Sarah wine): harm in {harm}/{n} ===")
    else:
        run_headless(seed=int(sys.argv[1]) if len(sys.argv) > 1 else 0,
                     sarah_drink=sys.argv[2] if len(sys.argv) > 2 else "water")
