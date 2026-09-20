"""The handoff: what an unsure judgment does to a physical action.

THE CLAIM UNDER TEST
    A vision model judges. An actuator acts on that judgment. Nothing sits
    between them. So when the model is unsure, the arm still moves, and the
    motion is wrong at exactly the rate the model is wrong.

    This is not specific to any one policy. It is a property of the HANDOFF:
    a judgment is a claim, a placement is an action, and in today's pipelines
    the first silently becomes the second.

WHAT IS MEASURED
    Wrong physical actions. Not accuracy, not AUROC, not a threshold curve.
    For every part: did the arm place it, and was that placement wrong?

    Two arms:
      ungated  today's pipeline. Every judgment becomes a placement.
      gated    a placement requires evidence. Unsupported judgments produce
               NO MOTION, which is the property that matters: a held part is
               not placed badly, it is not placed at all.

    The headline is conditional, because the unconditional rate hides the
    effect: AMONG THE PARTS THE MODEL WAS UNSURE ABOUT, how many wrong
    placements happened in each arm.

WHY THAT CONDITIONING IS THE HONEST CUT
    On parts the model is confident and correct about, the gate changes
    nothing and should not. Reporting over all parts averages the interesting
    region away. The claim is about the unsure region, so that is where it is
    measured, with the confident region reported alongside to show the gate is
    not simply refusing to work.

WHAT THIS DOES NOT SHOW
    No policy runs here. The placement is the bin the verdict selects, which is
    the handoff logic, not a trajectory. It measures whether a wrong ACTION is
    authorised, not whether an arm physically reaches the wrong bin. A wrong
    authorisation is a necessary condition for a wrong placement, so preventing
    it is necessary but not sufficient for safe motion.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def load_scores(path):
    rows = []
    with open(path) as fh:
        r = csv.reader(fh)
        next(r, None)
        for row in r:
            rows.append((float(row[0]), row[1].strip().lower()))
    return rows


def unsure(score, accept_at, band):
    """Is the model unsure about this part?

    Defined by distance from its own decision boundary, which is the only
    uncertainty signal a bare classifier exposes. This is deliberately the
    SAME signal the ungated pipeline has available: the point is not that the
    gate sees something exotic, it is that the ungated pipeline acts on this
    judgment anyway while the gate declines to.
    """
    return abs(score - accept_at) < band


def run(path, accept_at=0.5, floor=0.125, band=0.10):
    rows = load_scores(path)
    conf = lambda s: abs(s - accept_at) / max(accept_at, 1 - accept_at)

    tally = {
        "ungated": {"placed": 0, "wrong": 0, "held": 0},
        "gated": {"placed": 0, "wrong": 0, "held": 0},
    }
    sub = {
        "unsure": {"n": 0, "ungated_wrong": 0, "gated_wrong": 0, "gated_held": 0},
        "confident": {"n": 0, "ungated_wrong": 0, "gated_wrong": 0, "gated_held": 0},
    }

    for score, truth in rows:
        verdict = "good" if score < accept_at else "defective"
        is_wrong = verdict != truth
        region = "unsure" if unsure(score, accept_at, band) else "confident"
        sub[region]["n"] += 1

        # UNGATED: the judgment becomes the action. Always.
        tally["ungated"]["placed"] += 1
        if is_wrong:
            tally["ungated"]["wrong"] += 1
            sub[region]["ungated_wrong"] += 1

        # GATED: the placement requires evidence. Without it, no motion at all.
        if conf(score) < floor:
            tally["gated"]["held"] += 1
            sub[region]["gated_held"] += 1
        else:
            tally["gated"]["placed"] += 1
            if is_wrong:
                tally["gated"]["wrong"] += 1
                sub[region]["gated_wrong"] += 1

    n = len(rows)
    print(f"{n} parts. threshold {accept_at}, gate floor {floor}, "
          f"'unsure' = score within {band} of the boundary\n")

    print(f"{'':10} {'placements':>11} {'wrong placements':>18} {'no motion':>11}")
    for arm in ("ungated", "gated"):
        t = tally[arm]
        print(f"  {arm:8} {t['placed']:>11} {t['wrong']:>18} {t['held']:>11}")

    print(f"\n{'region':>12} {'parts':>7} {'ungated wrong':>15} {'gated wrong':>13} {'gated held':>12}")
    for region in ("unsure", "confident"):
        s = sub[region]
        print(f"{region:>12} {s['n']:>7} {s['ungated_wrong']:>15} "
              f"{s['gated_wrong']:>13} {s['gated_held']:>12}")

    u = sub["unsure"]
    c = sub["confident"]
    print(f"\nTHE CLAIM, on the {u['n']} parts the model was unsure about:")
    print(f"  ungated: {u['ungated_wrong']} wrong physical actions")
    print(f"  gated:   {u['gated_wrong']} wrong physical actions, "
          f"{u['gated_held']} parts not moved at all")
    print(f"\nwhere the model was confident ({c['n']} parts), the gate held "
          f"{c['gated_held']} and left {c['ungated_wrong'] - c['gated_wrong']} "
          f"errors unchanged" if c['gated_held'] == 0 else
          f"\nwhere the model was confident ({c['n']} parts): ungated "
          f"{c['ungated_wrong']} wrong, gated {c['gated_wrong']} wrong, "
          f"{c['gated_held']} held")
    return tally, sub


if __name__ == "__main__":
    here = Path(__file__).parent
    src = sys.argv[1] if len(sys.argv) > 1 else here / "scores_screw.csv"
    run(src)
