"""Derive the gate's floor from scores, instead of picking a number.

WHY THIS FILE EXISTS
    The gate reads a score and holds anything below a floor. That floor used to be a
    constant someone chose. A constant is indefensible for two reasons:

      1. Nobody can say why it is 0.75 rather than 0.7. It encodes a preference that
         was never stated, so it cannot be argued with.
      2. It is tied to one model's score distribution. Swap the detector and the
         interface still fits, but the number silently means something different.
         "Model-agnostic" is then true of the plumbing and false of the behaviour.

    So the floor is not an input. It is an OUTPUT, derived from a stated error budget
    on a held-out set. You do not say "hold below 0.75". You say "at most 1% of parts
    reaching the customer may be defective" and this finds the floor that delivers it,
    against whatever model you are actually running.

    That is what makes recalibration automatic: change the detector, rerun this, the
    budget is unchanged and the floor moves on its own.

WHAT A BUDGET IS
    A budget is a policy decision, and it belongs to whoever owns the consequences.
    It is expressed in outcomes, not in score units:

      max_shipped      fraction of acted-on parts that may be defective
      max_scrapped     fraction of acted-on parts that may be good but binned as bad
      max_held         fraction of all parts you are willing to hand to a human

    The first two are error budgets, the third is a cost ceiling. You will not always
    get all three: a model with heavy class overlap cannot meet a tight error budget
    without holding a lot. When they conflict this says so rather than quietly
    returning a floor that meets none of them.

HONEST LIMITS
    - A floor derived on one set is only valid for parts drawn like that set. New
      lighting, a new supplier, a new part shape: recalibrate.
    - This measures a model you already have. It does not make that model better. If
      the classes overlap badly, every floor is a bad floor, and the right response is
      a better detector, not a cleverer threshold.
    - Held-out means held out. Deriving the floor on the same scores you then report
      results against will flatter you. `split` exists so that is harder to do by
      accident.
"""

from __future__ import annotations

import random
from typing import Iterable, Sequence


def confidence(score: float, accept_at: float) -> float:
    """How far this score sits from the decision boundary, normalised to 0..1.

    Distance from the boundary is the evidence. A score sitting exactly on the
    threshold carries no information about which side it belongs to, whatever the
    model reports, and that is the case the gate exists to catch.
    """
    return abs(score - accept_at) / max(accept_at, 1.0 - accept_at)


def outcomes(samples: Iterable[tuple[float, str]], accept_at: float, floor: float | None):
    """Count what happens to a set of parts at a given floor.

    Returns acted (parts the arm placed), plus the three outcomes that matter. Rates
    are over ACTED parts, not all parts: a defect that was held was never shipped, and
    counting it against the shipped rate would make holding look like failure.
    """
    shipped = scrapped = correct = held = 0
    for score, truth in samples:
        if floor is not None and confidence(score, accept_at) < floor:
            held += 1
            continue
        verdict = "good" if score < accept_at else "defective"
        if verdict == "good" and truth == "defective":
            shipped += 1
        elif verdict == "defective" and truth == "good":
            scrapped += 1
        else:
            correct += 1
    acted = shipped + scrapped + correct
    total = acted + held
    return {
        "acted": acted,
        "held": held,
        "shipped": shipped,
        "scrapped": scrapped,
        "correct": correct,
        "shipped_rate": shipped / acted if acted else 0.0,
        "scrapped_rate": scrapped / acted if acted else 0.0,
        "held_rate": held / total if total else 0.0,
    }


def calibrate(
    samples: Sequence[tuple[float, str]],
    accept_at: float = 0.5,
    max_shipped: float | None = None,
    max_scrapped: float | None = None,
    max_held: float | None = None,
    steps: int = 200,
):
    """Find the lowest floor that satisfies the budget.

    Lowest, not safest: every unit of floor above what the budget requires is autonomy
    given up for nothing. The gate's purpose is to hold what must be held, not to hold
    as much as possible.

    Returns {floor, met, outcomes, reason}. `met` is False when no floor in [0,1]
    satisfies every stated limit, and `reason` says which one could not be met. A
    caller that ignores `met` and uses the floor anyway is running a budget it did not
    get, so check it.
    """
    samples = list(samples)
    if not samples:
        raise ValueError("no samples: a floor cannot be derived from nothing")

    grid = [i / steps for i in range(steps + 1)]
    best_effort = None

    for f in grid:
        r = outcomes(samples, accept_at, floor=f if f > 0 else None)
        if r["acted"] == 0:
            break  # holding everything satisfies any error budget and is useless
        ok_ship = max_shipped is None or r["shipped_rate"] <= max_shipped
        ok_scrap = max_scrapped is None or r["scrapped_rate"] <= max_scrapped
        ok_held = max_held is None or r["held_rate"] <= max_held
        if ok_ship and ok_scrap and ok_held:
            return {"floor": f, "met": True, "outcomes": r, "reason": "budget met"}
        if ok_ship and ok_scrap and best_effort is None:
            # errors are within budget but this costs more holding than allowed
            best_effort = (f, r)

    if best_effort is not None:
        f, r = best_effort
        return {
            "floor": f,
            "met": False,
            "outcomes": r,
            "reason": (
                f"the error budget needs {r['held_rate']:.1%} of parts held, "
                f"which is above the {max_held:.1%} you allowed. Either accept more "
                f"holding, loosen the error budget, or use a better detector."
            ),
        }

    r = outcomes(samples, accept_at, floor=None)
    return {
        "floor": 0.0,
        "met": False,
        "outcomes": r,
        "reason": (
            "no floor satisfies this budget. The classes overlap too much for a "
            "threshold to separate them: at every floor some part that must not be "
            "acted on still is. This is a detector problem, not a gate problem."
        ),
    }


def split(samples: Sequence[tuple[float, str]], holdout: float = 0.5, seed: int = 0):
    """Split into (derive, report). Derive the floor on one, report results on the other.

    Reporting on the same scores that chose the floor measures how well a number fits
    the set it was fitted to, which is not a result. This is here so that separation is
    the easy path.
    """
    xs = list(samples)
    random.Random(seed).shuffle(xs)
    cut = int(len(xs) * holdout)
    return xs[cut:], xs[:cut]


def report(derive_set, report_set, accept_at=0.5, **budget):
    """Derive a floor on one set, then print how it behaves on data it has not seen."""
    cal = calibrate(derive_set, accept_at=accept_at, **budget)
    floor = cal["floor"]
    held_out = outcomes(report_set, accept_at, floor=floor if floor > 0 else None)
    ungated = outcomes(report_set, accept_at, floor=None)

    stated = ", ".join(f"{k}={v:.1%}" for k, v in budget.items() if v is not None)
    print(f"budget: {stated or '(none stated)'}")
    print(f"derived floor: {floor:.3f}   ({cal['reason']})")
    print(f"derived on {len(derive_set)} parts, reported on {len(report_set)} unseen\n")
    print(f"{'':16} {'acted':>7} {'held':>7} {'shipped':>9} {'scrapped':>9} {'ship rate':>10}")
    for label, r in (("no gate", ungated), ("with gate", held_out)):
        print(f"  {label:14} {r['acted']:>7} {r['held']:>7} {r['shipped']:>9} "
              f"{r['scrapped']:>9} {r['shipped_rate']:>9.2%}")
    return cal, held_out, ungated
