"""Confidence-based deferral versus boundary-derived authority, head to head.

THE SETTING
    Published work on unreliable VLAs (conformal prediction, uncertainty-aware
    deferral, perturbation-based failure detection) shares one structure: the
    model reports how sure it is, and the system abstains when that number is
    low. The model's own confidence is the input.

    Those papers also report the failure this creates. Under distribution shift
    a model can be CONFIDENTLY WRONG: it "continues executing incorrect actions
    while exhibiting low uncertainty". When that happens, every method that
    reads the model's confidence acts, because by its own account there is
    nothing to abstain about.

THE TEST
    Both approaches see the same inputs and are allowed the same amount of
    holding, so neither wins by being more cautious. The only difference is
    WHAT EVIDENCE the decision reads.

      confidence_deferral   the model's score, and distance from its own
                            decision boundary. This is the published family.
      boundary_authority    the same score for the verdict, but the authority
                            to act is derived from a channel the model does not
                            produce: how far the input's embedding sits from the
                            distribution the model was fitted to.

    Inputs are screw images (in-distribution) plus bottle images fed to the same
    screw model (distribution shift). The bottles are the case that matters: the
    model has no standing to judge them at all, and any action on one is wrong
    whichever bin it picks.

WHY EQUAL HOLD BUDGET MATTERS
    A gate that holds more will always prevent more. Comparing at equal hold
    rate removes that, so the remaining difference is attributable to the
    evidence used and not to caution. Each method's threshold is set so both
    hold the same number of parts.

WHAT IS AND IS NOT SHOWN
    This shows a second evidence channel catching inputs a confidence channel
    cannot, at matched cost. It does NOT reproduce the cited papers' methods:
    conformal prediction with real coverage guarantees and perturbation
    sampling are stronger than the distance-to-boundary proxy used here, and a
    fair claim against them needs their implementations, not this stand-in.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path


def load(path):
    rows = []
    with open(path) as fh:
        r = csv.reader(fh)
        next(r, None)
        for score, distance, truth, origin in r:
            rows.append({"score": float(score), "distance": float(distance),
                         "truth": truth, "origin": origin})
    return rows


def familiar_range(distances, k=1.5):
    """What embedding distances the model produces on inputs it was fitted to.

    Built from in-distribution NORMAL inputs only, which is what a deployment
    has after validating its own model. Nothing here reads a label saying an
    input is out of distribution: that would be measuring a lookup.
    """
    xs = sorted(distances)
    q1, q3 = xs[len(xs) // 4], xs[(3 * len(xs)) // 4]
    return q3 + k * max(q3 - q1, 1e-9)


def at_hold_budget(rows, key, budget, reverse=False):
    """Pick the threshold that holds exactly `budget` fraction of parts."""
    vals = sorted((r[key] for r in rows), reverse=reverse)
    idx = min(int(budget * len(vals)), len(vals) - 1)
    return vals[idx]


def run(path, accept_at=0.5):
    rows = load(path)
    in_dist = [r for r in rows if r["origin"] == "in_dist"]
    ood = [r for r in rows if r["origin"] == "ood"]
    normal = [r for r in in_dist if r["truth"] == "good"]

    limit = familiar_range([r["distance"] for r in normal])
    conf = lambda r: abs(r["score"] - accept_at) / max(accept_at, 1 - accept_at)

    print(f"{len(rows)} inputs: {len(in_dist)} in-distribution, {len(ood)} under shift")
    print(f"familiarity limit from {len(normal)} normal inputs: {limit:.2f}\n")

    # the boundary method holds whatever is unfamiliar; match that budget exactly
    n_unfamiliar = sum(1 for r in rows if r["distance"] > limit)
    budget = n_unfamiliar / len(rows)
    conf_floor = at_hold_budget(rows, "score", budget) and None
    # threshold on confidence such that the same COUNT is held
    confs = sorted(conf(r) for r in rows)
    floor = confs[min(n_unfamiliar, len(confs) - 1)]

    print(f"hold budget matched at {n_unfamiliar}/{len(rows)} inputs ({budget:.1%})")
    print(f"  confidence method holds when confidence < {floor:.3f}")
    print(f"  boundary method holds when distance > {limit:.2f}\n")

    methods = {
        "confidence deferral": lambda r: conf(r) >= floor,
        "boundary authority": lambda r: r["distance"] <= limit,
    }

    print(f"{'':22} {'acted on':>9} {'held':>6} "
          f"{'SHIFTED acted on':>18} {'in-dist wrong':>15}")
    out = {}
    for name, may_act in methods.items():
        ood_acted = sum(1 for r in ood if may_act(r))
        held = sum(1 for r in rows if not may_act(r))
        acted = len(rows) - held
        wrong = sum(1 for r in in_dist if may_act(r)
                    and (("good" if r["score"] < accept_at else "defective") != r["truth"]))
        out[name] = (ood_acted, held, acted, wrong)
        print(f"  {name:20} {acted:>9} {held:>6} {ood_acted:>18} {wrong:>15}")

    c, b = out["confidence deferral"], out["boundary authority"]
    print(f"\nAt the same hold budget, inputs under distribution shift acted on:")
    print(f"  confidence deferral: {c[0]} of {len(ood)}")
    print(f"  boundary authority:  {b[0]} of {len(ood)}")
    return out


if __name__ == "__main__":
    here = Path(__file__).parent
    run(sys.argv[1] if len(sys.argv) > 1 else here / "scores_dual.csv")
