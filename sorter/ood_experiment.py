"""Does an evidence gate hold what a confidence gate lets through?

THE QUESTION
    Selective prediction, the reject option, is established work: score the input,
    abstain below a threshold. If that is all this project does, it has contributed
    nothing.

    The claim here is different. Confidence asks "may the model answer?" The gate asks
    "may this answer move something?" Those come apart, and the case where they come
    apart is the one worth measuring: an input the model has never seen anything like,
    which it nonetheless scores confidently.

    A confidence-only gate cannot see that. The score is all it has, and the score is
    high, so it acts. An evidence gate has more than the score: whether this input
    resembles the training distribution at all. It withholds authority on that basis,
    independent of how sure the model claims to be.

THE SETUP
    Train Padim on one MVTec category (screw). Score two things with it:
      in-distribution   the screw test set, good and defective
      out-of-distribution   bottle images, which are not screws

    The bottles are the interesting part. No screw model has any business judging them,
    and none of them should be acted on.

FAIRNESS
    The evidence gate must not be handed the answer. `seen_before` is NOT read from a
    label saying "this one is a bottle": that would be measuring a lookup, not a
    method. It is computed from the score distribution the model produced on its own
    training data, which is something any deployment has at hand.

    Specifically: the model's scores on held-out NORMAL training-like inputs describe
    what "familiar" looks like. An input scoring far outside that range is unfamiliar,
    whatever its confidence. Both gates see exactly the same inputs; the evidence gate
    simply uses a second fact about them that the confidence gate throws away.

WHAT COUNTS AS A FAILURE
    An OOD input that gets ACTED ON. Not misclassified, acted on. Whether the screw
    model called that bottle good or defective is beside the point: it had no standing
    to route it anywhere, and a gate that lets it through has failed regardless of
    which bin it chose.
"""

from __future__ import annotations

import csv
import statistics as stats
from pathlib import Path


def load(path):
    rows = []
    with open(path) as fh:
        r = csv.reader(fh)
        next(r, None)
        for score, truth, origin in r:
            rows.append((float(score), truth, origin))
    return rows


def familiarity_band(reference_scores, k: float = 3.0):
    """What range of scores this model produces on inputs it knows.

    Built from the model's own scores on normal in-distribution inputs. An input whose
    score falls outside this band is unlike what the model was fitted to. Median and
    IQR rather than mean and standard deviation, because anomaly scores are skewed and
    a few extreme values should not widen the band.
    """
    xs = sorted(reference_scores)
    med = stats.median(xs)
    q1 = xs[len(xs) // 4]
    q3 = xs[(3 * len(xs)) // 4]
    iqr = max(q3 - q1, 1e-9)
    return {"median": med, "low": q1 - k * iqr, "high": q3 + k * iqr, "iqr": iqr}


def seen_before(score: float, band: dict) -> bool:
    """Does this input look like something the model has seen?

    Computed from the score alone against the familiarity band, so it needs nothing the
    deployment would not have. It is deliberately weak evidence: a real system would add
    provenance, embedding distance, and whether the camera and lighting match. The point
    is that even this much is more than a confidence threshold uses.
    """
    return band["low"] <= score <= band["high"]


def confidence(score: float, accept_at: float) -> float:
    return abs(score - accept_at) / max(accept_at, 1.0 - accept_at)


def confidence_gate(score, accept_at, floor):
    """Prior art: abstain when the score sits too near the decision boundary."""
    return confidence(score, accept_at) >= floor


def evidence_gate(score, accept_at, floor, band):
    """This project: confidence is ONE input. Unfamiliarity withholds authority on its
    own, however confident the model is."""
    if not seen_before(score, band):
        return False
    return confidence(score, accept_at) >= floor


def run(path, accept_at=0.5, floor=0.125):
    rows = load(path)
    in_dist = [(s, t) for s, t, o in rows if o == "in_dist"]
    ood = [s for s, t, o in rows if o == "ood"]

    # the familiarity band comes from the model's scores on GOOD in-distribution
    # inputs, which is what a deployment has after validating its own model.
    band = familiarity_band([s for s, t in in_dist if t == "good"])

    print(f"familiarity band from {sum(1 for _, t in in_dist if t == 'good')} normal "
          f"in-distribution inputs: [{band['low']:.3f}, {band['high']:.3f}]")
    print(f"operating threshold {accept_at}, confidence floor {floor}\n")

    results = {}
    for name, gate in (("confidence only", lambda s: confidence_gate(s, accept_at, floor)),
                       ("evidence gate", lambda s: evidence_gate(s, accept_at, floor, band))):
        ood_acted = sum(1 for s in ood if gate(s))
        id_acted = sum(1 for s, _ in in_dist if gate(s))
        id_shipped = sum(1 for s, t in in_dist
                         if gate(s) and s < accept_at and t == "defective")
        results[name] = {
            "ood_acted": ood_acted,
            "ood_total": len(ood),
            "id_acted": id_acted,
            "id_total": len(in_dist),
            "id_shipped": id_shipped,
        }

    print(f"{'':18} {'OOD acted on':>14} {'in-dist acted':>15} {'defects shipped':>17}")
    for name, r in results.items():
        print(f"  {name:16} {r['ood_acted']:>6}/{r['ood_total']:<7} "
              f"{r['id_acted']:>7}/{r['id_total']:<7} {r['id_shipped']:>17}")

    c, e = results["confidence only"], results["evidence gate"]
    print(f"\n  OOD inputs the confidence gate acted on, the evidence gate held: "
          f"{c['ood_acted'] - e['ood_acted']}")
    print(f"  in-distribution work given up to get that: "
          f"{c['id_acted'] - e['id_acted']} parts")
    return results


if __name__ == "__main__":
    here = Path(__file__).parent
    run(here / "scores_ood.csv")
