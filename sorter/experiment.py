"""Measure what the gate actually changes, on real model scores.

The honest question is not "is a trust gate a good idea" but "how many parts end up
somewhere different, and is that better". This answers it from a model's own scores.

WHAT IT NEEDS
    A list of (score, truth) for a held-out test set:
      score  the anomaly score the model gave that part, 0..1
      truth  "good" or "defective", the ground truth for that part
    Nothing else. No robot, no Studio, no retraining.

WHAT IT REPORTS
    Baseline: today's pipeline. One threshold, two bins, every part is sorted.
    Gated:    the same threshold, plus a floor below which the placement is skipped.

    The numbers that matter:
      defects shipped       a defective part sent to the customer bin. The expensive
                            error: everything later built on it inherits the fault.
      good parts scrapped   a good part thrown away. Wasteful but recoverable.
      held                  parts nobody acted on, sent for review instead.

    The gate cannot make the model more accurate, so it cannot reduce both errors for
    free. What it does is convert some of them into holds. Whether that trade is worth
    it is a judgement about relative cost, and this prints the numbers to judge with.
"""

from __future__ import annotations


def evaluate(samples, accept_at: float, floor: float | None = None):
    """samples: iterable of (score, truth). accept_at: the good/defective threshold.
    floor: confidence needed to act at all. None reproduces today's pipeline."""
    shipped = scrapped = held = correct = 0
    for score, truth in samples:
        verdict = "good" if score < accept_at else "defective"
        confidence = abs(score - accept_at) / max(accept_at, 1 - accept_at)

        if floor is not None and confidence < floor:
            held += 1
            continue
        if verdict == "good" and truth == "defective":
            shipped += 1
        elif verdict == "defective" and truth == "good":
            scrapped += 1
        else:
            correct += 1
    return {"shipped": shipped, "scrapped": scrapped, "held": held, "correct": correct}


def compare(samples, accept_at: float, floor: float):
    samples = list(samples)
    base = evaluate(samples, accept_at)
    gated = evaluate(samples, accept_at, floor=floor)
    n = len(samples)
    print(f"{n} parts, accept threshold {accept_at}, gate floor {floor}\n")
    print(f"{'':22} {'defects shipped':>16} {'good scrapped':>14} {'held':>6} {'correct':>8}")
    for label, r in (("today (two bins)", base), ("with the gate", gated)):
        print(f"  {label:20} {r['shipped']:>16} {r['scrapped']:>14} {r['held']:>6} {r['correct']:>8}")
    print()
    caught = base["shipped"] - gated["shipped"]
    saved = base["scrapped"] - gated["scrapped"]
    print(f"  defects that would have shipped, now held: {caught}")
    print(f"  good parts that would have been scrapped, now held: {saved}")
    print(f"  cost: {gated['held']} parts need a human look")
    return base, gated


def sweep(samples, accept_at: float, floors=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5)):
    """Show the whole tradeoff rather than one chosen point.

    A higher floor catches more mistakes but hands more parts to a human. There is no
    setting that is simply "best": the right floor depends on what a shipped defect costs
    you relative to a person's time. Printing the curve is the honest way to present it,
    because it lets the reader pick.
    """
    samples = list(samples)
    n = len(samples)
    print(f"{n} parts, accept threshold {accept_at}\n")
    print(f"{'floor':>7} {'defects shipped':>16} {'good scrapped':>14} {'held':>7} {'held %':>7}")
    for f in floors:
        r = evaluate(samples, accept_at, floor=(None if f == 0 else f))
        print(f"{f:>7.2f} {r['shipped']:>16} {r['scrapped']:>14} {r['held']:>7} {100*r['held']/n:>6.1f}%")
