"""Run the gate experiment and print the table.

    python sorter/run_experiment.py                 # documented reference distribution
    python sorter/run_experiment.py scores.csv      # your own model: score,truth per line

The reference distribution is not a claim about your parts. It is a stand-in with a
realistic overlap, so the harness can be read and argued with before a trained model
exists. Replace it with real scores the moment you have them: that is the number worth
reporting.
"""
from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from experiment import compare, sweep


def reference(n_good=400, n_defective=100, seed=0):
    """A stand-in score distribution with a genuine overlap between classes.

    Anomaly scores for good and defective parts overlap in practice; that overlap is the
    whole problem, and a benchmark number can look excellent while it exists (pixel
    AUROC can exceed 0.99 by predicting nothing is defective). The spread here is chosen
    so the overlap is visible rather than flattering.
    """
    rng = random.Random(seed)
    clamp = lambda v: min(1.0, max(0.0, v))
    good = [(clamp(rng.gauss(0.30, 0.18)), "good") for _ in range(n_good)]
    bad = [(clamp(rng.gauss(0.70, 0.18)), "defective") for _ in range(n_defective)]
    return good + bad


def from_csv(path):
    """score,truth per line. truth is 'good' or 'defective'."""
    out = []
    with open(path) as fh:
        for row in csv.reader(fh):
            if not row or row[0].strip().lower() in ("score", "#"):
                continue
            out.append((float(row[0]), row[1].strip().lower()))
    return out


if __name__ == "__main__":
    if len(sys.argv) > 1:
        samples = from_csv(sys.argv[1])
        print(f"scores from {sys.argv[1]}\n")
    else:
        samples = reference()
        print("REFERENCE DISTRIBUTION (no trained model yet: replace with real scores)\n")

    sweep(samples, accept_at=0.5)
    print()
    compare(samples, accept_at=0.5, floor=0.2)
