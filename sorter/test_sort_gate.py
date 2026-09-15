"""Tests for the sorting gate.

Run:  python -m pytest sorter/test_sort_gate.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from sort_gate import BIN_GOOD, BIN_REJECT, BIN_REVIEW, decide, derive_scope


def ev(verdict="good", confidence=0.95, calibrated=True, seen_before=True):
    return {"verdict": verdict, "confidence": confidence,
            "calibrated": calibrated, "seen_before": seen_before}


# --- what the evidence supports -------------------------------------------------

def test_a_confident_verdict_earns_its_bin():
    assert BIN_GOOD in derive_scope(ev("good", 0.96))["allowed_bins"]
    assert BIN_REJECT in derive_scope(ev("defective", 0.93))["allowed_bins"]


def test_a_coin_flip_earns_nothing():
    scope = derive_scope(ev("good", 0.51))
    assert scope["allowed_bins"] == [BIN_REVIEW]
    assert "confidence" in scope["withheld_because"][0]


def test_conditions_outside_training_withhold_authority():
    """A confident score means nothing if the scene is not the one the model learned."""
    scope = derive_scope(ev("good", 0.99, calibrated=False))
    assert scope["allowed_bins"] == [BIN_REVIEW]


def test_an_unfamiliar_part_withholds_authority():
    scope = derive_scope(ev("good", 0.99, seen_before=False))
    assert scope["allowed_bins"] == [BIN_REVIEW]


def test_holding_is_always_permitted():
    """Whatever the evidence, setting a part aside is never the unsafe option."""
    for e in (ev(), ev("defective", 0.1), ev("nonsense", 0.0, False, False)):
        assert BIN_REVIEW in derive_scope(e)["allowed_bins"]


# --- the decision ---------------------------------------------------------------

def test_an_allowed_part_is_sorted():
    d = decide(ev("defective", 0.97))
    assert d["allowed"] and d["bin"] == BIN_REJECT


def test_a_held_part_produces_no_sorting_action():
    d = decide(ev("good", 0.60))
    assert d["allowed"] is False
    assert d["bin"] == BIN_REVIEW
    assert d["reason"]


def test_an_unrecognised_verdict_fails_safe():
    """A verdict the gate does not understand must not become an action."""
    d = decide(ev("probably fine?", 0.99))
    assert d["allowed"] is False


def test_protection_off_reproduces_todays_pipeline():
    """The comparison the demo rests on: the same part, ungoverned, is sorted."""
    spoof = ev("good", 0.62, calibrated=False)
    assert decide(spoof, protection_on=False)["bin"] == BIN_GOOD
    assert decide(spoof, protection_on=True)["bin"] == BIN_REVIEW


def test_the_reason_names_every_failing_check():
    d = decide(ev("good", 0.4, calibrated=False, seen_before=False))
    assert "confidence" in d["reason"]
    assert "trained on" in d["reason"]
    assert "training set" in d["reason"]
