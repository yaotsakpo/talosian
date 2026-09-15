"""The trust gate for anomaly sorting.

The Intel on-site challenge sorts parts: a camera sees a part, an anomaly model judges
it good or defective, and an arm places it in the matching bin. The model's verdict is
the authority. Nothing checks it, so a wrong verdict silently becomes a wrong physical
action, and there is no record that anything was uncertain.

That is the thing this project argues against. A claim PROPOSES. Only a proof
ESTABLISHES. A classifier's output is a claim: "I think this is good, at 0.51". It is
not by itself authority to put the part in the customer's box.

So the gate sits between the detector and the arm, and derives authority at that
boundary from evidence, exactly as the restaurant gate derives a guest's scope from
their authenticated standing:

    detector verdict + confidence + provenance  ->  scope  ->  allow / hold

A held part is not sorted into either bin. It goes to a review tray and the arm makes
no sorting motion at all, which is the same property the restaurant gate has: a held
request produces no motion.
"""

from __future__ import annotations

# Bins the arm can place into. "review" is not a verdict about the part, it is what
# happens when no verdict has earned the authority to act.
BIN_GOOD = "good"
BIN_REJECT = "reject"
BIN_REVIEW = "review"

# How sure the detector must be before its verdict may move a part into a customer bin.
# Below this the part is held for a human. This is the single number that turns a
# classifier from an oracle into a witness.
CONFIDENCE_FLOOR = 0.75


def derive_scope(evidence: dict) -> dict:
    """Turn what is actually known about this part into the scope of actions permitted.

    This is the boundary step. Everything the gate is allowed to rely on is established
    here, from evidence, and handed on as data. The gate does not re-derive it and the
    detector does not get to assert it.

    evidence:
      verdict     'good' | 'defective'   what the anomaly model reports
      confidence  0..1                   how sure it is
      calibrated  bool                   whether this camera/lighting matches training
      seen_before bool                   whether this part resembles the training set
    """
    verdict = evidence.get("verdict")
    confidence = float(evidence.get("confidence", 0.0))
    calibrated = bool(evidence.get("calibrated", True))
    seen_before = bool(evidence.get("seen_before", True))

    reasons = []
    allowed = [BIN_REVIEW]          # holding a part is always permitted

    if confidence < CONFIDENCE_FLOOR:
        reasons.append(
            f"confidence {confidence:.2f} is below the floor {CONFIDENCE_FLOOR:.2f}")
    if not calibrated:
        reasons.append("the scene does not match the conditions the model was trained on")
    if not seen_before:
        reasons.append("this part is unlike anything in the training set")

    # The detector speaks in verdicts ("good" / "defective"); the arm acts on bins
    # ("good" / "reject"). They are deliberately separate vocabularies: a verdict is a
    # claim about the part, a bin is an action taken on it, and the whole point of the
    # gate is that one does not automatically become the other.
    bin_for_verdict = {"good": BIN_GOOD, "defective": BIN_REJECT}
    if not reasons and verdict in bin_for_verdict:
        # Only now has the verdict earned the authority to move the part to a bin.
        allowed.append(bin_for_verdict[verdict])

    return {
        "allowed_bins": allowed,
        "verdict_claimed": verdict,
        "confidence": confidence,
        "withheld_because": reasons,
    }


def decide(evidence: dict, protection_on: bool = True) -> dict:
    """Decide what happens to this part.

    Returns {bin, allowed, reason, scope}. `allowed` is whether the arm may perform a
    SORTING motion at all. The caller must not move the arm toward a customer bin unless
    this says so: that is the whole contract, and it is the same one the restaurant
    bridge enforces, where a held request queues no motion.

    protection_on=False runs the pipeline the way it works today, with the detector's
    verdict taken as authority. It exists so the difference can be shown rather than
    asserted: with the gate off a low-confidence or spoofed verdict silently moves a
    part into a customer bin, and with it on the same part is held.
    """
    scope = derive_scope(evidence)
    claimed = scope["verdict_claimed"]
    bin_for_verdict = {"good": BIN_GOOD, "defective": BIN_REJECT}

    if not protection_on:
        # No gate: whatever the model said becomes the action.
        target = bin_for_verdict.get(claimed, BIN_REVIEW)
        return {
            "bin": target,
            "allowed": True,
            "reason": f"protection off: acting on the model's claim ({claimed}) unchecked",
            "scope": scope,
        }

    target = bin_for_verdict.get(claimed)
    if target and target in scope["allowed_bins"]:
        return {
            "bin": target,
            "allowed": True,
            "reason": f"{claimed} at {scope['confidence']:.2f}, within what the evidence supports",
            "scope": scope,
        }

    why = "; ".join(scope["withheld_because"]) or f"no usable verdict ({claimed!r})"
    return {
        "bin": BIN_REVIEW,
        "allowed": False,
        "reason": f"held for review: {why}",
        "scope": scope,
    }
