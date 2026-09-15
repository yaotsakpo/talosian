"""Inter-arm coordination under Talosian's trust layer.

Two arms set a place setting and must COORDINATE. The Pourer (P) may only pour
AFTER the Holder (H) reports the glass is secure. That report is a MESSAGE from
one agent to another, and Talosian governs it exactly like a human command:

  - Does the message really come from H?           (continuity)
  - Does H have authority to declare 'glass secure'? (predicate-scoped authority)

The attack: an adversary injects a spoofed 'glass_secure' before H has actually
secured the glass. Without the protocol, P trusts it and pours early, the glass
tips (physical failure). With the protocol, the spoofed message carries no valid
continuity proof from H (and 'declare secure' is not in the adversary's scope),
so P holds the pour and the glass survives.

This is the PSAP war-sim (a spoofed inter-agent message causing physical harm)
made literal with glassware, and it reuses the same trust primitives as the
human-command path. The crypto here mirrors agent-continuity: an HMAC token from
H's seed. Kept small and self-contained so the bridge can run it without the JS.
"""

from __future__ import annotations

import hashlib
import hmac
import re

# Each arm is a PRINCIPAL with a continuity seed and a scope of predicates it may
# assert. These mirror the browser's commander records.
ARM_SCOPES = {
    "H": {"report.glass_secure", "place.plate", "hold.glass"},
    "P": {"pour", "place.fork"},
}


def derive_seed(trust_secret: str, a: str, b: str) -> str:
    lo, hi = sorted([a, b])
    return hmac.new(trust_secret.encode(), f"pair:{lo}|{hi}".encode(), hashlib.sha256).hexdigest()


def _step_key(seed: str, counter: int) -> str:
    return hashlib.sha256(f"{seed}|ratchet|{counter}".encode()).hexdigest()


def sign_message(seed: str, counter: int) -> str:
    """The token a genuine arm embeds when it sends a coordination message."""
    k = bytes.fromhex(_step_key(seed, counter))
    nonce = f"attest-continuity-step-{counter}"
    resp = hmac.new(k, f"challenge:{counter}:{nonce}".encode(), hashlib.sha256).hexdigest()
    return f"[attest:continuity:{resp}]"


def verify_message(seed: str, counter: int, raw_text: str) -> bool:
    """True only if raw_text carries a token that matches seed at this step."""
    m = re.search(r"\[attest:continuity:([a-f0-9]{16,})\]", raw_text)
    if not m:
        return False
    expected = sign_message(seed, counter)
    exp_hex = re.search(r"([a-f0-9]{16,})", expected).group(1)
    # constant-time compare
    return hmac.compare_digest(m.group(1), exp_hex)


def govern_message(sender_arm: str, predicate: str, raw_text: str, *,
                   seed: str, counter: int, protection: bool) -> dict:
    """Talosian's decision for an ARM-TO-ARM coordination message. Same shape as
    the human-command gate: known sender? in scope? proven? Returns
    {accept, reason, attack}."""
    # 1. Scope: may this arm assert this predicate at all? (PSAP)
    if predicate not in ARM_SCOPES.get(sender_arm, set()):
        return {"accept": False, "reason": f"{sender_arm} has no authority to assert {predicate}",
                "attack": True}
    # 2. Continuity: is the message really from that arm's seed?
    proven = verify_message(seed, counter, raw_text)
    if not protection:
        # Protection OFF: act on the claim without enforcing the proof. A spoofed
        # message is obeyed, that is the takeover the demo shows.
        return {"accept": True, "reason": "protection off: message accepted unverified",
                "attack": not proven}
    if proven:
        return {"accept": True, "reason": "continuity confirmed: message is genuine", "attack": False}
    return {"accept": False,
            "reason": "continuity failed: message could not prove it came from " + sender_arm,
            "attack": True}
