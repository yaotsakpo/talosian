// agent-continuity — detect COUNTERPART TAKEOVER between two agents.
//
// The problem this solves (and, at time of writing, the only unclaimed gap in
// the agent-identity landscape): one-time authentication proves who a party was
// at first contact. It does NOT prove the party you talk to LATER is still that
// same principal, rather than an impostor who took over its address/mailbox.
//
// The mechanism is a composition of STANDARD primitives aimed at that target —
// there is nothing novel in the cryptography itself:
//   • TOFU  — trust is fixed at first authenticated contact (like SSH known_hosts)
//   • a FORWARD-SECRET RATCHET — the per-step key is hashed forward from a shared
//     seed, so capturing step N never yields step N+1 (Signal-style)
//   • CHALLENGE-RESPONSE over that pre-shared seed, per message
// An attacker who takes over the address but never held the seed cannot produce
// the forward-secret response, so the takeover is exposed at interaction time.
// It does NOT defend against theft of the seed itself — that is a deeper
// compromise and the honest limit of the mechanism.
//
// The core is PURE and STORAGE-AGNOSTIC: it never touches a database. The caller
// owns the continuity record (seed, counter, window, status) and persists it
// however it likes. Runs anywhere Web Crypto (crypto.subtle) exists — Node 20+,
// browsers, edge/workers, Deno.
// ── Low-level primitives ────────────────────────────────────────────────────
export { deriveSeed, ratchetKey, computeResponse, verifyResponse, } from "./continuity.js";
export { emitToken, readToken, verifyToken } from "./continuity-token.js";
export { createWindow, acceptStep, } from "./replay-window.js";
export { continuityVerdict, } from "./continuity-state.js";
// ── High-level, one-call API (composes the primitives) ──────────────────────
import { emitToken, readToken } from "./continuity-token.js";
import { acceptStep } from "./replay-window.js";
import { continuityVerdict, } from "./continuity-state.js";
import { deriveSeed } from "./continuity.js";
/**
 * Establish continuity with a counterpart at first trusted contact. Both sides
 * derive the SAME seed independently from their two agent identifiers (order-
 * independent) plus a `trustSecret` from an authenticated exchange at first
 * contact. Persist the returned seed against that counterpart; never transmit
 * it — you send TOKENS derived from it (see `sign`).
 */
export async function establish(selfId, counterpartId, trustSecret) {
    return { seed: await deriveSeed(selfId, counterpartId, trustSecret) };
}
/**
 * Produce the token a genuine agent embeds in its OUTGOING message so the
 * receiver can verify continuity. `counter` is the next ratchet step (the
 * receiver's stored counter + 1).
 */
export async function sign(seed, counter) {
    return emitToken(seed, counter);
}
/**
 * Verify an INCOMING message against stored continuity, advance the anti-replay
 * window on success, and return both the verdict and the next window/counter to
 * persist. Pure: pass in what you loaded, persist what comes back.
 *
 *   const { verdict, window, counter } = await verify({ seed, window, record, rawText });
 *   if (verdict.shouldHold) hold(); else deliver();
 *   save({ ...record, counter, window, status: verdict.status });
 */
export async function verify(input) {
    const { seed, window, record, rawText } = input;
    // A counterpart we have not seeded yet: continuity does not apply.
    if (!record.seeded) {
        const verdict = continuityVerdict(record, {
            hasResponse: false,
            responseValid: false,
        });
        return { verdict, window, counter: record.counter };
    }
    const tokenHex = readToken(rawText);
    let nextWindow = window;
    let responseValid = false;
    if (tokenHex !== null) {
        const accept = await acceptStep(window, seed, tokenHex);
        responseValid = accept.accepted;
        if (accept.accepted)
            nextWindow = accept.window;
    }
    const verdict = continuityVerdict(record, {
        hasResponse: tokenHex !== null,
        responseValid,
    });
    return { verdict, window: nextWindow, counter: nextWindow.highest };
}
