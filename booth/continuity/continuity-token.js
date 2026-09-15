// The continuity TOKEN wire-format — what a genuine Attest agent embeds in an
// outgoing message so the receiver can verify, with REAL crypto, that it is
// still the same principal that earned trust.
//
// A message from a seeded counterpart must carry `[attest:continuity:<hex>]`,
// where <hex> = computeResponse(seed, counter, nonce). The receiver recomputes
// the expected value from the stored seed + the step it expects. An impostor
// who holds the address but not the seed cannot produce a value that verifies —
// so marker-presence is NOT enough; the crypto is what gates.
//
// The nonce is derived from the counter so both sides compute the same challenge
// without a round-trip (the counter is the shared ratchet step).
import { computeResponse, verifyResponse } from "./continuity.js";
const TOKEN_RE = /\[attest:continuity:([a-f0-9]{16,})\]/i;
function nonceFor(counter) {
    return `attest-continuity-step-${counter}`;
}
// Produce the token a genuine agent embeds in its outgoing message at `counter`.
export async function emitToken(seed, counter) {
    const resp = await computeResponse(seed, counter, nonceFor(counter));
    return `[attest:continuity:${resp}]`;
}
// Extract the response hex from an incoming message, or null if absent.
export function readToken(message) {
    const m = message.match(TOKEN_RE);
    return m ? m[1] : null;
}
// Verify an extracted token against the stored seed at the expected step. This
// is the REAL check — HMAC verification, not marker-presence.
export async function verifyToken(seed, counter, responseHex) {
    return await verifyResponse(seed, counter, nonceFor(counter), responseHex);
}
