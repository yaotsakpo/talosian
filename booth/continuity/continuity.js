// The continuity handshake — proving an agent is STILL the same principal that
// earned trust, not an impostor who inherited its address.
//
// Composition (all standard primitives, aimed at an unsolved target):
//   • TOFU continuity  — trust is fixed at first authenticated contact, then
//     re-checked on every later interaction (like SSH known_hosts)…
//   • …but with a FORWARD-SECRET RATCHET instead of a static key: the per-step
//     key is derived by hashing forward from the seed, so capturing step N's
//     material never yields step N+1 (à la a Signal-style ratchet)…
//   • …driven by a bidirectional CHALLENGE-RESPONSE over a pre-shared seed.
//
// Threat model: an attacker takes over an agent's ADDRESS/mailbox (spoof, hijack)
// but never held the trust-time SEED. On challenge, they cannot produce the
// forward-secret response → takeover is exposed at interaction time. It does NOT
// defend against theft of the seed itself (a deeper compromise) — that bar is
// far higher than spoofing an address, and is the honest limit.
//
// Real crypto: HMAC-SHA256 via Web Crypto (available in Convex actions and the
// edge-runtime test env). All functions are async.
const enc = new TextEncoder();
async function hmac(keyBytes, msg) {
    const key = await crypto.subtle.importKey("raw", keyBytes, { name: "HMAC", hash: "SHA-256" }, false, ["sign"]);
    const sig = await crypto.subtle.sign("HMAC", key, enc.encode(msg));
    return toHex(new Uint8Array(sig));
}
async function sha256(msg) {
    const d = await crypto.subtle.digest("SHA-256", enc.encode(msg));
    return new Uint8Array(d);
}
function toHex(bytes) {
    let s = "";
    for (const b of bytes)
        s += b.toString(16).padStart(2, "0");
    return s;
}
function hexToBytes(hex) {
    const out = new Uint8Array(hex.length / 2);
    for (let i = 0; i < out.length; i++) {
        out[i] = parseInt(hex.slice(i * 2, i * 2 + 2), 16);
    }
    return out;
}
// The shared seed, established at trust-time. Both agents derive it independently
// from the SAME trust-establishment secret + the (order-independent) pair of
// agent identifiers. In a full deployment the trust secret comes from an
// authenticated key-exchange at first contact; here it is the shared input.
export async function deriveSeed(agentX, agentY, trustSecret) {
    // sort so the pair is symmetric: seed(A,B) === seed(B,A)
    const [a, b] = [agentX, agentY].sort();
    return await hmac(enc.encode(trustSecret), `pair:${a}|${b}`);
}
// Forward-secret per-step key: hash-chain forward from the seed. Deriving step N
// requires the seed (or the step N-1 key), so an observer of step N's outputs
// cannot roll forward without the seed. Deterministic: both sides reach the same
// key for the same counter.
export async function ratchetKey(seed, counter) {
    // domain-separated hash chain: k_n = H(seed | "ratchet" | n)
    const bytes = await sha256(`${seed}|ratchet|${counter}`);
    return toHex(bytes);
}
// The response to a challenge: HMAC over the challenge nonce, keyed by the
// forward-secret step key. Bound to (seed, counter, nonce) — replaying an old
// response at a new step or a different nonce fails.
export async function computeResponse(seed, counter, nonce) {
    const k = await ratchetKey(seed, counter);
    return await hmac(hexToBytes(k), `challenge:${counter}:${nonce}`);
}
// Verify a counterparty's response. True only if they hold the seed and produced
// the response for THIS step + nonce.
export async function verifyResponse(seed, counter, nonce, response) {
    const expected = await computeResponse(seed, counter, nonce);
    return timingSafeEqual(expected, response);
}
// Constant-time-ish string compare (avoid leaking match length via early exit).
function timingSafeEqual(a, b) {
    if (a.length !== b.length)
        return false;
    let diff = 0;
    for (let i = 0; i < a.length; i++)
        diff |= a.charCodeAt(i) ^ b.charCodeAt(i);
    return diff === 0;
}
