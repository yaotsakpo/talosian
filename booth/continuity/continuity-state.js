// The continuity STATE MACHINE. Pure: given a counterpart's continuity record
// and whether the incoming message carried a valid rotating response, decide the
// verdict the gate acts on. The crypto (deriveSeed/computeResponse/verifyResponse)
// is in continuity.ts; this file decides what the result MEANS for trust.
//
// Design (user's): first trusted contact is ranked as normal (no continuity yet),
// then we SEED the counterpart — embed a key in our reply that every Attest agent
// knows to decode. From then on we watch every message for the forward-secret
// response. A SEEDED counterpart that stops producing it (missing or wrong) is a
// takeover signal, because the impostor holds the address but not the seed.
export function continuityVerdict(record, proof) {
    // Never seen before → continuity doesn't apply yet. Caller ranks as today and
    // (if it decides to trust) seeds the counterpart for next time.
    if (!record) {
        return { status: "not_applicable", shouldHold: false, provable: false };
    }
    // Seed sent but no proof received yet — the counterpart hasn't replied since we
    // seeded. Not a takeover; just awaiting first confirmation. Don't hold on this
    // alone (the normal gate still applies).
    if (!record.seeded) {
        return { status: "pending", shouldHold: false, provable: false };
    }
    // Valid token → still the same principal.
    if (proof.hasResponse && proof.responseValid) {
        return { status: "confirmed", shouldHold: false, provable: false };
    }
    // COMMISSION: a token is present but does NOT verify. Only an entity that
    // produced a value without the seed does this — a self-contained proof of a
    // fault. Hold, and this is transferable (a network-wide reputation event).
    if (proof.hasResponse && !proof.responseValid) {
        return { status: "takeover_suspected", shouldHold: true, provable: true };
    }
    // OMISSION: no token at all. Hold locally (safe), but this is NOT a proof — it
    // could be reordering, delay, or a drop. It must not propagate as reputation.
    return { status: "unproven_gap", shouldHold: true, provable: false };
}
