// Anti-replay WINDOW for continuity proofs.
//
// WHY THIS EXISTS. The original design tracked a single monotone counter and
// expected exactly `counter + 1` on every message. That cannot survive ordinary
// email delivery. A reordered or duplicated message arrives BEHIND the counter,
// and a forward-only look-ahead can never recover it: accepting step n+1 first
// advances past n, so n is then permanently in the past. Measured on 1000-session
// runs, a legitimate peer was falsely flagged as a takeover in ~61% of sessions at
// 5% adjacent-pair reordering, and the rate was IDENTICAL for look-ahead windows
// of 1, 3 and 5 — the window does nothing for this failure mode.
//
// THE FIX is the IPsec anti-replay discipline (RFC 4303 §3.4.3): track the highest
// step seen plus a bitmap of which recent steps have already been consumed. A step
// inside the window is accepted exactly ONCE, in any order. Replay resistance is
// preserved because a consumed step is never accepted a second time, and a step
// that has fallen below the window floor is rejected as too old.
import { verifyToken } from "./continuity-token.js";
const DEFAULT_SIZE = 64;
// How far ahead of the highest-seen step we will look. Bounds the work done on a
// message that claims a wildly future step.
const LOOKAHEAD = 32;
export function createWindow(size = DEFAULT_SIZE) {
    return { highest: 0, size, seen: [] };
}
// Is `step` inside the window and not yet consumed?
function isFresh(w, step) {
    if (step <= 0)
        return false;
    const floor = Math.max(0, w.highest - w.size);
    if (step <= floor)
        return false; // too old: fell out of the window
    return !w.seen.includes(step);
}
function consume(w, step) {
    const highest = Math.max(w.highest, step);
    const floor = Math.max(0, highest - w.size);
    // keep only steps still inside the (possibly advanced) window
    const seen = [...w.seen, step].filter((s) => s > floor);
    return { highest, size: w.size, seen };
}
// Verify an incoming token against the window. We do not know which step the
// sender used, so we try every fresh candidate in range: from the window floor up
// to LOOKAHEAD beyond the highest step seen. A token verifies for at most one
// step, so at most one candidate can match.
export async function acceptStep(w, seed, responseHex) {
    if (responseHex === null)
        return { accepted: false, step: null, window: w };
    const floor = Math.max(0, w.highest - w.size);
    const ceiling = w.highest + LOOKAHEAD;
    for (let step = floor + 1; step <= ceiling; step++) {
        if (!isFresh(w, step))
            continue;
        if (await verifyToken(seed, step, responseHex)) {
            return { accepted: true, step, window: consume(w, step) };
        }
    }
    return { accepted: false, step: null, window: w };
}
