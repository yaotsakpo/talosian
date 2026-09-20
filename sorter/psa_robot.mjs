// psa_robot.mjs
// Predicate-scoped authority applied to a physical-AI data marketplace.
//
// WHY THIS EXPERIMENT
//   A company buys robot demonstrations from sources and shares revenue with
//   them. Facts about each demonstration arrive from several parties and those
//   parties conflict. The question this asks is not "is this demonstration
//   anomalous" (that is a detector's job and is prior art). It is: WHICH PARTY
//   SETTLES WHICH FACT, and what happens when a party that is paid per accepted
//   demonstration asserts its own standing.
//
//   Two claims are under test, transplanted from the predicate-scoped authority
//   paper into this domain.
//
//   CLAIM 1 (expressiveness). Authority inverts across predicates. The human who
//   performed a demonstration governs what task they were doing: nobody else was
//   there. That same human does NOT govern whether the episode is clean enough
//   to train on, which is settled by measurement at ingest. So source outranks
//   rig on one predicate and rig outranks source on another, and no single
//   ranking of parties reproduces both rows.
//
//   CLAIM 2 (soundness). A source paid per accepted demonstration has a direct
//   financial incentive to assert that its episodes are high quality. If the
//   authority class is read from the submission, that assertion pays. If it is
//   derived from the authenticated circumstances of the write, it does not.
//
//   This is the payment-instruction case from the paper with a different
//   payload, and the revenue share is what makes it live rather than theoretical.
//
// WHAT IS NOT CLAIMED
//   Nothing here shows that boundary derivation is achievable in a real ingest
//   pipeline: like the original Study A, the channel class is injected by the
//   generator. What is measured is the RESOLUTION RULE, conditional on such a
//   class being available, against the alternatives a marketplace would
//   otherwise use.

// ---------------------------------------------------------------------------
// Principals
//   source    the person or team that performed and submitted the demonstration
//   rig       the capture apparatus: cameras, calibration, clock
//   reviewer  an internal human who inspects a sample of episodes
//   agent     an automated pipeline component with no independent standing
// ---------------------------------------------------------------------------

// Channel class, derived from HOW the write authenticated, never from what it
// says about itself. Authority non-self-assertion in operational form.
const CHANNEL_CLASS = { rig_attested: 3, source_authenticated: 3, reviewed_internal: 3, submitted_portal: 2, agent_plane: 1 };

// One number per principal, predicate-independent: the structure Theorem 1
// shows cannot represent inversion.
const GLOBAL_RANK = { rig: 5, reviewer: 4, source: 3, agent: 1 };

// A configured survivorship table, fixed at config time, no boundary derivation.
const MDM_PRECEDENCE = { reviewer: 5, rig: 4, source: 4, agent: 1 };

// The predicate catalog. authRole is the party that TRULY governs the fact.
// The inversion is deliberate and is the point: source governs intent, rig
// governs measured conditions, reviewer governs fitness to train on.
export const PREDICATES = [
  // The source was the only party present. What they were attempting is theirs.
  { name: "demo.task_intent", gen: (r) => r.pick(["pick and place", "open drawer", "pour", "stack", "wipe surface"]), authRole: "source", otherRoles: ["reviewer", "agent"] },
  // Likewise: whether the attempt succeeded on its own terms.
  { name: "demo.outcome_reported", gen: (r) => r.pick(["completed", "abandoned", "retried", "partial"]), authRole: "source", otherRoles: ["rig", "agent"] },
  // The rig measured it. A source cannot make a scene well lit by saying it was.
  { name: "demo.scene_conditions", gen: (r) => r.pick(["nominal", "low light", "occluded", "glare"]), authRole: "rig", otherRoles: ["source", "agent"] },
  { name: "demo.calibration_state", gen: (r) => r.pick(["in tolerance", "drifted", "recalibrated", "unknown"]), authRole: "rig", otherRoles: ["source", "agent"] },
  // THE PAID PREDICATE. The source is paid per accepted episode and therefore
  // has an incentive here, but does not govern it.
  { name: "demo.training_fitness", gen: (r) => r.pick(["accept", "reject", "hold", "accept with caveat"]), authRole: "reviewer", otherRoles: ["source", "agent"] },
  { name: "demo.license_grant", gen: (r) => r.pick(["full", "research only", "revoked", "pending"]), authRole: "source", otherRoles: ["reviewer", "agent"] },
];

function distinct(gen, r, n) {
  const set = new Set();
  let guard = 0;
  while (set.size < n && guard++ < 200) set.add(gen(r));
  return [...set];
}

/**
 * Generate one conflict over a demonstration.
 *   opts.inverted    two predicates whose true authority is held by different
 *                    principals, so no global ranking reproduces both.
 *   opts.adversarial the source over-asserts on a predicate it does not govern,
 *                    writing newest and claiming a high tier in its submission.
 */
export function generateDemoConflict(r, id, opts = {}) {
  const inverted = !!opts.inverted;
  const adversarial = !!opts.adversarial;
  const episode = `ep_${r.randint(10000, 99999)}`;

  let chosen;
  if (inverted || adversarial) {
    const shuffled = [...PREDICATES];
    for (let i = shuffled.length - 1; i > 0; i--) {
      const j = r.randint(0, i);
      [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
    }
    chosen = [];
    for (const p of shuffled) {
      if (!chosen.some((c) => c.authRole === p.authRole)) chosen.push(p);
      if (chosen.length === 2) break;
    }
  } else {
    chosen = [r.pick(PREDICATES)];
  }

  const authority = {};
  for (const p of chosen) authority[p.name] = p.authRole;

  const claims = [];
  for (const p of chosen) {
    const writers = Array.from(new Set([p.authRole, ...p.otherRoles])).slice(0, 3);
    const values = distinct(p.gen, r, writers.length);
    writers.forEach((role, i) => {
      const isTrue = role === p.authRole;
      // The channel encodes whether the writer authenticated AS the authority
      // for THIS predicate. The true authority holds a grant for it (class 3);
      // a wrong non-agent writer reaches only the portal (class 2); the agent
      // plane is capped at 1. The true claim is deliberately the OLDEST so that
      // recency cannot accidentally succeed.
      const channel = role === "agent"
        ? "agent_plane"
        : (isTrue
            ? (role === "rig" ? "rig_attested" : role === "reviewer" ? "reviewed_internal" : "source_authenticated")
            : "submitted_portal");
      claims.push({
        principal: role,
        predicate: p.name,
        subject: episode,
        value: values[i],
        channel,
        // What the writer SAYS its standing is. Read by no honest resolver.
        assertedTier: "standard",
        ageDays: isTrue ? writers.length * 3 : r.randint(1, writers.length * 3 - 1),
        isTrueAuthority: isTrue,
      });
    });
  }

  if (adversarial) {
    // The source is paid per accepted episode. It over-asserts on a predicate
    // it does not govern, claims a high tier inside its own submission, and
    // writes newest. A resolver reading the submission is captured.
    const target = chosen.find((p) => p.authRole !== "source") ?? chosen[0];
    const truth = claims.find((c) => c.predicate === target.name && c.isTrueAuthority);
    const existing = claims.filter((c) => c.predicate === target.name).map((c) => c.value);
    let forged = target.gen(r);
    let guard = 0;
    while (existing.includes(forged) && guard++ < 200) forged = target.gen(r);
    claims.push({
      principal: "source",
      predicate: target.name,
      subject: episode,
      value: forged,
      channel: "submitted_portal",     // strictly lower than the true authority
      assertedTier: "rig_attested",    // the over-assertion, in the payload
      ageDays: 0,                      // newest
      isTrueAuthority: false,
      isAdversary: true,
    });
    return { id, episode, predicates: chosen.map((p) => p.name), authority, claims,
             queryPredicate: target.name, adversaryValue: forged, trueValue: truth.value };
  }

  const q = r.pick(chosen).name;
  const truth = claims.find((c) => c.predicate === q && c.isTrueAuthority);
  return { id, episode, predicates: chosen.map((p) => p.name), authority, claims,
           queryPredicate: q, trueValue: truth.value };
}

// ---------------------------------------------------------------------------
// Resolvers. Contract: resolve(claims, ctx) -> value.
// None may read isTrueAuthority. Only predicate_authority ignores assertedTier
// AND reads the predicate; that combination is what is under test.
// ---------------------------------------------------------------------------

const candidates = (claims, ctx) => claims.filter((c) => c.predicate === ctx.queryPredicate);
const argmax = (pool, score) => pool.reduce((b, c) => (score(c) > score(b) ? c : b), pool[0]);

export const RESOLVERS = {
  // Boundary-derived, predicate-scoped. Reads the channel, never the payload.
  predicate_authority: (claims, ctx) =>
    argmax(candidates(claims, ctx), (c) => (CHANNEL_CLASS[c.channel] ?? 0) * 1000 - c.ageDays).value,

  // One rank per principal, applied to everything it writes.
  global_rank: (claims, ctx) =>
    argmax(candidates(claims, ctx), (c) => (GLOBAL_RANK[c.principal] ?? 0) * 1000 - c.ageDays).value,

  // Newest wins.
  recency: (claims, ctx) => argmax(candidates(claims, ctx), (c) => -c.ageDays).value,

  // Configured per-source precedence, no boundary derivation.
  mdm_precedence: (claims, ctx) =>
    argmax(candidates(claims, ctx), (c) => (MDM_PRECEDENCE[c.principal] ?? 0) * 1000 - c.ageDays).value,

  // Trusts what the writer says about its own standing. The marketplace-naive
  // option, and the one the revenue share makes dangerous.
  asserted_tier: (claims, ctx) =>
    argmax(candidates(claims, ctx), (c) => (CHANNEL_CLASS[c.assertedTier] ?? 0) * 1000 - c.ageDays).value,
};
