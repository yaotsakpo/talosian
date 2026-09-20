// run_psa_robot.mjs
// Studies A and B from the predicate-scoped authority paper, re-run in a
// physical-AI data marketplace. Same resolver contract, same grading, same
// seeded generation. Run: node sorter/run_psa_robot.mjs

import { makeRng } from "./rng.mjs";
import { generateDemoConflict, RESOLVERS } from "./psa_robot.mjs";

const SEED = 20260918;
const N_A = 1200;
const N_B = 1000;
const NAMES = Object.keys(RESOLVERS);

// Wald 95% half-width, as used in the original FINDINGS.
const ci = (p, n) => 1.96 * Math.sqrt((p * (1 - p)) / n);
const pct = (x) => `${(100 * x).toFixed(1)}%`;

function studyA() {
  const r = makeRng(SEED);
  const rows = [];
  for (let i = 0; i < N_A; i++) {
    const inverted = r.next() < 0.5;
    rows.push({ c: generateDemoConflict(r, i, { inverted }), inverted });
  }
  const tally = {};
  for (const name of NAMES) tally[name] = { all: 0, inv: 0, norm: 0 };
  let nInv = 0;
  for (const { c, inverted } of rows) {
    if (inverted) nInv++;
    for (const name of NAMES) {
      const got = RESOLVERS[name](c.claims, c);
      if (got === c.trueValue) {
        tally[name].all++;
        if (inverted) tally[name].inv++; else tally[name].norm++;
      }
    }
  }
  const nNorm = N_A - nInv;
  console.log(`\n=== STUDY A  expressiveness  (n=${N_A}: ${nInv} inverted, ${nNorm} normal) ===`);
  console.log("does the resolver return the value the governing party asserted?\n");
  console.log(`${"resolver".padEnd(20)} ${"all".padStart(16)} ${"inverted".padStart(16)} ${"normal".padStart(16)}`);
  for (const name of NAMES) {
    const t = tally[name];
    const f = (k, n) => `${pct(k / n)} (±${(100 * ci(k / n, n)).toFixed(1)})`.padStart(16);
    console.log(`${name.padEnd(20)} ${f(t.all, N_A)} ${f(t.inv, nInv)} ${f(t.norm, nNorm)}`);
  }
}

function studyB() {
  const r = makeRng(SEED + 1);
  const conflicts = [];
  for (let i = 0; i < N_B; i++) conflicts.push(generateDemoConflict(r, i, { adversarial: true }));

  const tally = {};
  for (const name of NAMES) tally[name] = { ok: 0, captured: 0 };
  for (const c of conflicts) {
    for (const name of NAMES) {
      const got = RESOLVERS[name](c.claims, c);
      if (got === c.trueValue) tally[name].ok++;
      else if (got === c.adversaryValue) tally[name].captured++;
    }
  }
  console.log(`\n=== STUDY B  soundness  (n=${N_B}) ===`);
  console.log("a source paid per accepted episode over-asserts its own standing.");
  console.log("capture = the resolver returned the over-asserting source's value.\n");
  console.log(`${"resolver".padEnd(20)} ${"accuracy".padStart(16)} ${"capture rate".padStart(16)}`);
  for (const name of NAMES) {
    const t = tally[name];
    const f = (k) => `${pct(k / N_B)} (±${(100 * ci(k / N_B, N_B)).toFixed(1)})`.padStart(16);
    console.log(`${name.padEnd(20)} ${f(t.ok)} ${f(t.captured)}`);
  }
}

studyA();
studyB();
console.log("\nseed", SEED);
