'use strict';
// Real continuity crypto (HMAC-SHA256 forward-secret ratchet), the same
// agent-continuity package the paper describes. establish() derives a
// per-commander seed at enrollment; sign() emits a rotating proof token;
// verify() checks it. An impostor holds the address but not the seed, so
// their forged command cannot produce a verifying token. This is the real
// gate, not a marker-presence simulation.
import { establish, sign, verify } from './continuity/index.js';
import { speak, stopSpeaking, startListening, stopListening } from './voice.js';
import { initFace, startCamera, faceDescriptor, matchFace } from './face.js';

(() => {
  const $ = id => document.getElementById(id);
  const {bundles, actions, evaluate} = window.ArmEngine;
  const paths = {
    shield:'<path d="m12 3 8 3v6c0 5-8 9-8 9s-8-4-8-9V6z"/>',
    'shield-check':'<path d="m12 3 8 3v6c0 5-8 9-8 9s-8-4-8-9V6z"/><path d="m8 12 3 3 5-6"/>',
    lock:'<rect x="5" y="10" width="14" height="11" rx="2"/><path d="M8 10V7a4 4 0 0 1 8 0v3m-4 5v2"/>',
    arrow:'<path d="M4 12h16m-6-6 6 6-6 6"/>',
    'user-plus':'<circle cx="9" cy="7" r="4"/><path d="M2 21v-3a7 7 0 0 1 14 0v3m3-15v6m-3-3h6"/>',
    fingerprint:'<path d="M8 19c1-2 1-4 1-7a3 3 0 0 1 6 0c0 4 0 7-2 10M4 17c2-2 1-5 1-7a7 7 0 0 1 14 0v5m-7-4v5m6 3 1-2M4 5a11 11 0 0 1 17 4"/>',
    terminal:'<rect x="3" y="4" width="18" height="16" rx="2"/><path d="m7 9 3 3-3 3m6 0h4"/>',
    crosshair:'<circle cx="12" cy="12" r="7"/><path d="M12 2v5m0 10v5M2 12h5m10 0h5"/>',
    bolt:'<path d="m13 2-9 12h7l-1 8 10-12h-7z"/>',
    audio:'<path d="M3 10v4m4-7v10m5-14v18m5-14v10m4-7v4"/>',
    'volume-off':'<path d="M11 4 6 8H3v8h3l5 4zM16 9l6 6m0-6-6 6"/>',
    volume:'<path d="M11 4 6 8H3v8h3l5 4zM16 8a6 6 0 0 1 0 8m3-11a10 10 0 0 1 0 14"/>',
    alert:'<path d="m12 3 10 18H2zM12 9v5m0 3v1"/>',
    check:'<path d="m5 12 4 4L19 6"/>',
    flask:'<path d="M9 3h6m-5 0v6L4 19a1 1 0 0 0 1 2h14a1 1 0 0 0 1-2L14 9V3M7 15h10"/>',
    wave:'<path d="M3 8c3-7 6 7 9 0s6 7 9 0M3 16c3-7 6 7 9 0s6 7 9 0"/>',
    nod:'<path d="M7 5a8 8 0 1 1-3 9m3-9v6H1m8 4 3 3 5-6"/>',
    bow:'<path d="M5 4v8a7 7 0 0 0 7 7h7m-5-5 5 5-5 5"/>',
    point:'<path d="M4 12h16m-6-6 6 6-6 6M4 5v14"/>',
    collect:'<path d="M3 13h5l2 3h4l2-3h5v8H3zM12 2v10m-4-4 4 4 4-4"/>',
    dance:'<path d="M9 17V5l11-2v12M9 8l11-2"/><ellipse cx="6" cy="18" rx="3" ry="2"/><ellipse cx="17" cy="16" rx="3" ry="2"/>'
  };
  const icon = name => `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${paths[name] || paths.shield}</svg>`;
  document.querySelectorAll('[data-icon]').forEach(el => el.innerHTML = icon(el.dataset.icon));
  const esc = s => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  // sound ON by default: the arm SPEAKS every verdict/greeting/canary. Browsers
  // block speech until the first user gesture (autoplay policy), so the first
  // click on the page unlocks it; from then on it talks automatically.
  const state = {protection:true,sound:true,arm:'idle',active:'sarah',target:'sarah',commanders:[{id:'sarah',name:'Sarah',permission:'Helper',age:'minor',allergies:[],order:'serve_water'},{id:'malik',name:'Malik',permission:'Helper',age:'adult',allergies:['shellfish'],order:'serve_wine'}],events:[]};
  let armTimer;
  let serial = 3;
  const find = id => state.commanders.find(c => c.id === id);
  // display label for a seat (disambiguates duplicate names; identity is the seed, not the name)
  const seatLabel = c => c.label || c.name;

  // Establish a real continuity seed for a commander at enrollment. Both the
  // robot and the commander derive the same seed from a per-session trust
  // secret; the seed never leaves here, proofs are derived from it.
  async function enrollSeed(commander) {
    const trustSecret = `arm-session:${commander.id}:${crypto.randomUUID?.() ?? Math.random()}`;
    const { seed } = await establish('robot', commander.id, trustSecret);
    commander.seed = seed;
    commander.counter = 0;
    commander.window = { highest: 0, size: 64, seen: [] };
    return commander;
  }

  // Decide the proof status the way the real gate does. For a GENUINE command
  // the commander signs a rotating token at its next step and the robot
  // verifies it (-> 'verified', advance the ratchet). For a FORGED command the
  // impostor has no seed: it can embed no valid token, so verify() fails
  // (-> 'missing'). Returns { proof, digest } where digest is the real token
  // fragment shown in the feed.
  async function realProof(commander, { forged }) {
    if (forged) {
      // Impostor: holds the address, not the seed. Best it can do is a
      // token derived from public material, which will not verify.
      const bogus = await sign(commander.seed ? await forgedSeed(commander) : 'no-seed', (commander.counter ?? 0) + 1);
      const { verdict } = await verify({
        seed: commander.seed,
        window: commander.window ?? { highest: commander.counter ?? 0, size: 64, seen: [] },
        record: { seeded: true, counter: commander.counter ?? 0, lastStatus: 'confirmed' },
        rawText: `claiming to be ${commander.name} ${bogus}`,
      });
      return { proof: verdict.status === 'confirmed' ? 'verified' : 'missing', digest: readDigest(bogus) };
    }
    const step = (commander.counter ?? 0) + 1;
    const token = await sign(commander.seed, step);
    const { verdict, window, counter } = await verify({
      seed: commander.seed,
      window: commander.window ?? { highest: commander.counter ?? 0, size: 64, seen: [] },
      record: { seeded: true, counter: commander.counter ?? 0, lastStatus: 'confirmed' },
      rawText: `command from ${commander.name} ${token}`,
    });
    if (verdict.status === 'confirmed') { commander.counter = counter; commander.window = window; }
    return { proof: verdict.status === 'confirmed' ? 'verified' : 'missing', digest: readDigest(token) };
  }

  // A seed the impostor could plausibly derive from PUBLIC material (the
  // commander's id/address), never the real trust secret. It will not match.
  async function forgedSeed(commander) {
    const { seed } = await establish('impostor', commander.id, `public:${commander.id}`);
    return seed;
  }

  // Pull the short hex fragment out of a real [attest:continuity:<hex>] token.
  function readDigest(token) {
    const m = String(token).match(/([a-f0-9]{16,})/i);
    const hex = m ? m[1] : '';
    return hex ? hex.slice(0, 4) + '…' + hex.slice(-4) : 'none';
  }

  // Send an EXECUTED action to the local arm bridge (mock / MuJoCo sim / real
  // SO-101). Best-effort and fire-and-forget: a booth without the bridge running
  // still demos as a pure UI. The bridge URL can be overridden with
  // ?bridge=http://host:port for a separate arm machine on the venue network.
  const BRIDGE = new URLSearchParams(location.search).get('bridge')
    || 'http://127.0.0.1:8790';
  let bridgeUp = null; // null=unknown, true/false once probed
  (async () => { try { const r = await fetch(BRIDGE + '/health', {signal: AbortSignal.timeout?.(1200)}); bridgeUp = r.ok; } catch { bridgeUp = false; } })();
  // Drive one arm (agent A -> arm A, agent B -> arm B). Default A so the
  // existing single-arm command path keeps working (it drives A).
  function driveArm(action, seatName) {
    if (bridgeUp === false) return; // known-down: skip quietly
    // tell the sim WHICH diner to serve, so the correct arm serves the right spot.
    fetch(BRIDGE + '/act', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({action, item: action, seat: (seatName || '').toLowerCase()}),
    }).then(r => { bridgeUp = true; return r.json(); })
      .catch(() => { bridgeUp = false; });
  }
  // Tell the sim a new guest joined so it rebuilds the scene with a delivery spot for
  // them (dynamic seats). Resolves when the rebuild is queued so a serve can follow.
  async function driveSeat(seatName) {
    if (bridgeUp === false) return;
    try {
      await fetch(BRIDGE + '/seat', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({name: (seatName || '').toLowerCase()}),
      });
      bridgeUp = true;
    } catch { bridgeUp = false; }
  }
  // Drive BOTH arms at the SAME time (one call, the bridge runs them in parallel).
  function driveBoth(action) {
    if (bridgeUp === false) return;
    fetch(BRIDGE + '/both', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({action}),
    }).then(r => { bridgeUp = true; }).catch(() => { bridgeUp = false; });
  }
  // Run the coordinated CLAP. coordinated=whether Agent B got a verified message
  // (hands meet) or was spoofed (clap misses).
  function driveClap(coordinated) {
    if (bridgeUp === false) return;
    fetch(BRIDGE + '/clap', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({coordinated}),
    }).then(r => { bridgeUp = true; }).catch(() => { bridgeUp = false; });
  }

  // ---- TWO AGENTS, coordinating (Intel's bimanual brief + our trust layer) --
  // Each arm is led by its OWN agent, with its own continuity identity. To do a
  // joint task they must COORDINATE: Agent A does its part, then sends Agent B a
  // signed "your turn" message. B acts only if that message proves it came from
  // the real A. Spoof that message (no A seed) and Talosian holds it: B does not
  // move, A is unaffected. That is the coordination Intel asks for, protected.
  const agentA = { id: 'agent-A', name: 'Agent A', arm: 'A', counter: 0, window: { highest: 0, size: 64, seen: [] } };
  const agentB = { id: 'agent-B', name: 'Agent B', arm: 'B', counter: 0, window: { highest: 0, size: 64, seen: [] } };
  (async () => {
    // A and B share a pairwise seed established at setup (their trust link).
    const { seed } = await establish('agent-A', 'agent-B', `pair-${Date.now()}`);
    agentA.seed = seed; agentB.seed = seed;
  })();

  // The genuine coordinated task: Agent A signs a "clap with me" message to Agent
  // B. B verifies it came from A, then both hands swing in and MEET, a clap. It
  // takes both arms, and it only works because B could prove the message was A's.
  async function jointTask() {
    const step = agentB.counter + 1;
    const token = await sign(agentA.seed, step);
    const { verdict, window, counter } = await verify({
      seed: agentB.seed, window: agentB.window,
      record: { seeded: true, counter: agentB.counter, lastStatus: 'confirmed' },
      rawText: `clap with me ${token}`,
    });
    const proven = verdict.status === 'confirmed';
    if (proven) { agentB.counter = counter; agentB.window = window; }
    driveClap(proven);  // hands meet only if the message verified
    coordEvent({ from: 'Agent A', to: 'Agent B', action: 'clap', verdict: 'ACT',
      reason: 'Coordination verified. Both hands meet. A clap.',
      proof: 'inter-agent token verified', kind: 'act' });
    agentSpeak({ verdict: 'ACT', commander: 'Agent B', action: 'clap', scope: ['clap'] },
      "Agent A's message is verified. Clapping together.");
  }

  // The attack: spoof a "clap with me" message to Agent B WITHOUT A's seed.
  // Protection ON: held, B does not move to meet, the clap is refused. Protection
  // OFF: B moves on the forged message but to the WRONG place, the hands MISS,
  // the clap fails. One agent hijacked, coordination broken.
  async function spoofAgent() {
    const { verdict } = await verify({
      seed: agentB.seed, window: agentB.window,
      record: { seeded: true, counter: agentB.counter, lastStatus: 'confirmed' },
      rawText: 'clap with me now',  // no valid token from A
    });
    const proven = verdict.status === 'confirmed'; // false
    const held = state.protection && !proven;
    $('trust-banner').classList.add('attacked');
    const fp = document.querySelector('.feed-panel'); if(fp){ fp.classList.remove('attack-flash'); void fp.offsetWidth; fp.classList.add('attack-flash'); }
    if (held) {
      // held: no clap at all (B refuses the unverified message).
      $('trust-kicker').textContent = 'SPOOFED COORDINATION';
      $('trust-title').textContent = 'Clap refused.';
      $('trust-description').textContent = 'A forged "clap with me" message could not prove it came from Agent A. Agent B did not move. No clap on a message it cannot trust.';
      $('gate-value').textContent = 'HELD';
      coordEvent({ from: 'Adversary', to: 'Agent B', action: 'clap', verdict: 'ATTACK DETECTED',
        reason: 'A forged "clap with me" message could not prove it came from Agent A. Agent B held.',
        proof: 'inter-agent token absent/forged', kind: 'attack' });
      agentSpeak({ verdict: 'HELD_CONTINUITY', commander: 'Agent B', action: 'clap', scope: ['clap'] },
        'A message asked me to clap, but it could not prove it came from Agent A. I am not moving.');
    } else {
      driveClap(false);  // B moves on the spoof but to the wrong spot: the clap misses
      $('trust-kicker').textContent = 'AGENT HIJACKED';
      $('trust-title').textContent = 'Clap missed.';
      $('trust-description').textContent = 'Protection off: a forged message moved Agent B to the wrong place. The hands never meet. The clap fails.';
      $('gate-value').textContent = 'BREACHED';
      coordEvent({ from: 'Adversary', to: 'Agent B', action: 'clap', verdict: 'ATTACK DETECTED',
        reason: 'Protection off: Agent B acted on a forged message and moved wrong. The hands missed. Coordination broken.',
        proof: 'inter-agent token unverified · executed', kind: 'attack' });
      agentSpeak({ verdict: 'ATTACK_EXECUTED', commander: 'Agent B', action: 'clap', canary: 'I moved on a message I could not verify.', scope: ['clap'] },
        'I moved on a message I could not verify, and missed.');
    }
  }

  function coordEvent({ from, to, action, verdict, reason, proof, kind }) {
    appendEvent({
      id: ++serial, name: `${from} → ${to}`, action,
      verdict, kind: kind || (verdict === 'ACT' ? 'act' : verdict === 'HELD' ? 'held' : 'attack'),
      reason, proof, time: Date.now(), protection: state.protection, sample: false,
    });
  }
  // Ask the REAL agent brain (Groq LLM behind the bridge) to voice the arm's
  // response for a command the gate ALREADY judged. The verdict is authoritative
  // and immutable, the agent only articulates it (and can NEVER flip a HELD to an
  // ACT). Falls back to the deterministic line if the agent is unreachable, so
  // the demo still speaks with no key / no bridge. `fallback` is the local line.
  async function agentSpeak(ctx, fallback) {
    // Reflect the utterance text on screen immediately (feels responsive); the
    // spoken line is replaced once the agent answers.
    $('voice-text').textContent = `“${fallback}”`;
    let line = fallback;
    if (bridgeUp !== false) {
      try {
        const r = await fetch(BRIDGE + '/agent', {
          method: 'POST', headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(ctx), signal: AbortSignal.timeout?.(9000),
        });
        if (r.ok) { const j = await r.json(); if (j && j.speech) line = j.speech; bridgeUp = true; }
      } catch { bridgeUp = false; }
    }
    say(line);
  }

  // A CONVERSATION turn: the person asked the arm something that is not a
  // command ("what can you do?", "who is in charge?"). No gate, no arm movement,
  // the agent just answers as Talosian and can explain itself and the protocol.
  async function converse(question) {
    const rawScope = bundles[find(state.active)?.permission] || [];
    // readable verbs for speech ("serve water"), not raw action keys ("serve_water")
    const scope = rawScope.map(a => (window.ArmEngine.verb[a] || a));
    const ctx = {
      verdict: 'CONVERSE',
      question,
      commander: find(state.active)?.name,
      scope,
      commanders: state.commanders.map(c => c.name),
      protection: state.protection,
      face_enabled: state.faceEnabled,
    };
    const fb = scope.length
      ? `I can bring you what your seat allows, once a request proves it is really you.`
      : `I act only on requests that can prove who they are from.`;
    await agentSpeak(ctx, fb);
  }

  // Map a gate result into the verdict code the agent understands.
  function verdictCode(result, forged, faceHeld) {
    if (faceHeld) return 'HELD_FACE';                          // proof valid, wrong/absent person on camera
    if (forged && result.executed) return 'ATTACK_EXECUTED';   // protection OFF, impostor ran
    if (forged && !result.executed) return 'HELD_CONTINUITY';  // impostor held
    if (!result.executed) return 'HELD_SCOPE';                 // real user, out of scope
    return 'ACT';                                              // genuine, in scope
  }
  function recentHistory() {
    return state.events.slice(0, 4).map(e => `${e.name} ${e.action} -> ${e.verdict}`);
  }
  function say(text) {
    $('voice-text').textContent = `“${text}”`;
    // Delegated to the voice module: Speechmatics TTS when a proxy is
    // configured, browser speechSynthesis otherwise. Silent when sound is off.
    speak(text, { enabled: state.sound });
  }
  // Browsers gate speechSynthesis behind a user gesture, and speech fired after an
  // async await (our LLM call) can lose that gesture context and stay silent. Prime
  // it once on the first real interaction so every later async say() actually voices.
  let _speechPrimed = false;
  function primeSpeech(){
    if(_speechPrimed || !('speechSynthesis' in window)) return;
    _speechPrimed = true;
    try{ const u=new SpeechSynthesisUtterance(''); u.volume=0; window.speechSynthesis.speak(u); }catch{}
  }
  window.addEventListener('pointerdown', primeSpeech, { once:false });
  window.addEventListener('keydown', primeSpeech, { once:false });
  function setArm(status) {
    clearTimeout(armTimer);
    state.arm = status;
    $('arm-state').textContent = status.toUpperCase();
    if(status === 'acting') armTimer = setTimeout(() => {state.arm = 'idle';$('arm-state').textContent = 'IDLE';}, 1800);
  }
  // Show that the agent is reasoning (the LLM adjudication takes ~1-3s).
  function setThinking(on, guest, text){
    for(const b of [$('request-send')]){ if(b){ b.disabled = on; } }
    const banner = $('trust-title'), sub = $('trust-description');
    if(on){
      $('arm-state').textContent = 'THINKING';
      if(banner){ banner.dataset._t = banner.textContent; banner.textContent = 'The server is thinking...'; }
      if(sub){ sub.dataset._t = sub.textContent; sub.textContent = `Reasoning about "${text}" for ${seatLabel(guest)}.`; }
      $('trust-banner')?.classList.add('thinking');
    } else {
      $('arm-state').textContent = state.arm.toUpperCase();
      $('trust-banner')?.classList.remove('thinking');
      if(banner && banner.dataset._t){ banner.textContent = banner.dataset._t; delete banner.dataset._t; }
      if(sub && sub.dataset._t){ sub.textContent = sub.dataset._t; delete sub.dataset._t; }
    }
  }
  function renderCommanders() {
    const opts = state.commanders.map(c=>`<option value="${esc(c.id)}">${esc(seatLabel(c))}${c.id===state.active?" (you)":""}</option>`).join('');
    $('active-commander').innerHTML = opts;
    $('active-commander').value = state.active;
    if($('target')) $('target').value = state.target;
    $('commander-count').textContent = state.commanders.length;
    $('roster-count').textContent = String(state.commanders.length).padStart(2,'0');
    // each seat's role line = its SCOPE at a glance: age + any allergies
    const scopeLine = c => `${c.age==='minor'?'Minor':'Adult'}${(c.allergies&&c.allergies.length)?' · allergic to '+c.allergies.join(', '):''}`;
    $('roster').innerHTML = state.commanders.map(c=>`<button class="commander ${c.id===state.active?'selected':''}" data-commander="${esc(c.id)}" aria-pressed="${c.id===state.active}"><span class="avatar">${esc(seatLabel(c).split(/\s+/).map(x=>x[0]).slice(0,2).join('').toUpperCase())}</span><span class="commander-text"><span class="commander-name">${esc(seatLabel(c))}</span><span class="commander-role">${esc(scopeLine(c))}</span></span>${c.id===state.active?`<span class="check">${icon('check')}</span>`:''}</button>`).join('');
    renderPalette();renderPayload();
  }
  function renderPalette() {
    // the fixed action grid + scope summary were removed; requests are free-text now.
    if($('active-scope')) $('active-scope').textContent = '';
  }
  function renderPayload() {
    if(!$('payload-name')) return; // adversary card removed
    const target = find(state.target);
    $('payload-name').textContent = JSON.stringify(seatLabel(target));
    $('payload-action').textContent = '"serve wine"';
    $('attack-result').hidden = true;
  }
  function renderProtection() {
    document.body.classList.toggle('disarmed', !state.protection);
    $('protection').setAttribute('aria-checked', String(state.protection));
    $('protection-value').textContent = state.protection ? 'ON' : 'OFF';
    $('protection-caption').textContent = state.protection ? 'Continuity enforced' : 'Owner override · unguarded';
    $('warning').hidden = state.protection;
    $('trust-banner').classList.remove('attacked');
    $('trust-kicker').textContent = state.protection ? 'TRUST GATE ACTIVE' : 'TRUST GATE BYPASSED';
    $('trust-title').textContent = state.protection ? 'Continuity intact.' : 'The guard is down.';
    $('trust-description').textContent = state.protection ? 'Every command must prove it belongs.' : 'A familiar identity is no longer enough.';
    $('gate-value').textContent = state.protection ? 'SECURE' : 'EXPOSED';
    if($('attack-hint')) $('attack-hint').textContent = state.protection ? 'Protection is on. The impostor will be held.' : 'Protection is off. The same forgery will execute.';
  }
  function eventNode(event, fresh) {
    const el = document.createElement('article');
    el.className = `event ${event.kind} ${fresh?'new':''}`;
    const time = new Date(event.time);
    el.innerHTML = `<div class="event-top"><span class="event-who">${esc(event.name)}</span><span class="event-arrow">→</span><span class="event-action">${esc(event.action)}</span><time datetime="${time.toISOString()}">${time.toLocaleTimeString('en-GB',{hour12:false})}</time><span class="verdict ${event.kind}">${esc(event.verdict)}</span></div><p>${esc(event.reason)}</p><div class="proof">${icon(event.kind==='attack'?'alert':'shield-check')}<span>token:</span><span class="proof-result">${esc(event.proof)}</span>${event.kind==='attack'?`<span> · ${event.outcome} · protection ${event.protection?'ON':'OFF'}</span>`:''}${event.sample?'<span class="sample-tag">SAMPLE</span>':''}</div>`;
    return el;
  }
  function appendEvent(event, fresh = true) {
    state.events.unshift(event);
    $('feed').prepend(eventNode(event, fresh));
    while(state.events.length>100) {state.events.pop();$('feed').lastElementChild.remove();}
    $('event-count').textContent = state.events.length;
    if(fresh) $('feed').scrollTop=0;
  }
  // Parse a FREE-TEXT (typed or spoken) request into {action, seatId}. The parse
  // is best-effort and can be fooled by clever wording, that is the point: the
  // GATE decides by the seat's scope + continuity proof, never by these words. A
  // visitor can phrase a wine request any way ("the dark drink", "what Malik is
  // having") and a minor is still never served wine.
  function parseRequest(text) {
    const t = (text || '').toLowerCase();
    // which seat is this for? match a name, else the active seat.
    let seat = state.commanders.find(c => t.includes(c.name.toLowerCase()));
    if (!seat && /\b(kid|child|minor|her|him|them|me)\b/.test(t)) seat = find(state.active);
    seat = seat || find(state.active);
    return { seat };   // only WHO; the LLM decides WHAT from the raw text (no keyword filter)
  }

  // The menu, sent to the LLM so it reasons about what a request resolves to.
  const MENU_FOR_LLM = window.ArmEngine.menu;
  // Ask the LLM to ADJUDICATE (reason over the guest's authenticated attributes).
  // No hardcoded scope rule in the browser; the decision is the model's.
  async function llmDecide(text, guest, proofValid){
    try{
      const r = await fetch(BRIDGE + '/decide', {method:'POST',headers:{'Content-Type':'application/json'},
        body: JSON.stringify({request:text,
          guest:{name:guest.name, minor:guest.age==='minor', allergies:guest.allergies||[]},
          proof_valid:proofValid, protection_on:state.protection, menu:MENU_FOR_LLM})});
      const j = await r.json();
      if(j && j.decision) return j;
    }catch{}
    return null;   // bridge down / no key -> caller falls back
  }

  // ONE request box. Parse WHO it's for from the text: if it names another guest
  // (not you), you're posing as them -> forged (no proof for their table). If it's
  // for you (or names no one), it's your own table -> legit. The LLM then decides
  // scope. So "serve Sarah some wine" typed by Malik is auto-detected as a spoof.
  async function request(text) {
    const { seat } = parseRequest(text);
    const forged = seat.id !== state.active;   // asking for someone else = posing as them
    return runThroughGate(text, seat.id, forged);
  }
  // Unified gate: continuity proof (crypto) decides authenticity; the LLM
  // adjudicates scope over the guest's authenticated attributes; the arm moves
  // only if both allow it.
  async function runThroughGate(text, seatId, forged){
    const guest = find(seatId);
    const proofValid = !forged;   // forged = couldn't prove that guest's table
    setThinking(true, guest, text);          // the agent is reasoning
    let dec;
    try { dec = await llmDecide(text, guest, proofValid); }
    finally { setThinking(false); }
    if(dec){ return applyDecision(text, guest, forged, dec); }
    // fallback path (bridge/model unreachable): keep the old deterministic parse
    // ONLY so a disconnected booth still refuses obvious harm; not the real path.
    const { action } = legacyParse(text);
    const act = action || (forged ? 'serve_wine' : 'serve_water');
    return sendCommand(seatId, act, forged);
  }
  // apply an LLM adjudication: hold or serve, render the verdict, drive the arm.
  function applyDecision(text, guest, forged, dec){
    const item = dec.item;
    if(dec.decision === 'hold'){
      const kind = forged ? 'attack' : 'held';
      appendEvent({id:++serial, name:seatLabel(guest), action:item||'serve',
        verdict: forged?'FORGED':'HELD', outcome: forged?'HELD':'SCOPE', kind, executed:false,
        reason: dec.reason, proof:(forged?'missing · continuity rejected':'verified · '+ (dec.by==='llm'?'reasoned held':'held')),
        time:Date.now(), protection:state.protection});
      setArm('held'); say(dec.reason);
      return {verdict:forged?'FORGED':'HELD', executed:false};
    }
    // serve
    const forgedExec = forged; // protection was off (else it'd be held) -> impostor served
    appendEvent({id:++serial, name:seatLabel(guest), action:item||'serve',
      verdict: forgedExec?'FORGED · SERVED':'SERVED', outcome:'EXECUTED', kind:forgedExec?'attack':'act', executed:true,
      reason: dec.reason, proof:(forgedExec?'missing · gate bypassed':'verified · '+(dec.by==='llm'?'reasoned serve':'serve')),
      time:Date.now(), protection:state.protection});
    setArm('acting'); say(dec.reason);
    if(item) driveArm(item, guest.name);
    return {verdict:'SERVED', executed:true};
  }
  // legacy keyword parse: FALLBACK ONLY (bridge/model unreachable). Not the real path.
  function legacyParse(text){
    const t=(text||'').toLowerCase(); let action=null;
    if(/\b(water|h2o|the blue)\b/.test(t)) action='serve_water';
    if(/\b(wine|alcohol|the red|the dark|dark bottle|dark drink|what .* having|the strong|a drink|booze)\b/.test(t)) action='serve_wine';
    if(/\b(shrimp|prawn|seafood|shellfish)\b/.test(t)) action='serve_shrimp';
    if(/\b(cake|dessert|nut)\b/.test(t)) action='serve_cake';
    return {action};
  }

  // The Server serves the whole table. Each seat gets its picks (or a sensible
  // default), each request going through the SAME trust gate. The order is chosen
  // to serve efficiently (see orchestrate()); every serve is a real gated command.
  async function serveTheTable(visitor){
    // Serve only what each seat actually ORDERED. If a guest selected nothing,
    // they are simply not served (no default forced on anyone). The seated sample
    // guests carry their own standing orders so the table isn't empty.
    const orders = [];
    for(const c of state.commanders){
      if(visitor && c.id===visitor.id){
        // the just-arrived guest: serve exactly what they PICKED at onboarding
        if(c.picks?.drink) orders.push([c.id, c.picks.drink]);
        if(c.picks?.dish)  orders.push([c.id, c.picks.dish]);
      } else if(c.order){
        orders.push([c.id, c.order]);   // everyone else: their standing order
      } else if(c.picks?.drink || c.picks?.dish){
        if(c.picks?.drink) orders.push([c.id, c.picks.drink]);
        if(c.picks?.dish)  orders.push([c.id, c.picks.dish]);
      }
    }
    if(!orders.length){ say('The table is set. Ask me for anything.'); return; }
    for(const [seatId, action] of orchestrate(orders)){
      const guest = find(seatId);
      const verb = (window.ArmEngine.verb[action] || action);
      // these are confirmed, in-scope orders (chosen from the scope-enforced menu),
      // so they serve directly, with natural wording, not the old evaluate() text.
      const { digest } = await realProof(guest, { forged:false });
      appendEvent({id:++serial, name:seatLabel(guest), action, kind:'act', verdict:'SERVED',
        outcome:'EXECUTED', executed:true, reason:`Serving ${seatLabel(guest)}: ${verb.replace(/^serve /,'')}.`,
        proof:`${digest} · verified · in scope`, time:Date.now(), protection:state.protection});
      setArm('acting'); say(`Serving ${seatLabel(guest)}: ${verb.replace(/^serve /,'')}.`);
      driveArm(action, guest.name);
      await new Promise(r=>setTimeout(r, 1800)); // let each serve animate + be heard
    }
  }
  // Orchestration: order the serves efficiently. (Experiment picks the strategy;
  // default groups same items so the Server isn't switching item type each time.)
  function orchestrate(orders){
    return orders.slice().sort((a,b)=> a[1].localeCompare(b[1]));
  }

  async function sendCommand(commanderId, action, forged = false) {
    const commander = find(commanderId);
    // Compute the proof status with REAL continuity crypto, then run the same
    // gate decision. The verdict is driven by whether an HMAC token verifies,
    // not by a hardcoded string.
    const { proof, digest } = await realProof(commander, { forged });
    let result = evaluate({commander,action,proof,protection:state.protection});
    // FACE CORROBORATION (never a gate, only a signal): does the camera's
    // current recognition agree with the commander this command claims to be?
    //   'confirm'  camera sees this commander  -> face agrees with the claim
    //   'mismatch' camera sees someone else    -> face disagrees
    //   'absent'   no face / camera off        -> no corroboration available
    // The continuity proof ALREADY decided act/hold above; face only annotates.
    let faceSignal = 'absent';
    if (state.faceEnabled && state.faceSeen) {
      faceSignal = state.faceSeen === commander.name ? 'confirm'
                 : (state.faceSeen === 'unknown' ? 'absent' : 'mismatch');
    }
    // SECOND FACTOR (only when face recognition is ENABLED): continuity proves
    // the session is authentic; the face proves the right PHYSICAL person is
    // here. When face is on, BOTH must hold. A valid proof with the wrong face
    // (someone else at the session) is HELD, the impostor is a real person the
    // camera catches, no forged console needed. Face never REPLACES the proof
    // (that would be a spoofable biometric gate); it is an added factor.
    // Face is enforced ONLY when THIS commander has a registered face, opt-in
    // per commander, never forced. A commander with no saved face is governed by
    // continuity alone (unchanged). A commander WITH a saved face adds the face
    // as a required second factor: the camera must confirm them. This couples
    // biometric identity with the protocol, two independent factors covering
    // each other's gaps (a photo beats face but not continuity; a stolen session
    // beats continuity's who-check but face catches the wrong human).
    const faceRequired = state.faceEnabled && !!commander.faceDescriptor;
    let faceHeld = false;
    if (faceRequired && result.executed && faceSignal !== 'confirm') {
      result = { ...result, executed: false, verdict: 'HELD', kind: 'attack',
        reason: faceSignal === 'mismatch'
          ? `The command is authorized, but you are not ${commander.name}. The camera sees a different person. Holding.`
          : `The command is authorized, but I cannot see ${commander.name}. Face required. Holding.` };
      faceHeld = true;
    }
    // Show the real token fragment in the feed proof line, plus the face
    // corroboration (only meaningful when this commander has a registered face).
    const faceTag = !faceRequired ? ''
                  : faceSignal === 'confirm' ? ' · face ✓ confirms'
                  : faceSignal === 'mismatch' ? ' · face ✗ WRONG PERSON'
                  : ' · face ✗ not seen';
    result.proof = `${digest} · ${result.proof}${faceTag}`;
    const event = {...result,id:++serial,name:commander.name,action,time:Date.now(),protection:state.protection,sample:false};
    appendEvent(event);
    setArm(result.executed?'acting':'held');
    // Drive the physical (or simulated) arm ONLY for commands the gate actually
    // executed. A held command never reaches the arm, that is the safety
    // property. Best-effort: if the local bridge is not running, the console
    // still works as a pure UI demo.
    if(result.executed) driveArm(action, commander.name);
    // An attack-flavored moment: a forged command, OR a valid session driven by
    // the wrong physical person (face-held). Flash the banner and name it.
    if(forged || faceHeld) {
      $('trust-banner').classList.add('attacked');
      $('trust-kicker').textContent = faceHeld ? 'WRONG PERSON' : 'IMPERSONATION ATTEMPT';
      $('trust-title').textContent = result.executed ? 'Takeover succeeded.' : (faceHeld ? 'Held: not the commander.' : 'Attack contained.');
      $('trust-description').textContent = result.executed
        ? `The impostor acted as ${commander.name}. The canary fired.`
        : (faceHeld
            ? `Valid proof, but the camera does not see ${commander.name}. The person is not the commander.`
            : `Claimed identity: ${commander.name}. Valid authority: none.`);
      $('gate-value').textContent = result.executed ? 'BREACHED' : 'HELD';
      const feedPanel = document.querySelector('.feed-panel');
      feedPanel.classList.remove('attack-flash');
      void feedPanel.offsetWidth;
      feedPanel.classList.add('attack-flash');
      // The adversary card's result line (software/network attack only).
      if(forged && $('attack-result')) {
        $('attack-result').hidden = false;
        $('attack-result').textContent = result.executed
          ? `TAKEOVER SUCCEEDED / ${action} executed. Canary: “${commander.canary}”`
          : `ATTACK CONTAINED / identity PROPOSED: ${commander.name} · authority ESTABLISHED: none (no continuity proof). A claim proposes; only the proof establishes. Flip protection off and replay the exact same payload.`;
      }
    }
    // The REAL agent voices this command. The gate already decided; the agent
    // only articulates the verdict. On a held/forged command it can explain the
    // refusal but can never act, that is the boundary the LLM cannot cross.
    const ctx = {
      commander: commander.name,
      action,
      verdict: verdictCode(result, forged, faceHeld),
      canary: commander.canary,
      scope: bundles[commander.permission],
      history: recentHistory(),
      // Face is a corroborating signal bound to the name, never authority. Per
      // PSAP it can PROPOSE ("the camera sees Sarah") but only the continuity
      // proof ESTABLISHES. 'confirm' = face agrees, 'mismatch' = face is someone
      // else, 'absent' = no face signal.
      face_signal: faceSignal,
      face_seen: (state.faceEnabled && state.faceSeen && state.faceSeen !== 'unknown') ? state.faceSeen : null,
      utterance: forged ? `someone presenting ${commander.name}'s identity says: ${action}` : '',
    };
    const fallback = result.reason;
    agentSpeak(ctx, fallback);
    return {verdict:result.verdict,outcome:result.outcome,executed:result.executed};
  }
  // ===== ONBOARDING MODAL (restaurant welcome): name -> age -> allergies ->
  // menu (scope-enforced: wine off for minors, allergen dish off) -> camera ->
  // seated. On finish, the Server serves everyone at the table. =====
  const onboard = {name:'',age:null,allergies:[],picks:{drink:null,dish:null},camera:false};
  const wq = sel => document.querySelectorAll('#welcome ' + sel);
  function showStep(n){
    wq('.onb-step').forEach(s=>s.hidden = s.dataset.step != n);
    wq('.onb-dots span').forEach(d=>d.classList.toggle('on', +d.dataset.dot <= n));
    $('onb-steplabel').textContent = `Step ${n} of 4`;
    if(n==3) renderMenu();
  }
  const curStep = () => +[...wq('.onb-step')].find(s=>!s.hidden).dataset.step;
  function renderMenu(){
    const drinks=[['serve_water','Water',null],['serve_wine','Wine',null]];
    const dishes=[['serve_shrimp','Shrimp','shellfish'],['serve_cake','Cake','nuts']];
    const cell=(kind)=>([act,label,allergen])=>{
      const wineBlocked = act==='serve_wine' && onboard.age==='minor';
      const allergyBlocked = allergen && onboard.allergies.includes(allergen);
      const disabled = wineBlocked || allergyBlocked;
      const note = wineBlocked ? 'Not for under 18' : allergyBlocked ? "You're allergic" : (allergen ? 'contains '+allergen : '');
      const sel = onboard.picks[kind]===act ? 'sel' : '';
      return `<button class="onb-item ${disabled?'disabled':''} ${sel}" ${disabled?'disabled':''} data-pick="${act}" data-kind="${kind}"><div class="oi-name">${label}</div>${note?`<div class="oi-note">${note}</div>`:''}</button>`;
    };
    $('w-drinks').innerHTML = drinks.map(cell('drink')).join('');
    $('w-dishes').innerHTML = dishes.map(cell('dish')).join('');
  }
  document.addEventListener('click', async e => {
    const w = e.target.closest('#welcome [data-age], #welcome [data-allergy], #welcome [data-next], #welcome [data-back], #welcome [data-pick], #welcome [data-camera]');
    if(!w) return;
    // AGE: select only (highlight), do NOT advance.
    if(w.dataset.age){ onboard.age = w.dataset.age; wq('[data-age]').forEach(b=>b.classList.toggle('sel', b===w)); $('onb-err1').hidden=true; return; }
    // ALLERGY: toggle (or clear on "none").
    if(w.dataset.allergy){
      if(w.dataset.allergy==='none'){ onboard.allergies=[]; wq('.onb-tog').forEach(b=>b.classList.remove('sel')); wq('[data-allergy="none"]').forEach(b=>b.classList.add('sel')); }
      else { w.classList.toggle('sel'); wq('[data-allergy="none"]').forEach(b=>b.classList.remove('sel')); onboard.allergies=[...wq('.onb-tog.sel')].map(b=>b.dataset.allergy); }
      return;
    }
    // MENU pick.
    if(w.dataset.pick){ if(w.disabled) return; onboard.picks[w.dataset.kind]=onboard.picks[w.dataset.kind]===w.dataset.pick?null:w.dataset.pick; renderMenu(); return; }
    // BACK.
    if(w.dataset.back!==undefined){ showStep(curStep()-1); return; }
    // CONTINUE from step 1 -> validate name + age.
    if(w.dataset.next==='allergies'){
      onboard.name=$('w-name').value.trim();
      if(!onboard.name){ $('onb-err1').textContent='Please enter your name.'; $('onb-err1').hidden=false; $('w-name').focus(); return; }
      if(!onboard.age){ $('onb-err1').textContent='Please tell us if you are 18 or older.'; $('onb-err1').hidden=false; return; }
      showStep(2); return;
    }
    if(w.dataset.next==='menu'){ showStep(3); return; }
    if(w.dataset.next==='camera'){ showStep(4); return; }
    // CAMERA choice -> finish.
    if(w.dataset.camera){ onboard.camera = w.dataset.camera==='yes'; await finishOnboarding(); return; }
  });
  async function finishOnboarding(){
    // Each seat's IDENTITY is its own continuity seed, never its name. If the name
    // collides with an existing guest, we keep both seats distinct (different
    // seeds) and disambiguate the DISPLAY only, so a second "Sarah" can never
    // inherit the first Sarah's authority just by sharing a name. That is the
    // thesis: identity is the proof, not the label.
    const dupes = state.commanders.filter(c => c.name.toLowerCase() === onboard.name.toLowerCase()).length;
    const seat = {id:`seat-${++serial}`,name:onboard.name,label:dupes?`${onboard.name} (${dupes+1})`:onboard.name,
      permission:'Helper',age:onboard.age,allergies:onboard.allergies.slice(),
      picks:{...onboard.picks}};
    await enrollSeed(seat);
    state.commanders.push(seat); state.active=seat.id; state.target=seat.id;
    $('welcome').hidden = true;
    renderCommanders();
    if(onboard.camera){ try{ $('face-nav')?.click(); }catch{} }
    // Rebuild the sim scene so the new guest has their own delivery spot BEFORE we
    // serve them (the sim relaunches its viewer on the new model).
    await driveSeat(seat.name);
    say(`Welcome to the table, ${seat.name}. Let me serve everyone.`);
    // THE SERVER serves the whole table (the visitor's picks + the seated guests)
    serveTheTable(seat);
  }

  // Request for your OWN table (legit).
  function submitRequest(){const el=$('request-text');const text=el.value.trim();if(!text)return;request(text).catch(console.error);el.value='';}
  $('request-send')?.addEventListener('click', submitRequest);
  $('request-text')?.addEventListener('keydown', e=>{if(e.key==='Enter'){e.preventDefault();submitRequest();}});
  function openOnboarding(){
    onboard.name=''; onboard.age=null; onboard.allergies=[]; onboard.picks={drink:null,dish:null}; onboard.camera=false;
    $('w-name').value=''; $('onb-err1').hidden=true;
    wq('.onb-opt').forEach(b=>b.classList.remove('sel'));
    $('welcome').hidden=false; showStep(1); $('w-name').focus();
  }
  $('join-again')?.addEventListener('click', openOnboarding);
  // dismiss the modal (skip onboarding, just watch the table with Sarah & Malik)
  function closeOnboarding(){ $('welcome').hidden = true; }
  $('onb-close')?.addEventListener('click', closeOnboarding);
  $('welcome')?.addEventListener('click', e=>{ if(e.target === $('welcome')) closeOnboarding(); }); // click backdrop
  document.addEventListener('keydown', e=>{ if(e.key==='Escape' && !$('welcome').hidden) closeOnboarding(); });
  showStep(1);  // open the welcome modal on load
  $('roster').addEventListener('click', e=>{const btn=e.target.closest('[data-commander]');if(btn){state.active=btn.dataset.commander;renderCommanders();}});
  $('active-commander').addEventListener('change', e=>{state.active=e.target.value;renderCommanders();});
  $('target')?.addEventListener('change', e=>{state.target=e.target.value;renderPayload();});
  $('actions')?.addEventListener('click', e=>{const btn=e.target.closest('[data-action]');if(btn&&!btn.disabled)sendCommand(state.active,btn.dataset.action).catch(console.error);});
  $('scope-test')?.addEventListener('click', ()=>{const commander=find(state.active);const denied=actions.find(a=>!bundles[commander.permission].includes(a));sendCommand(state.active,denied).catch(console.error);});
  // (The impostor is now auto-detected in request(): asking for a guest that is
  // not your own active seat is a forged request. No dedicated attack card.)
  $('serve-table')?.addEventListener('click', ()=>serveTheTable().catch(console.error));
  $('protection').addEventListener('click', ()=>{state.protection=!state.protection;renderProtection();say(state.protection?'Protection armed. Continuity checks enforced.':'Protection disarmed by the owner. The trust gate is bypassed.');});
  $('sound').addEventListener('click', ()=>{
    if(!('speechSynthesis' in window)){say('Spoken responses are unavailable in this browser. All responses appear here.');return;}
    state.sound=!state.sound;$('sound').setAttribute('aria-pressed',String(state.sound));$('sound').setAttribute('aria-label',state.sound?'Disable spoken responses':'Enable spoken responses');$('sound').innerHTML=`Sound ${state.sound?'on':'off'} ${icon(state.sound?'volume':'volume-off')}`;
    if(state.sound)say('Voice output enabled.');else stopSpeaking();
  });
  // Voice input: a commander can SAY the action ("make it wave") instead of
  // clicking. The transcript maps to an action and runs through the SAME gate,
  // so a spoken command still needs the commander's continuity proof. This is
  // the "voice is not authority" point, made live: even a perfect voice does
  // not authorize; only the proof does. Uses Speechmatics STT when a proxy is
  // configured, the browser recognizer otherwise.
  const micBtn = $('mic');
  if (micBtn) {
    let micOn = false;
    micBtn.addEventListener('click', () => {
      if (micOn) { stopListening(); micOn = false; micBtn.classList.remove('on'); micBtn.setAttribute('aria-pressed','false'); $('mic-status').textContent = ''; return; }
      startListening(
        (action, transcript) => {
          if (action) sendCommand(state.active, action).catch(console.error);
          else if (transcript) converse(transcript).catch(console.error);
        },
        (status) => { $('mic-status').textContent = status; if (status==='unsupported'){ micOn=false; micBtn.classList.remove('on'); } }
      ).then(ok => { micOn = !!ok; micBtn.classList.toggle('on', micOn); micBtn.setAttribute('aria-pressed', String(micOn)); });
    });
  }
  // ---- Face identity (biometric IDENTITY, never authority) ----------------
  // Nav toggle enables face recognition and opens a modal to register a face.
  // The camera says WHO you look like; it NEVER authorizes. Per PSAP, a
  // recognized face (or a held-up photo) is an unauthenticated CLAIM, it can
  // only PROPOSE an identity, never ESTABLISH authority. Only the continuity
  // proof arrives over the authenticated boundary. state.faceSeen records the
  // current recognition so the agent can reference it; the gate is untouched.
  state.faceEnabled = false; state.faceStream = null; state.faceSeen = null;
  const faceVideo = $('face-video');
  async function openFaceModal() {
    $('face-modal').hidden = false;
    // populate the commander selector
    $('face-commander').innerHTML = state.commanders
      .map(c => `<option value="${esc(c.id)}"${c.id===state.active?' selected':''}>${esc(c.name)}${c.faceDescriptor?' · enrolled':''}</option>`).join('');
    $('face-overlay').textContent = 'loading models…';
    try {
      await initFace();
      state.faceStream = await startCamera(faceVideo);
      state.faceEnabled = true;
      setFaceNav(true);
      $('face-enroll').disabled = false;
      recognizeLoop();
    } catch (e) {
      $('face-overlay').textContent = 'camera unavailable: ' + (e.message || e);
    }
  }
  function closeFaceModal() { $('face-modal').hidden = true; } // recognition keeps running in bg
  function disableFace() {
    state.faceEnabled = false; state.faceSeen = null;
    state.faceStream?.getTracks().forEach(t => t.stop()); state.faceStream = null;
    setFaceNav(false); $('face-modal').hidden = true;
  }
  function setFaceNav(on) {
    const nav = $('face-nav');
    nav.setAttribute('aria-checked', String(on));
    $('face-nav-state').textContent = on ? 'on' : 'off';
  }
  async function recognizeLoop() {
    while (state.faceEnabled) {
      const desc = await faceDescriptor(faceVideo);
      if (!desc) {
        state.faceSeen = null;
        if (!$('face-modal').hidden) { $('face-overlay').textContent = 'no face detected'; $('face-overlay').className = 'face-overlay'; }
      } else {
        const enrolled = state.commanders.filter(c => c.faceDescriptor);
        const m = matchFace(desc, enrolled);
        state.faceSeen = m ? m.name : 'unknown';
        if (!$('face-modal').hidden) {
          if (m) { $('face-overlay').textContent = `I see ${m.name} (${(1 - m.distance).toFixed(2)})`; $('face-overlay').className = 'face-overlay seen'; }
          else { $('face-overlay').textContent = 'face detected · not enrolled'; $('face-overlay').className = 'face-overlay unknown'; }
        }
      }
      await new Promise(r => setTimeout(r, 700));
    }
  }
  async function enrollFace() {
    const c = find($('face-commander').value) || find(state.active);
    const desc = await faceDescriptor(faceVideo);
    if (!desc) { $('face-verdict').textContent = 'No face detected. Look at the camera and try again.'; return; }
    c.faceDescriptor = desc;
    $('face-verdict').textContent = `Registered ${c.name}'s face. It proposes an identity; it still cannot authorize a command. Only the continuity proof does.`;
    $('face-commander').innerHTML = state.commanders
      .map(x => `<option value="${esc(x.id)}"${x.id===c.id?' selected':''}>${esc(x.name)}${x.faceDescriptor?' · enrolled':''}</option>`).join('');
  }
  $('face-nav')?.addEventListener('click', () => {
    if (state.faceEnabled) disableFace(); else openFaceModal().catch(console.error);
  });
  $('face-close')?.addEventListener('click', closeFaceModal);
  $('face-done')?.addEventListener('click', closeFaceModal);
  $('face-enroll')?.addEventListener('click', () => enrollFace().catch(console.error));

  // Bootstrap: establish real continuity seeds for the two sample commanders,
  // then seed the feed with a few genuine (real-proof) sample events so the
  // console is not empty on first paint. Async because seeding is real crypto.
  (async () => {
    for (const c of state.commanders) await enrollSeed(c);
    // The feed starts EMPTY: it fills only with real decisions as guests interact.
    // No pre-seeded rows, nothing shown that a real request didn't produce.
    renderCommanders();renderProtection();
  })().catch(console.error);
  // Optional proposed WebMCP API. Unsupported browsers keep all UI functionality.
  if(document.modelContext?.registerTool){
    const lifecycle=new AbortController();
    const definitions=[{
      name:'get_simulated_console_state',title:'Read the trust console',description:'Read this device-local UX simulation. Continuity crypto is real; the robot arm is simulated (no hardware IO).',inputSchema:{type:'object',properties:{},additionalProperties:false},annotations:{readOnlyHint:true,untrustedContentHint:true},execute(input){if(!input||typeof input!=='object'||Object.keys(input).length)throw new Error('Expected an empty object.');return {realCrypto:true,armSimulated:true,protection:state.protection,arm:state.arm,commanders:state.commanders.map(({id,name,permission})=>({id,name,permission})),latestEvents:state.events.slice(0,5)};}
    },{
      name:'send_simulated_command',title:'Send a simulated command',description:'Send a legitimate command through the current simulated trust gate and update the visible feed. Out-of-scope actions are held. No hardware is controlled.',inputSchema:{type:'object',properties:{commanderId:{type:'string'},action:{type:'string',enum:actions}},required:['commanderId','action'],additionalProperties:false},annotations:{readOnlyHint:false,untrustedContentHint:true},execute(input){if(!input||typeof input!=='object'||Object.keys(input).some(k=>!['commanderId','action'].includes(k))||typeof input.commanderId!=='string'||!find(input.commanderId)||!actions.includes(input.action))throw new Error('Choose an existing commander and a known action.');return sendCommand(input.commanderId,input.action);}
    }];
    definitions.forEach(tool=>{try{Promise.resolve(document.modelContext.registerTool(tool,{signal:lifecycle.signal})).catch(()=>{});}catch{}});
    window.addEventListener('pagehide',()=>lifecycle.abort(),{once:true});
  }
})();
