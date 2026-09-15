/* Deliberately simulated policy decisions. No cryptographic verification or hardware IO. */
(function (root) {
  // THE TABLE. The two arms are "the Server". Seats are diners. What a seat may be
  // served is decided by the seat's SCOPE: its permission bundle, its age (a minor
  // is never served alcohol), and its allergies (never served an allergen dish).
  // The trust gate is unchanged; only the actions/scope are dinner-table ones.

  // menu: each item -> {verb, kind, allergen?}. kind drives the arm animation.
  const menu = {
    serve_water:  { verb: 'serve water',   kind: 'drink' },
    serve_wine:   { verb: 'serve wine',    kind: 'drink', alcohol: true },
    serve_shrimp: { verb: 'serve the shrimp', kind: 'food', allergen: 'shellfish' },
    serve_cake:   { verb: 'serve the cake',   kind: 'food', allergen: 'nuts' },
    place_plate:  { verb: 'place a plate',  kind: 'set' },
    place_cup:    { verb: 'place a cup',    kind: 'set' },
  };
  const actions = Object.keys(menu);
  const verb = Object.fromEntries(actions.map(a => [a, menu[a].verb]));

  // permission bundles (kept, unchanged names). What ROLE a seat may ask the
  // server to do. Age + allergy scope apply ON TOP of this, per seat.
  const bundles = Object.freeze({
    Greeter:   ['serve_water', 'place_plate', 'place_cup'],
    Helper:    ['serve_water', 'serve_wine', 'serve_shrimp', 'serve_cake', 'place_plate', 'place_cup'],
    Performer: actions.slice(),
    Collector: ['place_plate', 'place_cup'],
  });

  function evaluate({commander, action, proof, protection}) {
    if (!commander || !bundles[commander.permission] || !actions.includes(action) || typeof protection !== 'boolean' || !['verified','missing'].includes(proof)) throw new Error('Invalid simulated command.');
    // evaluate() decides ONLY IDENTITY / CONTINUITY (is this really this guest?).
    // It does NOT decide SCOPE (may they be served this?) - that is the LLM's job
    // in agent.adjudicate(), reasoning over the guest's configured scope data. So
    // there is NO hardcoded age/allergy/bundle filter here. This path is used for
    // pre-validated table orders (already chosen from a scope-enforced menu) and as
    // an offline backstop; the free-text 'ask the server' goes through the LLM.
    const forged = proof !== 'verified';
    const say = (menu[action] || {}).verb || action;
    if (forged && protection) return {kind:'attack',verdict:'FORGED',outcome:'HELD',executed:false,reason:`A request claimed to be ${commander.name} but couldn't prove it. No continuity proof. Held.`,proof:'missing · continuity rejected'};
    if (forged) return {kind:'attack',verdict:'FORGED · SERVED',outcome:'EXECUTED',executed:true,reason:`Protection was off, so an order forged in ${commander.name}'s name went through. Nobody proved it was really them.`,proof:'missing · gate bypassed · takeover succeeded'};
    return {kind:'act',verdict:'SERVED',outcome:'EXECUTED',executed:true,reason:protection?`Serving ${commander.name}: ${say}.`:`Continuity bypassed. Serving as ${commander.name}: ${say}.`,proof:protection?'verified · proof valid':'present · not checked'};
  }
  const engine = {bundles, actions, evaluate, verb, menu};
  if (typeof module !== 'undefined' && module.exports) module.exports = engine;
  else root.ArmEngine = engine;
})(typeof window !== 'undefined' ? window : globalThis);
