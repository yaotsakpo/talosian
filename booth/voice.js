// Talosian voice: the arm SPEAKS its verdicts and LISTENS for spoken commands.
//
// Two-tier by design, so the booth is robust:
//   1. Speechmatics (the sponsor tech): a natural low-latency TTS voice and
//      real-time STT. Used when a proxy is configured (VOICE_PROXY), because a
//      static page cannot safely hold the API key, a tiny server mints a
//      temporary key / proxies TTS. See voice-proxy/ (added later).
//   2. Browser built-ins: speechSynthesis (TTS) + webkitSpeechRecognition (STT).
//      Always available, no key, no network, offline. This is the fallback and
//      what runs until Speechmatics is wired.
//
// The rest of the app calls speak(text) and startListening(onAction) and does
// not care which tier serves them. If Speechmatics is unavailable at any point,
// we silently fall back so voice never breaks the demo.

const PROXY = new URLSearchParams(location.search).get('voice')
  || window.TALOSIAN_VOICE_PROXY
  || null; // e.g. "http://127.0.0.1:8795" when the Speechmatics proxy is running

let smAvailable = null; // null=unprobed, true/false once checked
async function probeSpeechmatics() {
  if (!PROXY) return (smAvailable = false);
  if (smAvailable !== null) return smAvailable;
  try {
    const r = await fetch(PROXY + '/health', { signal: AbortSignal.timeout?.(1200) });
    smAvailable = r.ok;
  } catch { smAvailable = false; }
  return smAvailable;
}

// ---- SPEAK ---------------------------------------------------------------

let currentAudio = null;
// True while the arm is speaking, the mic ignores transcripts during this and
// for a short tail after, so it never hears (and re-commands from) its own voice.
let speaking = false;
let speakingUntil = 0;
export function isSpeaking() { return speaking || Date.now() < speakingUntil; }
function markSpeaking(on) { speaking = on; if (!on) speakingUntil = Date.now() + 700; }

export async function speak(text, { enabled }) {
  if (!enabled || !text) return;
  markSpeaking(true);
  // Tier 1: Speechmatics TTS via the proxy (returns audio bytes we play).
  if (await probeSpeechmatics()) {
    try {
      const r = await fetch(PROXY + '/tts', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      });
      if (r.ok) {
        const buf = await r.arrayBuffer();
        stopSpeaking();
        const ctx = new (window.AudioContext || window.webkitAudioContext)();
        const audio = await ctx.decodeAudioData(buf);
        const src = ctx.createBufferSource();
        src.buffer = audio; src.connect(ctx.destination); src.start();
        src.onended = () => markSpeaking(false);
        currentAudio = { ctx, src };
        return;
      }
    } catch { /* fall through to browser */ }
  }
  // Tier 2: browser speechSynthesis.
  if ('speechSynthesis' in window) {
    const synth = window.speechSynthesis;
    // Chrome bug: the speech engine silently pauses (no sound, no error), and stays
    // paused after ~15s idle or when speak() is called from an async context. resume()
    // before speaking, plus a keep-alive resume ping while speaking, works around it.
    synth.cancel();
    synth.resume();
    const u = new SpeechSynthesisUtterance(text);
    u.rate = 0.98; u.pitch = 1.0;
    const v = pickVoice();
    if (v) u.voice = v;
    let keepAlive = setInterval(() => { try { synth.resume(); } catch {} }, 4000);
    const done = () => { clearInterval(keepAlive); markSpeaking(false); };
    u.onend = done;
    u.onerror = done;
    synth.speak(u);
    setTimeout(() => { try { synth.resume(); } catch {} }, 100); // kick Chrome if it parked
    return;
  }
  // No TTS tier available: clear the flag so the mic is not stuck muted.
  markSpeaking(false);
}

export function stopSpeaking() {
  if ('speechSynthesis' in window) window.speechSynthesis.cancel();
  if (currentAudio) { try { currentAudio.src.stop(); } catch {} currentAudio = null; }
}

// Prefer a clear, natural English voice when the browser offers a choice.
let cachedVoice;
function pickVoice() {
  if (cachedVoice !== undefined) return cachedVoice;
  const voices = window.speechSynthesis?.getVoices?.() || [];
  const prefer = [/Samantha/i, /Daniel/i, /Google US English/i, /en[-_]US/i, /English/i];
  for (const p of prefer) {
    const hit = voices.find(v => p.test(v.name) || p.test(v.lang));
    if (hit) return (cachedVoice = hit);
  }
  return (cachedVoice = voices[0] || null);
}
// getVoices() populates async in some browsers.
if ('speechSynthesis' in window) {
  window.speechSynthesis.onvoiceschanged = () => { cachedVoice = undefined; pickVoice(); };
}

// ---- LISTEN --------------------------------------------------------------
// Map a free spoken phrase to one of the six actions. "make it wave" -> wave.
const ACTION_WORDS = ['wave', 'nod', 'bow', 'point', 'collect', 'dance'];
export function phraseToAction(text) {
  const t = (text || '').toLowerCase();
  // direct word match first
  for (const a of ACTION_WORDS) if (t.includes(a)) return a;
  // a few natural synonyms
  if (/\b(pick ?up|grab|fetch)\b/.test(t)) return 'collect';
  if (/\b(salute|hello|hi)\b/.test(t)) return 'wave';
  if (/\b(yes|agree)\b/.test(t)) return 'nod';
  return null;
}

let recognition = null;
export async function startListening(onAction, onStatus) {
  // Tier 1: Speechmatics real-time STT via the proxy would open a WebSocket with
  // a temporary key from PROXY + '/token'. Wired when the proxy is added; until
  // then we use the browser recognizer below so speech input works today.
  // (Intentionally left as the documented seam: buildSpeechmaticsSession(PROXY).)

  // Tier 2: browser SpeechRecognition.
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) { onStatus?.('unsupported'); return false; }
  recognition = new SR();
  recognition.lang = 'en-US';
  recognition.continuous = true;
  recognition.interimResults = false;
  recognition.onresult = (e) => {
    // Ignore anything the mic picks up while the arm is speaking (or for the
    // short tail after) so it never re-commands itself from its own TTS.
    if (isSpeaking()) return;
    const last = e.results[e.results.length - 1];
    const text = (last[0].transcript || '').trim();
    const action = phraseToAction(text);
    onStatus?.(`heard: "${text}"${action ? ' -> ' + action : ''}`);
    // A recognized action runs as a COMMAND (through the gate). Anything else is
    // a CONVERSATION turn: pass the raw transcript so the app can ask the agent.
    onAction(action, text);
  };
  recognition.onerror = (e) => onStatus?.('mic error: ' + e.error);
  recognition.onend = () => { if (listening) { try { recognition.start(); } catch {} } };
  try { recognition.start(); listening = true; onStatus?.('listening'); return true; }
  catch (e) { onStatus?.('mic error: ' + e.message); return false; }
}

let listening = false;
export function stopListening() {
  listening = false;
  if (recognition) { try { recognition.stop(); } catch {} recognition = null; }
}
