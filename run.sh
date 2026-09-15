#!/usr/bin/env bash
# Talosian, one-command launcher. Starts the arm bridge (MuJoCo sim by default)
# and the web console, then opens the browser. Ctrl-C stops both.
#
#   ./run.sh            # sim arm (MuJoCo window) + console
#   ARM_DRIVER=mock ./run.sh    # no arm, log only
#   ARM_DRIVER=so101 ./run.sh   # real SO-101 (on-site)
#
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ARM_DRIVER="${ARM_DRIVER:-sim}"

# The agent brain (Groq LLM) speaks for the arm. Reuse Inbin's GROQ_API_KEY if
# it is not already set, so the arm talks with a real, varied voice. Without a
# key the arm still speaks, using deterministic lines (never fabricated).
if [ -z "${GROQ_API_KEY:-}" ]; then
  INBIN_ENV="$HERE/../../.env.local"  # docs/hackathon -> docs -> inbin/.env.local
  if [ -f "$INBIN_ENV" ]; then
    GROQ_API_KEY="$(grep -E '^GROQ_API_KEY=' "$INBIN_ENV" | head -1 | cut -d= -f2- | tr -d '"'"'"'"'"'"'"'"'")"
    export GROQ_API_KEY
  fi
fi
[ -n "${GROQ_API_KEY:-}" ] && echo "[talosian] agent brain: Groq LLM (real speech)" || echo "[talosian] agent brain: deterministic fallback (no GROQ_API_KEY)"
BRIDGE_PORT="${ARM_BRIDGE_PORT:-8790}"
WEB_PORT="${WEB_PORT:-8777}"

cleanup() {
  echo "\n[talosian] stopping..."
  [ -n "${BRIDGE_PID:-}" ] && kill "$BRIDGE_PID" 2>/dev/null || true
  [ -n "${WEB_PID:-}" ] && kill "$WEB_PID" 2>/dev/null || true
  lsof -ti:"$BRIDGE_PORT" 2>/dev/null | xargs kill -9 2>/dev/null || true
  lsof -ti:"$WEB_PORT" 2>/dev/null | xargs kill -9 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Free the ports if a previous run left something behind.
lsof -ti:"$BRIDGE_PORT" 2>/dev/null | xargs kill -9 2>/dev/null || true
lsof -ti:"$WEB_PORT" 2>/dev/null | xargs kill -9 2>/dev/null || true

echo "[talosian] restaurant bridge on :$BRIDGE_PORT"
if [ -x "$HERE/arm-bridge/.venv/bin/mjpython" ]; then
  # macOS: the interactive MuJoCo window needs mjpython (viewer on main thread).
  # restaurant_bridge.py runs the two-arm restaurant viewer + the HTTP bridge
  # (/serve, /act, /decide LLM, /agent) together.
  ( cd "$HERE/arm-bridge" && ARM_BRIDGE_PORT="$BRIDGE_PORT" GROQ_API_KEY="${GROQ_API_KEY:-}" ./.venv/bin/mjpython restaurant_bridge.py ) &
else
  echo "[talosian] mjpython not found; the live viewer needs it on macOS." >&2
fi
BRIDGE_PID=$!

echo "[talosian] web console: http://localhost:$WEB_PORT/index.html"
( cd "$HERE/booth" && python3 -m http.server "$WEB_PORT" >/dev/null 2>&1 ) &
WEB_PID=$!

sleep 2
command -v open >/dev/null && open "http://localhost:$WEB_PORT/index.html" || true

echo "[talosian] running. Ctrl-C to stop."
wait
