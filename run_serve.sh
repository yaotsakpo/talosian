#!/usr/bin/env bash
# Talosian serve demo launcher. Starts the MuJoCo viewer + serve bridge and the
# web console, then opens the serve dashboard. Ctrl-C stops both.
#
#   ./run_serve.sh
#
# In the dashboard: type a command (or click an attack), toggle PROTECTION, and
# hit "Send to arms". Watch the two arms serve the authorized drink in the window.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
BRIDGE_PORT="${ARM_BRIDGE_PORT:-8790}"
WEB_PORT="${WEB_PORT:-8777}"

cleanup(){
  echo ""; echo "[serve] stopping..."
  [ -n "${BRIDGE_PID:-}" ] && kill "$BRIDGE_PID" 2>/dev/null || true
  [ -n "${WEB_PID:-}" ] && kill "$WEB_PID" 2>/dev/null || true
  lsof -ti:"$BRIDGE_PORT" 2>/dev/null | xargs kill -9 2>/dev/null || true
  lsof -ti:"$WEB_PORT" 2>/dev/null | xargs kill -9 2>/dev/null || true
}
trap cleanup EXIT INT TERM
lsof -ti:"$BRIDGE_PORT" 2>/dev/null | xargs kill -9 2>/dev/null || true
lsof -ti:"$WEB_PORT" 2>/dev/null | xargs kill -9 2>/dev/null || true

echo "[serve] viewer + bridge on :$BRIDGE_PORT (mjpython)"
( cd "$HERE/arm-bridge" && ARM_BRIDGE_PORT="$BRIDGE_PORT" ./.venv/bin/mjpython serve_viewer_bridge.py ) &
BRIDGE_PID=$!

echo "[serve] web console: http://localhost:$WEB_PORT/serve.html"
( cd "$HERE/booth" && python3 -m http.server "$WEB_PORT" >/dev/null 2>&1 ) &
WEB_PID=$!

sleep 2
command -v open >/dev/null && open "http://localhost:$WEB_PORT/serve.html" || true
echo "[serve] running. Ctrl-C to stop."
wait
