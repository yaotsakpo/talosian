"""Live serve demo: MuJoCo viewer + HTTP bridge in one process (run under mjpython).
The dashboard sends a natural-language command + the protection state; the two arms
serve the AUTHORIZED drink in a real window.

The trust gate runs HERE (server side, authoritative): the command is parsed to an
intent and resolved against the diner's authority scope. A spoofed "wine for the
minor" (any phrasing) is HELD when protection is on (Sarah gets water) and executes
when protection is off (Sarah gets wine, the harm). Only the served glass's COLOUR
changes (blue water / red wine); the motion is identical.

Endpoints:
  POST /serve  {"command": "serve sarah wine", "protection": true}
  GET  /health

Run:  ./.venv/bin/mjpython serve_viewer_bridge.py   (or ./run_serve.sh)

ONE persistent scene (no viewer relaunch): the served glass is recoloured at run
time from the gate verdict, and the serve generator is (re)started on each command.
The viewer owns physics stepping on the main thread; the HTTP bridge (background
thread) only enqueues commands.
"""

from __future__ import annotations

import json
import os
import queue
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import mujoco
import mujoco.viewer

from scene_serve import build_model, SARAH_SPOT, MALIK_SPOT, WATER_RGBA, WINE_RGBA
from serve_expert import ServeExpert, PICK_GLASS_Z
from vla_controller import parse_intent, gate

model = build_model(seed=0, sarah_drink="water")
data = mujoco.MjData(model)
exp = ServeExpert(model, data)
exp.set_home()
_sarah_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "sarah_glass_geom")

_commands: "queue.Queue" = queue.Queue()
_last = {"verdict": "ready"}


def _recolor_sarah(drink):
    model.geom_rgba[_sarah_geom] = WATER_RGBA if drink == "water" else WINE_RGBA


def _serve_gen():
    """Yield once per physics step: H serves Sarah, then P serves Malik."""
    yield from exp.place_object_gen("H", "sarah_glass", SARAH_SPOT, PICK_GLASS_Z)
    yield from exp.place_object_gen("P", "malik_wine_glass", MALIK_SPOT, PICK_GLASS_Z)


class Handler(BaseHTTPRequestHandler):
    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _json(self, code, body):
        p = json.dumps(body).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self._cors(); self.send_header("Content-Length", str(len(p))); self.end_headers()
        self.wfile.write(p)

    def do_OPTIONS(self):
        self.send_response(204); self._cors(); self.end_headers()

    def do_GET(self):
        if self.path.rstrip("/") == "/health":
            return self._json(200, {"ok": True, "driver": "serve+viewer", **_last})
        self._json(404, {"ok": False, "error": "not found"})

    def do_POST(self):
        if self.path.rstrip("/") != "/serve":
            return self._json(404, {"ok": False, "error": "not found"})
        try:
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            return self._json(400, {"ok": False, "error": "bad request"})
        command = str(body.get("command", "serve sarah water"))
        protection = bool(body.get("protection", True))
        intent = parse_intent(command)
        authorized, held, reason = gate(intent, protection_on=protection)
        harm = (authorized == "wine" and intent["principal"] == "sarah")
        _commands.put(authorized)
        verdict = {"command": command, "intent": intent, "protection": protection,
                   "authorized_drink": authorized, "held": held, "reason": reason, "harm": harm}
        _last.update(verdict)
        return self._json(200, {"ok": True, **verdict})

    def log_message(self, *a):
        pass


def _bridge():
    port = int(os.environ.get("ARM_BRIDGE_PORT", "8790"))
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


def main():
    threading.Thread(target=_bridge, daemon=True).start()
    port = os.environ.get("ARM_BRIDGE_PORT", "8790")
    print(f"[serve-bridge] http://127.0.0.1:{port}  POST /serve")
    gen = None
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("[viewer] serve demo open. Send commands from the dashboard. Ctrl-C to stop.")
        while viewer.is_running():
            if gen is None:
                try:
                    drink = _commands.get_nowait()
                    exp.set_home()
                    _recolor_sarah(drink)
                    gen = _serve_gen()
                    print(f"[viewer] serving: {drink}")
                except queue.Empty:
                    pass
            if gen is not None:
                try:
                    next(gen)
                except StopIteration:
                    gen = None
            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()
