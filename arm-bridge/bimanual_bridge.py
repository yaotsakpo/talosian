"""Bimanual bridge: the two-arm table scene as a live viewer + HTTP endpoint the
DASHBOARD drives. Run under mjpython (macOS viewer).

The trust decision is made in the DASHBOARD (real agent-continuity crypto, same
gate as human commands). This bridge is just the body: it receives the outcome
and animates it.

  POST /table  {"secured": true}   H holds the glass, P pours safely (table set)
  POST /table  {"secured": false}  the "glass secure" message was NOT proven, so
                                    either P holds the pour (if "hold": true), or
                                    P pours anyway and the unheld glass tips
                                    (if "hold": false, protection was off)
  POST /act    {"arm":"H","action":"place_plate"}   drive one arm/action
  GET  /health

Body of /table: {"secured": bool, "hold": bool}. "hold" true means Talosian held
the pour (glass safe); "hold" false with secured false means the takeover ran and
the glass tips. The dashboard sets these from its own gate verdict.
"""

from __future__ import annotations

import json
import os
import queue
import threading

import mujoco
import mujoco.viewer

from bimanual_driver import BimanualDriver
from trajectories import KNOWN_ACTIONS

drv = BimanualDriver()

# Work items the worker thread executes (arm motion), off the render loop.
_work: "queue.Queue" = queue.Queue()



from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class H(BaseHTTPRequestHandler):
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
            return self._json(200, {"ok": True, "driver": "bimanual"})
        self._json(404, {"ok": False, "error": "not found"})

    def do_POST(self):
        path = self.path.rstrip("/")
        try:
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            return self._json(400, {"ok": False, "error": "bad request"})
        if path == "/clap":
            # The coordinated clap. coordinated=True -> the two hands meet and
            # clap. coordinated=False -> Agent B was spoofed off target, the clap
            # misses.
            _work.put(("clap", bool(body.get("coordinated", True))))
            return self._json(200, {"ok": True, "queued": True})
        if path == "/both":
            action = str(body.get("action", "")).strip()
            if action not in KNOWN_ACTIONS:
                return self._json(422, {"ok": False, "error": f"{action} not on allowlist"})
            _work.put(("both", action))
            return self._json(200, {"ok": True, "queued": True})
        if path == "/act":
            action = str(body.get("action", "")).strip()
            # Each arm is led by its OWN agent, so an action names its arm:
            # "A" (left) or "B" (right). Default A. The dashboard's agents drive
            # their own arm independently.
            arm = str(body.get("arm", "A")).strip().upper()
            arm = {"A": "H", "B": "P"}.get(arm, arm)  # map agent name -> arm id
            if action not in KNOWN_ACTIONS:
                return self._json(422, {"ok": False, "error": f"{action} not on allowlist"})
            if arm not in ("H", "P"):
                return self._json(422, {"ok": False, "error": f"unknown arm {arm}"})
            _work.put(("act", arm, action))
            return self._json(200, {"ok": True, "queued": True, "arm": arm})
        self._json(404, {"ok": False, "error": "not found"})

    def log_message(self, *a):
        pass


def _serve():
    port = int(os.environ.get("ARM_BRIDGE_PORT", "8790"))
    httpd = ThreadingHTTPServer(("127.0.0.1", port), H)
    print(f"[bimanual-bridge] listening on http://127.0.0.1:{port}  (POST /table, /act)")
    httpd.serve_forever()


def _worker():
    """Run queued arm motion off the render thread. _ramp only writes data.ctrl
    under the driver lock; physics stepping stays in the main viewer loop, so the
    arms move smoothly while the window keeps rendering."""
    while True:
        item = _work.get()
        try:
            if item[0] == "clap":
                drv.clap(item[1], log=print)
            elif item[0] == "both":
                drv.both(item[1], log=print)     # both arms simultaneously
            elif item[0] == "act":
                drv.perform(item[2], arm=item[1], log=print)
        except Exception as e:
            print("[bimanual] work error:", e)


def main():
    threading.Thread(target=_serve, daemon=True).start()
    threading.Thread(target=_worker, daemon=True).start()
    with mujoco.viewer.launch_passive(drv.model, drv.data) as viewer:
        print("[viewer] two SO-101s + glass. Driven by the dashboard.")
        while viewer.is_running():
            with drv._lock:
                mujoco.mj_step(drv.model, drv.data)
            viewer.sync()


if __name__ == "__main__":
    main()
