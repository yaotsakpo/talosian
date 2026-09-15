"""Talosian arm bridge: the local HTTP endpoint the web console POSTs an ACCEPTED
action to. The browser makes the trust decision (real continuity crypto); only a
command the gate accepted is ever sent here. This server maps action -> driver.

  POST /act    {"action": "wave"}   -> drive the arm, return the DriveResult
  GET  /health                      -> {ok, driver}

Driver is chosen by env:
  ARM_DRIVER=mock  (default) : log only, no hardware, no physics
  ARM_DRIVER=sim             : MuJoCo SO-101, a visible 3D arm that moves
  ARM_DRIVER=so101           : the real SO-101 over LeRobot (on-site)

CORS is open so the static console (served from another origin / file) can call
it. This is a LOCAL bridge on the booth machine, not a public service.

SAFETY: this server only ever performs allowlisted actions (the driver refuses
anything else). It has no authority of its own: it does not decide trust, it
executes what the trust layer already approved. It must be reachable only on the
booth's local network / localhost.
"""

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def _make_driver():
    kind = os.environ.get("ARM_DRIVER", "mock").lower()
    if kind == "sim":
        from sim_driver import SimDriver
        return "sim", SimDriver()
    if kind == "so101":
        from so101_driver import So101Driver
        return "so101", So101Driver()
    from mock_driver import MockDriver
    return "mock", MockDriver()


DRIVER_KIND, DRIVER = _make_driver()
print(f"[bridge] driver = {DRIVER_KIND}")


class Handler(BaseHTTPRequestHandler):
    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _json(self, code, body):
        payload = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self._cors()
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self):
        if self.path.rstrip("/") == "/health":
            return self._json(200, {"ok": True, "driver": DRIVER_KIND})
        self._json(404, {"ok": False, "error": "not found"})

    def do_POST(self):
        if self.path.rstrip("/") == "/agent":
            return self._agent()
        if self.path.rstrip("/") == "/decide":
            # LLM adjudicates the scope decision over the guest's authenticated
            # attributes (no hardcoded rule). See agent.adjudicate().
            try:
                n = int(self.headers.get("Content-Length", 0))
                ctx = json.loads(self.rfile.read(n) or b"{}")
            except Exception:
                return self._json(400, {"ok": False, "error": "bad request"})
            from agent import adjudicate
            return self._json(200, {"ok": True, **adjudicate(ctx)})
        if self.path.rstrip("/") != "/act":
            return self._json(404, {"ok": False, "error": "not found"})
        try:
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            action = str(body.get("action", "")).strip()
        except Exception:
            return self._json(400, {"ok": False, "error": "bad request"})
        try:
            result = DRIVER.perform(action)
        except ValueError as e:  # off the allowlist
            return self._json(422, {"ok": False, "error": str(e)})
        except Exception as e:  # hardware / sim fault
            return self._json(500, {"ok": False, "error": str(e)})
        return self._json(200, {
            "ok": True, "driver": DRIVER_KIND, "action": result.action,
            "performed": result.performed, "frames": result.frames,
            "detail": result.detail,
        })

    def _agent(self):
        """The agent brain speaks for a command the browser's gate already
        judged. The verdict in the body is authoritative; the agent only voices
        it (and, on ACT, may name the allowlisted action to perform)."""
        try:
            n = int(self.headers.get("Content-Length", 0))
            ctx = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            return self._json(400, {"ok": False, "error": "bad request"})
        from agent import decide_speech
        out = decide_speech(ctx)
        return self._json(200, {"ok": True, **out})

    def log_message(self, *args):  # quieter console
        pass


def main():
    port = int(os.environ.get("ARM_BRIDGE_PORT", "8790"))
    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"[bridge] listening on http://127.0.0.1:{port}  (POST /act, GET /health)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
