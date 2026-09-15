"""Restaurant viewer + bridge in one process (run under mjpython). The MuJoCo
window shows the two arms; the dashboard drives them. When the dashboard's trust
gate SERVES a request, it posts the item + diner here and the arms physically serve
it. Held requests never post here, so a held request produces no motion.

Seats are DYNAMIC: when a guest joins from the dashboard, /seat rebuilds the scene
(spec.recompile returns a NEW model/data, which we swap into the viewer and rebind
the driver to). Delivery spots are placed inside each arm's measured clean band, so
any number of guests is served reliably (no hand-picked coordinates).

Endpoints:
  POST /serve  {"item":"serve_wine","diner":"malik"}    arms serve that item to that diner
  POST /act    {"action":"serve_wine","seat":"malik"}    alias (what the dashboard posts)
  POST /seat   {"name":"yao"}                            add a guest, rebuild the scene
  POST /decide {...}                                     LLM adjudication (agent.adjudicate)
  POST /agent  {...}                                     LLM speech (agent.decide_speech)
  GET  /health

Run:  ./.venv/bin/mjpython restaurant_bridge.py    (needs GROQ_API_KEY for /decide)

The viewer loop (main thread, mjpython requirement) advances the serve generators one
physics step per frame and applies pending scene rebuilds, so the arms MOVE and new
seats APPEAR live. HTTP runs on a background thread and only enqueues jobs.
"""

from __future__ import annotations

import json
import os
import queue
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import mujoco
import mujoco.viewer

from scene_restaurant import make_spec, DEFAULT_DINERS, MAX_SEATS
from restaurant_driver import RestaurantDriver

# live, mutable scene state. The viewer thread owns model/data/drv; HTTP only posts
# jobs (serve or rebuild) onto the queue so there is no cross-thread model mutation.
_diners = list(DEFAULT_DINERS)
_spec, _ = make_spec(_diners)
model = _spec.compile()
data = mujoco.MjData(model)
drv = RestaurantDriver(model, data, _diners)

_jobs: "queue.Queue" = queue.Queue()   # ("serve", item, diner) | ("seat", name, None)


def _known(name):
    """Resolve a seat display name to a current diner id (exact, else first-name)."""
    name = (name or "").lower()
    if name in _diners:
        return name
    return next((d for d in _diners if d in name or name in d), None)


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
            return self._json(200, {"ok": True, "driver": "restaurant+viewer", "diners": _diners})
        self._json(404, {"ok": False, "error": "not found"})

    def _read(self):
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n) or b"{}")

    def do_POST(self):
        path = self.path.rstrip("/")
        try:
            body = self._read()
        except Exception:
            return self._json(400, {"ok": False, "error": "bad request"})
        if path in ("/serve", "/act"):
            item = body.get("item") or body.get("action")
            diner = _known(body.get("diner") or body.get("seat") or "")
            if not item or diner is None:
                return self._json(422, {"ok": False, "error": "need item + known diner"})
            _jobs.put(("serve", item, diner))
            return self._json(200, {"ok": True, "queued": True, "item": item, "diner": diner})
        if path == "/seat":
            name = (body.get("name") or body.get("id") or "").lower().strip()
            if not name:
                return self._json(422, {"ok": False, "error": "need name"})
            _jobs.put(("seat", name, None))
            return self._json(200, {"ok": True, "queued": True, "seat": name})
        if path == "/request":
            # THE GATED ENTRY POINT. A request PROPOSES; only the adjudication at this
            # boundary can ESTABLISH authority to move the arm. The decision is made here,
            # from the guest's authenticated standing, and the arm job is queued ONLY on a
            # serve. A hold returns the reason and queues nothing, so a held request
            # produces no motion at all: that is the demo, and it is enforced here rather
            # than trusted to the caller. (/act stays for direct control, but the dashboard
            # goes through this path so the gate is never bypassed.)
            from agent import adjudicate
            decision = adjudicate(body)
            diner = _known(body.get("diner") or body.get("seat")
                           or (body.get("guest") or {}).get("name") or "")
            served = decision.get("decision") == "serve"
            item = decision.get("item")
            queued = False
            if served and item and diner is not None:
                _jobs.put(("serve", item, diner))
                queued = True
            return self._json(200, {"ok": True, "queued": queued, "diner": diner, **decision})
        if path == "/decide":
            from agent import adjudicate
            return self._json(200, {"ok": True, **adjudicate(body)})
        if path == "/agent":
            from agent import decide_speech
            return self._json(200, {"ok": True, **decide_speech(body)})
        self._json(404, {"ok": False, "error": "not found"})

    def log_message(self, *a):
        pass


def _serve_http():
    port = int(os.environ.get("ARM_BRIDGE_PORT", "8790"))
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()


def _light_slot(i):
    """Light the marker of slot i (a seated guest). Done on the LIVE model's site_rgba,
    no recompile, so it is safe while the single viewer is open."""
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"slot_{i}")
    if sid >= 0:
        model.site_rgba[sid] = [0.5, 0.6, 0.7, 0.35]


def main():
    global model, data, drv, _diners
    threading.Thread(target=_serve_http, daemon=True).start()
    print(f"[restaurant-bridge] http://127.0.0.1:{os.environ.get('ARM_BRIDGE_PORT','8790')}  POST /serve, /act, /seat")
    active = None   # the current serve generator

    # ONE viewer for the whole session. mjpython allows only a single viewer on the UI
    # thread, so seating a guest must NOT recompile/relaunch. The model is built once
    # with the full seat-slot pool (markers for every slot); a join just assigns the
    # guest to the next fixed slot (drv.rebind recomputes delivery spots on the SAME
    # model/data) and lights that slot's marker. No new model, no second viewer.
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # open on the fixed "scene" camera (the framed front view), not the default free
        # cam, so the live view matches the intended composition.
        scam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "scene")
        if scam >= 0:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = scam
        print(f"[viewer] restaurant open, seats: {_diners}. Ctrl-C to stop.")
        while viewer.is_running():
            if active is None:
                try:
                    kind, a, b = _jobs.get_nowait()
                    if kind == "serve":
                        active = drv.serve_gen(a, b)
                        print(f"[viewer] serving {a} -> {b}")
                    elif kind == "seat" and a not in _diners and len(_diners) < MAX_SEATS:
                        _diners.append(a)
                        drv.rebind(model, data, _diners)   # same model/data, new spots
                        _light_slot(len(_diners) - 1)
                        print(f"[viewer] seated {a} at slot {len(_diners)-1}; seats now {_diners}")
                    elif kind == "seat" and len(_diners) >= MAX_SEATS:
                        print(f"[viewer] table full ({MAX_SEATS} seats); ignoring {a}")
                except queue.Empty:
                    pass
            if active is not None:
                try:
                    next(active)
                except StopIteration:
                    active = None
            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()
