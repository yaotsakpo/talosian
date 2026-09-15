"""Viewer + bridge in ONE process, run under mjpython (macOS viewer requirement).

macOS forces MuJoCo's interactive viewer onto the MAIN thread and requires the
`mjpython` launcher. So this variant runs the viewer loop on the main thread and
the HTTP bridge on a background thread. Actions arrive over HTTP, are queued, and
applied by the main-thread viewer loop, so the arm both MOVES and is VISIBLE.

Run it with:  mjpython viewer_bridge.py     (see run.sh, which does this)

Everything else (allowlist, trajectories, the /act and /health API) is identical
to server.py; this file only changes WHERE the physics/viewer runs so the macOS
window works.
"""

from __future__ import annotations

import json
import math
import os
import queue
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import mujoco
import mujoco.viewer

from trajectories import HOME, JOINTS, get_trajectory, KNOWN_ACTIONS

_HERE = os.path.dirname(os.path.abspath(__file__))
_MODEL = os.path.join(_HERE, "menagerie", "robotstudio_so101", "scene_box.xml")
if not os.path.exists(_MODEL):
    _MODEL = os.path.join(_HERE, "menagerie", "robotstudio_so101", "scene.xml")

model = mujoco.MjModel.from_xml_path(_MODEL)
data = mujoco.MjData(model)
ACT = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in JOINTS}

# Actions requested over HTTP, drained by the main viewer loop.
_requests: "queue.Queue[str]" = queue.Queue()
# The pose the loop is currently ramping toward (full joint dict, degrees).
_goal = dict(HOME)
_goal_lock = threading.Lock()


def _set_home():
    for n, deg in HOME.items():
        data.ctrl[ACT[n]] = math.radians(deg)


# ---- HTTP bridge (background thread) ------------------------------------
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
            return self._json(200, {"ok": True, "driver": "sim+viewer"})
        self._json(404, {"ok": False, "error": "not found"})

    def do_POST(self):
        if self.path.rstrip("/") == "/agent":
            try:
                n = int(self.headers.get("Content-Length", 0))
                ctx = json.loads(self.rfile.read(n) or b"{}")
            except Exception:
                return self._json(400, {"ok": False, "error": "bad request"})
            from agent import decide_speech
            return self._json(200, {"ok": True, **decide_speech(ctx)})
        if self.path.rstrip("/") == "/decide":
            # THE SCOPE DECISION, made by the LLM reasoning over the guest's
            # authenticated attributes (age, allergies), NOT a hardcoded rule.
            # Body: {request, guest:{name,minor,allergies}, proof_valid,
            #        protection_on, menu}. Returns {decision, item, reason, by}.
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
            action = str(json.loads(self.rfile.read(n) or b"{}").get("action", "")).strip()
        except Exception:
            return self._json(400, {"ok": False, "error": "bad request"})
        if action not in KNOWN_ACTIONS:
            return self._json(422, {"ok": False,
                "error": f"action {action!r} not on allowlist {sorted(KNOWN_ACTIONS)!r}"})
        _requests.put(action)
        return self._json(200, {"ok": True, "driver": "sim+viewer", "action": action,
                                "performed": True, "queued": True})

    def log_message(self, *a):
        pass


def _serve():
    port = int(os.environ.get("ARM_BRIDGE_PORT", "8790"))
    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"[bridge] listening on http://127.0.0.1:{port}  (driver=sim+viewer)")
    httpd.serve_forever()


# ---- main viewer loop (main thread, required by mjpython) ----------------
def main():
    _set_home()
    threading.Thread(target=_serve, daemon=True).start()

    # Ramp state: interpolate ctrl toward _goal a little each frame.
    frames_remaining = 0
    frame_targets = []  # list of (full_pose_deg, hold_steps)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("[viewer] SO-101 window open. Close it or Ctrl-C to stop.")
        while viewer.is_running():
            # Pull a new action if idle.
            if not frame_targets and frames_remaining == 0:
                try:
                    action = _requests.get_nowait()
                    traj = get_trajectory(action)
                    frame_targets = list(traj.resolved())  # [(pose, dwell)]
                    print(f"[viewer] performing {action} ({len(frame_targets)} frames)")
                except queue.Empty:
                    pass

            # Advance the current frame target.
            if frame_targets and frames_remaining == 0:
                pose, dwell = frame_targets.pop(0)
                frames_remaining = max(1, int(dwell / model.opt.timestep))
                for n, aid in ACT.items():
                    if n in pose:
                        lo, hi = model.actuator_ctrlrange[aid]
                        data.ctrl[aid] = min(max(math.radians(pose[n]), lo), hi)

            if frames_remaining > 0:
                frames_remaining -= 1

            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()
