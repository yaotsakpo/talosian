"""Closed-loop evaluation of the ACTION-CHUNKING VLA policy. The policy predicts a
chunk of K actions from one observation; the env executes the chunk (open-loop)
before the policy is re-queried. This is the ACT rollout that fixes the compounding-
error freeze that single-step BC hit.

The learned policy drives the arm (no expert). Success = the served glass lands on
Sarah's setting for the commanded drink and Malik gets his wine. Reports per-seed
success for both commands.

Run with the ML venv:
  .venv-ml/bin/python eval_chunk.py --policy serve_chunk --device CPU --seeds 5
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import openvino as ov

from serve_env import ServeEnv
from eval_serve import _camera

DRINK_ID = {"water": 0, "wine": 1}
CHUNK_EXEC = 12    # sim steps to execute per predicted action within the chunk
MAX_QUERIES = 340  # policy re-queries (each advances K*CHUNK_EXEC... capped by MAX_STEPS)
MAX_STEPS = 5000


class OVChunkPolicy:
    def __init__(self, policy_dir, device="AUTO"):
        self.core = ov.Core()
        self.stats = json.load(open(os.path.join(policy_dir, "stats.json")))
        self.k = self.stats["k"]
        m = self.core.read_model(os.path.join(policy_dir, "policy.onnx"))
        self.compiled = self.core.compile_model(m, device)

    def chunk(self, image, state, lang):
        out = list(self.compiled({"image": image, "state": state, "lang": lang}).values())[0]
        return np.asarray(out).reshape(self.k, -1)   # (K,14) normalized


def run_episode(pol, seed, drink):
    env = ServeEnv(seed=seed, drink=drink)
    stats = pol.stats
    import mujoco
    renderer = mujoco.Renderer(env.model, height=stats["image_hw"][0], width=stats["image_hw"][1])
    cam = _camera(env.model)
    lang = np.eye(stats["n_drinks"], dtype=np.float32)[DRINK_ID[drink]][None]
    s_mean, s_std = np.array(stats["state_mean"]), np.array(stats["state_std"])
    a_mean, a_std = np.array(stats["action_mean"]), np.array(stats["action_std"])

    steps = 0
    for _ in range(MAX_QUERIES):
        if steps >= MAX_STEPS:
            break
        renderer.update_scene(env.data, camera=cam)
        img = np.transpose(renderer.render().astype(np.float32) / 255.0, (2, 0, 1))[None]
        state = ((env.state() - s_mean) / s_std).astype(np.float32)[None]
        chunk = pol.chunk(img, state, lang) * a_std + a_mean   # (K,14) denorm
        # execute the chunk open-loop
        for a in chunk:
            ctrl, grasp = a[:12], a[12:14]
            for _ in range(CHUNK_EXEC):
                env.step_ctrl(ctrl, grasp)
                steps += 1
                if steps >= MAX_STEPS:
                    break
            if steps >= MAX_STEPS:
                break
    renderer.close()
    return {"seed": seed, "drink": drink, **env.outcome()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=os.path.join(os.path.dirname(__file__), "serve_chunk"))
    ap.add_argument("--device", default="AUTO")
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    pol = OVChunkPolicy(args.policy, args.device)
    print(f"chunked VLA policy | device {args.device} | K={pol.k}\n")
    ok = tot = 0
    for drink in ("water", "wine"):
        for s in range(args.seeds):
            r = run_episode(pol, s, drink)
            ok += r["success"]; tot += 1
            print(f"seed {s} {drink}: sarah={r['sarah_served']} malik={r['malik_served']} "
                  f"SUCCESS={r['success']}")
    print(f"\n=== chunked policy: {ok}/{tot} success ===")


if __name__ == "__main__":
    main()
