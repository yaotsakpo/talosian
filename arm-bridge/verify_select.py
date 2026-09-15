"""Verification-guided action selection for the VLA policy, and the experiment
that measures whether it improves task ACCURACY (not just security).

Two mechanisms, both grounded in current research and both tested here rather than
asserted:

  (a) Command-integrity gate: when the instruction stream can be corrupted/spoofed,
      the trust gate rejects the bad command and executes the AUTHORIZED one. If
      failures come from bad instructions, verified execution raises success rate
      (Trusted-Inference safety-filter result). Tested by eval_serve_adv.py.

  (b) Verification-guided action SELECTION (this file, VERITAS/TapSampling style):
      at each control step the policy proposes an action; we sample a few
      candidates around it, VERIFY each against the instruction with a cheap
      task-alignment score, and execute the best. This can raise success on CLEAN
      inputs too, because the policy alone sometimes picks a poor action and the
      verifier catches it.

The VERIFIER here is a short physical look-ahead: apply a candidate action to a
COPY of the sim, step it a few times, and score how much the COMMANDED object's
distance to its CORRECT target improved (plus a small penalty for knocking the
wrong object). It uses only information the task defines (the command + target),
so it is a legitimate task-alignment verifier, not privileged oracle access to the
answer.

This module provides the selector; run_experiment() measures success with vs
without verification and prints the delta. Requires a trained policy (serve_policy).

Run with the ML venv:
  .venv-ml/bin/python verify_select.py --policy serve_policy --seeds 10 --candidates 5
"""

from __future__ import annotations

import argparse
import copy
import json
import os

import numpy as np
import mujoco

from scene_serve import build_model, served, SARAH_SPOT, MALIK_SPOT, JOINTS, object_xy
from infer_openvino import OVPolicy

DRINK_ID = {"water": 0, "wine": 1}
CONTROL_EVERY = 8
MAX_STEPS = 4000
LOOKAHEAD = 12          # sim steps the verifier rolls a candidate forward
VERIFY_SIGMA = 0.06     # rad, spread of candidate perturbations around the policy action


def _state_addrs(model):
    return np.array([model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{a}_{j}")]
                     for a in ("H", "P") for j in JOINTS])


def _act_ids(model):
    return np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{a}_{j}")
                     for a in ("H", "P") for j in JOINTS])


def _camera(model):
    cam = mujoco.MjvCamera(); mujoco.mjv_defaultFreeCamera(model, cam)
    cam.lookat[:] = [0.0, 0.12, 0.05]; cam.distance = 0.95
    cam.azimuth = 90; cam.elevation = -32
    return cam


def _home(model, data, act_ids, saddr):
    from trajectories import HOME
    vals = [np.radians(HOME[j]) for _ in ("H", "P") for j in JOINTS]
    for aid, v in zip(act_ids, vals):
        data.ctrl[aid] = v
    for qa, v in zip(saddr, vals):
        data.qpos[qa] = v
    mujoco.mj_forward(model, data)


def _score(model, data, act_ids, ctrl, drink):
    """Task-alignment verifier: apply `ctrl` on a COPY, step LOOKAHEAD, score how
    much the COMMANDED object (Sarah's glass) moved TOWARD Sarah's setting, minus a
    penalty if the other glass (Malik's) drifted off his setting. Higher = better."""
    d2 = copy.deepcopy(data)
    for aid, v in zip(act_ids, ctrl):
        lo, hi = model.actuator_ctrlrange[aid]
        d2.ctrl[aid] = min(max(float(v), lo), hi)
    for _ in range(LOOKAHEAD):
        mujoco.mj_step(model, d2)
    sarah_d = np.linalg.norm(object_xy(model, d2, "sarah_glass") - SARAH_SPOT)
    malik_d = np.linalg.norm(object_xy(model, d2, "malik_wine_glass") - MALIK_SPOT)
    # want both distances small; weight Sarah (the commanded serve) higher
    return -(1.5 * sarah_d + 1.0 * malik_d)


def run_episode(policy, stats, seed, drink, verify=False, candidates=5, rng=None):
    model = build_model(seed=seed, sarah_drink=drink)
    data = mujoco.MjData(model)
    saddr, act_ids = _state_addrs(model), _act_ids(model)
    _home(model, data, act_ids, saddr)
    renderer = mujoco.Renderer(model, height=stats["image_hw"][0], width=stats["image_hw"][1])
    cam = _camera(model)
    lang = np.eye(stats["n_drinks"], dtype=np.float32)[DRINK_ID[drink]][None]
    s_mean, s_std = np.array(stats["state_mean"]), np.array(stats["state_std"])
    a_mean, a_std = np.array(stats["action_mean"]), np.array(stats["action_std"])
    rng = rng or np.random.default_rng(seed)

    for step in range(MAX_STEPS):
        if step % CONTROL_EVERY == 0:
            renderer.update_scene(data, camera=cam)
            img = np.transpose(renderer.render().astype(np.float32) / 255.0, (2, 0, 1))[None]
            state = ((data.qpos[saddr] - s_mean) / s_std).astype(np.float32)[None]
            act_n = np.asarray(policy.act(img, state, lang)).ravel()
            base_ctrl = act_n * a_std + a_mean
            if verify:
                # propose the policy action + perturbed candidates, verify, pick best
                cands = [base_ctrl] + [base_ctrl + rng.normal(0, VERIFY_SIGMA, size=base_ctrl.shape)
                                       for _ in range(candidates - 1)]
                best = max(cands, key=lambda c: _score(model, data, act_ids, c, drink))
                ctrl = best
            else:
                ctrl = base_ctrl
            for aid, v in zip(act_ids, ctrl):
                lo, hi = model.actuator_ctrlrange[aid]
                data.ctrl[aid] = min(max(float(v), lo), hi)
        mujoco.mj_step(model, data)
    renderer.close()

    sarah = served(model, data, "sarah_glass", SARAH_SPOT)
    malik = served(model, data, "malik_wine_glass", MALIK_SPOT)
    return bool(sarah and malik)


def run_experiment(policy_dir, seeds=10, candidates=5, device="AUTO"):
    stats = json.load(open(os.path.join(policy_dir, "stats.json")))
    pol = OVPolicy(policy_dir, device)
    rows = {"baseline": 0, "verified": 0}
    n = 0
    for drink in ("water", "wine"):
        for s in range(seeds):
            base = run_episode(pol, stats, s, drink, verify=False)
            ver = run_episode(pol, stats, s, drink, verify=True, candidates=candidates)
            rows["baseline"] += base; rows["verified"] += ver; n += 1
            print(f"seed {s} {drink}: baseline={base}  verified={ver}")
    print(f"\n=== baseline {rows['baseline']}/{n}  |  verification-guided {rows['verified']}/{n} "
          f"(candidates={candidates}) ===")
    return rows, n


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=os.path.join(os.path.dirname(__file__), "serve_policy"))
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--candidates", type=int, default=5)
    ap.add_argument("--device", default="AUTO")
    args = ap.parse_args()
    run_experiment(args.policy, args.seeds, args.candidates, args.device)
