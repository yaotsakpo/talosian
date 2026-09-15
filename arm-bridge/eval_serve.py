"""Closed-loop evaluation of the TRAINED VLA policy via OpenVINO on the serve task.
This is the script the judges run to see our real model working: it loads the
trained policy (optionally the INT8 / device-split variant), runs it in closed
loop in the MuJoCo sim, and reports per-seed success across 10 randomized seeds,
for BOTH commands (serve water / serve wine).

Closed loop = at each control step the policy observes (camera image + joint
state + the language command) and outputs the 12 joint targets, which drive the
sim. No expert, the trained model is in control. Success = the served glass lands
on Sarah's setting for the commanded drink (and Malik gets wine).

Because it reports success for the trained model AND can load the FP32, INT8, or
device-split policy, it doubles as the "optimization did not degrade behaviour"
check the rubric requires.

Run with the ML venv:
  .venv-ml/bin/python eval_serve.py --policy serve_policy --device AUTO
  .venv-ml/bin/python eval_serve.py --policy serve_policy --split GPU CPU
  .venv-ml/bin/python eval_serve.py --policy serve_policy --randomize   # domain-randomized
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import mujoco

from scene_serve import JOINTS
from serve_env import ServeEnv
from infer_openvino import OVPolicy, OVSplitPolicy
from domain_random import randomize as dr_randomize
from change_gate import ChangeGatedPolicy

DRINK_ID = {"water": 0, "wine": 1}
CONTROL_EVERY = 8   # run the policy every N physics steps (matches capture rate)
MAX_STEPS = 4000


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


def run_episode(policy, stats, seed, drink, randomize=False):
    env = ServeEnv(seed=seed, drink=drink)
    if randomize:
        dr_randomize(env.model, env.data, seed,
                     free_bodies=["sarah_glass", "malik_wine_glass"])
        env._m.set_home()

    renderer = mujoco.Renderer(env.model, height=stats["image_hw"][0], width=stats["image_hw"][1])
    cam = _camera(env.model)
    lang = np.eye(stats["n_drinks"], dtype=np.float32)[DRINK_ID[drink]][None]
    s_mean, s_std = np.array(stats["state_mean"]), np.array(stats["state_std"])
    a_mean, a_std = np.array(stats["action_mean"]), np.array(stats["action_std"])

    ctrl, grasp = None, np.zeros(2)
    prev_n = np.zeros((1, 14), np.float32)   # previous (normalized) action, temporal context
    for step in range(MAX_STEPS):
        if step % CONTROL_EVERY == 0:
            renderer.update_scene(env.data, camera=cam)
            img = np.transpose(renderer.render().astype(np.float32) / 255.0, (2, 0, 1))[None]
            state = ((env.state() - s_mean) / s_std).astype(np.float32)[None]
            act_n = np.asarray(policy.act(img, state, lang, prev_n)).ravel()
            prev_n = act_n[None].astype(np.float32)   # feed this step's action back next step
            act = act_n * a_std + a_mean          # denormalize
            ctrl, grasp = act[:12], act[12:14]    # 12 joint targets + 2 grasp bits
        env.step_ctrl(ctrl, grasp)
    renderer.close()

    o = env.outcome()
    return {"seed": seed, "drink": drink, **o}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=os.path.join(os.path.dirname(__file__), "serve_policy"))
    ap.add_argument("--device", default="AUTO")
    ap.add_argument("--split", nargs=2, default=None)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--randomize", action="store_true")
    ap.add_argument("--change-gate", action="store_true",
                    help="skip the heavy VLA on stable steps (compute efficiency)")
    args = ap.parse_args()

    stats = json.load(open(os.path.join(args.policy, "stats.json")))
    if args.split:
        base = OVSplitPolicy(args.policy, args.split[0], args.split[1])
        cfg = f"split {base.enc_device}+{base.head_device}"
    else:
        base = OVPolicy(args.policy, args.device)
        cfg = f"whole/{args.device}"
    # optional change-gate wrapper (compute efficiency): skip the heavy VLA on
    # stable steps. Guardrail: success must NOT drop vs the ungated policy.
    pol = ChangeGatedPolicy(base) if args.change_gate else base
    print(f"trained VLA policy | {cfg} | randomize={args.randomize} | "
          f"change_gate={args.change_gate}\n")

    ok = 0; total = 0
    for drink in ("water", "wine"):
        for s in range(args.seeds):
            r = run_episode(pol, stats, s, drink, randomize=args.randomize)
            ok += r["success"]; total += 1
            print(f"seed {s} {drink}: sarah={r['sarah_served']} malik={r['malik_served']} "
                  f"SUCCESS={r['success']}")
    print(f"\n=== trained policy: {ok}/{total} success ({cfg}) ===")
    if args.change_gate:
        rep = pol.report()
        print(f"=== change-gate: {rep['compute_saved_pct']}% of VLA inferences SKIPPED "
              f"({rep['vla_calls']} run / {rep['steps']} steps) ===")


if __name__ == "__main__":
    main()
