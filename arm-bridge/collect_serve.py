"""Data collection for the DRINK-SERVICE task. Runs the deterministic serve expert
over many seeds AND both drink commands, recording each timestep as a
(observation, action) pair plus the LANGUAGE command, so the policy is a true
Vision-Language-Action model:

  observation:
    image    (H,W,3) uint8    offscreen camera frame          (Vision)
    state    (12,)  float32   12 joint positions, both arms    (proprioception)
  language:
    command  str              e.g. "serve sarah water"         (Language)
    drink_id int              0=water, 1=wine (encoded command)
  action     (12,) float32    12 joint control targets         (Action)

We record BOTH sarah_drink outcomes (water and wine) so the policy learns that the
SAME scene + a DIFFERENT command produces a different serve. That is what makes it
language-conditioned: the drink is dictated by the command, not the pixels.

Only successful episodes are kept (the expert is 10/10, gated anyway).

Layout (LeRobot-compatible: one dir per episode + arrays + frames + command):
  <out>/meta.json
  <out>/episode_000/{obs_state.npy, action.npy, drink.npy, command.txt, frames/*.png}

Run with the ML venv:
  .venv-ml/bin/python collect_serve.py --episodes 30 --out serve_dataset
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

import numpy as np
import mujoco

from scene_serve import build_model, JOINTS
from serve_expert import ServeExpert

try:
    import cv2
    def _save_png(path, rgb):
        cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
except Exception:
    from PIL import Image
    def _save_png(path, rgb):
        Image.fromarray(rgb).save(path)

IMG_H, IMG_W = 240, 320
DRINK_ID = {"water": 0, "wine": 1}


def _state_addrs(model):
    addrs = []
    for arm in ("H", "P"):
        for j in JOINTS:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{arm}_{j}")
            addrs.append(model.jnt_qposadr[jid])
    return np.array(addrs)


def _camera(model):
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.lookat[:] = [0.0, 0.12, 0.05]
    cam.distance = 0.95
    cam.azimuth = 90
    cam.elevation = -32
    return cam


def collect_episode(seed: int, drink: str, out_dir: str, capture_every: int = 8):
    model = build_model(seed=seed, sarah_drink=drink)
    data = mujoco.MjData(model)
    exp = ServeExpert(model, data)
    exp.set_home()
    saddr = _state_addrs(model)

    renderer = mujoco.Renderer(model, height=IMG_H, width=IMG_W)
    cam = _camera(model)
    frames_dir = os.path.join(out_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    states, actions, drinks, grasps = [], [], [], []
    counter = {"n": 0, "saved": 0}
    did = DRINK_ID[drink]

    # grasp signal per arm: 1 if that arm's gripper is currently welded to an
    # object. The weld is toggled by the expert (eq_active); we record it as part
    # of the ACTION so the policy learns WHEN to grasp/release. In closed loop the
    # env turns the predicted grasp bit into the weld (a real gripper command).
    # Map each weld's obj to the arm serving it: sarah_glass->H, malik_wine->P.
    def grasp_bits():
        h = p = 0.0
        for obj, eid in exp.weld.items():
            if data.eq_active[eid]:
                if obj.startswith("sarah") or obj == "water_glass":
                    h = 1.0
                else:
                    p = 1.0
        return np.array([h, p], dtype=np.float32)

    def record():
        renderer.update_scene(data, camera=cam)
        img = renderer.render()
        _save_png(os.path.join(frames_dir, f"{counter['saved']:06d}.png"), img)
        states.append(data.qpos[saddr].astype(np.float32).copy())
        actions.append(data.ctrl.astype(np.float32).copy())
        grasps.append(grasp_bits())
        drinks.append(did)
        counter["saved"] += 1

    def on_step(t):
        counter["n"] += 1
        if counter["n"] % capture_every == 0:
            record()

    record()
    res = exp.run(sarah_drink=drink, on_step=on_step)
    renderer.close()

    np.save(os.path.join(out_dir, "obs_state.npy"), np.stack(states))
    np.save(os.path.join(out_dir, "action.npy"), np.stack(actions))
    np.save(os.path.join(out_dir, "grasp.npy"), np.stack(grasps))
    np.save(os.path.join(out_dir, "drink.npy"), np.array(drinks, dtype=np.int64))
    with open(os.path.join(out_dir, "command.txt"), "w") as fh:
        fh.write(f"serve sarah {drink}\n")
    return counter["saved"], res["correct"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=30, help="seeds per drink (x2 drinks)")
    ap.add_argument("--start-seed", type=int, default=100,
                    help="first training seed (kept disjoint from eval seeds 0-9)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "serve_dataset"))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    kept, total_frames = 0, 0
    for i in range(args.episodes):
        seed = args.start_seed + i
        for drink in ("water", "wine"):
            ep_dir = os.path.join(args.out, f"episode_{kept:03d}")
            os.makedirs(ep_dir, exist_ok=True)
            n, ok = collect_episode(seed, drink, ep_dir)
            if ok:
                kept += 1; total_frames += n
                print(f"seed {seed} {drink}: {n} frames OK -> episode_{kept-1:03d}")
            else:
                shutil.rmtree(ep_dir, ignore_errors=True)
                print(f"seed {seed} {drink}: {n} frames FAILED (dropped)")

    meta = {
        "task": "bimanual drink service (Sarah minor=water, Malik adult=wine)",
        "episodes": kept, "total_frames": total_frames,
        "image_hw": [IMG_H, IMG_W], "state_dim": 12, "action_dim": 12,
        "language": "command drink_id: 0=water 1=wine",
        "joints": list(JOINTS), "arms": ["H", "P"],
        "note": "VLA: obs=image+12 joint pos, language=drink command, action=12 joint ctrl; both drinks per seed",
    }
    with open(os.path.join(args.out, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"=== kept {kept} episodes, {total_frames} frames -> {args.out} ===")


if __name__ == "__main__":
    main()
