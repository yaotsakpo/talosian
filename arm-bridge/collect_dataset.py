"""Data collection: run the deterministic table-setup expert over many seeds and
record each timestep as a (observation, action) pair, the demonstrations a policy
is trained on (behavior cloning / VLA).

Per timestep we record:
  observation:
    image  (H, W, 3) uint8   the offscreen camera frame (vision)
    state  (12,) float32     the 12 joint positions, both arms (proprioception)
  action   (12,) float32     the 12 joint control targets commanded this step

We record only COORDINATED runs (the correct behavior we want the policy to
learn), and only SUCCESSFUL episodes (the expert is 10/10, but we gate anyway so
a bad episode never poisons training).

Layout on disk (LeRobot-compatible: one dir per episode, arrays + frames):
  <out>/
    meta.json                 dataset-level metadata (dims, fps, counts)
    episode_000/
      obs_state.npy   (T,12)
      action.npy      (T,12)
      frames/000000.png ... (T frames)
    episode_001/ ...

This is deliberately framework-light so it reproduces anywhere; a small adapter
can convert it to a LeRobotDataset if we use the LeRobot training loop.

Run with the ML venv (has cv2):
  .venv-ml/bin/python collect_dataset.py --episodes 30 --out dataset
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import mujoco

from scene_setup import build_model, PLATE_TARGET, CUP_TARGET, on_target
from setup_expert import SetupExpert, JOINTS

try:
    import cv2
    def _save_png(path, rgb):
        cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
except Exception:  # fall back to a tiny PNG writer via mujoco/pillow if cv2 absent
    from PIL import Image
    def _save_png(path, rgb):
        Image.fromarray(rgb).save(path)

IMG_H, IMG_W = 240, 320   # training-resolution frames (smaller than the demo video)


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


def collect_episode(seed: int, out_dir: str, capture_every: int = 4):
    """Run one coordinated expert episode for `seed`, recording obs/action every
    `capture_every` physics steps. Returns (num_frames, success)."""
    model = build_model(seed=seed)
    data = mujoco.MjData(model)
    exp = SetupExpert(model, data)
    exp.set_home()
    saddr = _state_addrs(model)

    renderer = mujoco.Renderer(model, height=IMG_H, width=IMG_W)
    cam = _camera(model)

    frames_dir = os.path.join(out_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    states, actions = [], []
    counter = {"n": 0, "saved": 0}

    def record():
        renderer.update_scene(data, camera=cam)
        img = renderer.render()
        idx = counter["saved"]
        _save_png(os.path.join(frames_dir, f"{idx:06d}.png"), img)
        states.append(data.qpos[saddr].astype(np.float32).copy())
        actions.append(data.ctrl.astype(np.float32).copy())
        counter["saved"] += 1

    def on_step(t):
        counter["n"] += 1
        if counter["n"] % capture_every == 0:
            record()

    record()  # initial obs/action
    res = exp.run(coordinated=True, on_step=on_step)
    renderer.close()

    np.save(os.path.join(out_dir, "obs_state.npy"), np.stack(states))
    np.save(os.path.join(out_dir, "action.npy"), np.stack(actions))
    return counter["saved"], res["success"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=30, help="number of seeds to record")
    ap.add_argument("--start-seed", type=int, default=100,
                    help="first training seed (kept DISJOINT from eval seeds 0-9)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "dataset"))
    ap.add_argument("--capture-every", type=int, default=4)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    kept, total_frames = 0, 0
    for i in range(args.episodes):
        seed = args.start_seed + i
        ep_dir = os.path.join(args.out, f"episode_{kept:03d}")
        os.makedirs(ep_dir, exist_ok=True)
        n, ok = collect_episode(seed, ep_dir)
        if ok:
            kept += 1
            total_frames += n
            print(f"seed {seed}: {n} frames  SUCCESS  -> episode_{kept-1:03d}")
        else:
            # drop the failed episode's dir so only clean demos remain
            import shutil
            shutil.rmtree(ep_dir, ignore_errors=True)
            print(f"seed {seed}: {n} frames  FAILED   (dropped)")

    meta = {
        "task": "bimanual table setting (plate + cup)",
        "episodes": kept,
        "total_frames": total_frames,
        "image_hw": [IMG_H, IMG_W],
        "state_dim": 12,
        "action_dim": 12,
        "joints": list(JOINTS),
        "arms": ["H", "P"],
        "note": "coordinated successful episodes only; obs=image+12 joint pos, action=12 joint ctrl",
    }
    with open(os.path.join(args.out, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"=== kept {kept}/{args.episodes} episodes, {total_frames} frames -> {args.out} ===")


if __name__ == "__main__":
    main()
