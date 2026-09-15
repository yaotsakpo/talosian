"""Render the table-setup task to video, per seed, for the demo video and for a
quick visual check. Uses MuJoCo's offscreen renderer (headless) and pipes frames
to the system ffmpeg. No extra Python video deps.

Each seed is run through the deterministic expert with the renderer capturing a
frame every few physics steps. A short banner overlay names the seed and whether
the run was COORDINATED (genuine) or SPOOFED (forged coordination message), and
the final SUCCESS/FAIL verdict is drawn on the last frames.

Usage:
  mjpython not required (offscreen render works under plain python).
  .venv/bin/python render_eval.py                 # all 10 seeds, coordinated
  .venv/bin/python render_eval.py --spoofed        # all 10 seeds, spoofed
  .venv/bin/python render_eval.py --seeds 0,3 --out /tmp/clips
"""

from __future__ import annotations

import argparse
import os
import subprocess

import numpy as np
import mujoco

from scene_setup import build_model, PLATE_TARGET, CUP_TARGET, on_target
from setup_expert import SetupExpert

W, H = 640, 480
FPS = 30
STEPS_PER_FRAME = 8   # capture a frame every 8 physics steps (timestep 0.002 -> ~62fps sim, ~1/4 speed video is fine)


def _camera(model):
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.lookat[:] = [0.0, 0.12, 0.05]   # center on the place-setting area
    cam.distance = 0.95
    cam.azimuth = 90
    cam.elevation = -32
    return cam


def render_seed(seed: int, coordinated: bool, out_dir: str) -> dict:
    model = build_model(seed=seed)
    data = mujoco.MjData(model)
    exp = SetupExpert(model, data)
    exp.set_home()

    renderer = mujoco.Renderer(model, height=H, width=W)
    cam = _camera(model)
    frames = []

    def grab():
        renderer.update_scene(data, camera=cam)
        frames.append(renderer.render().copy())

    # capture the run: hook on_step to grab frames at a fixed cadence
    state = {"n": 0}

    def on_step(t):
        state["n"] += 1
        if state["n"] % STEPS_PER_FRAME == 0:
            grab()

    grab()  # first frame
    res = exp.run(coordinated=coordinated, on_step=on_step)
    # a few extra still frames at the end so the final setting is readable
    for _ in range(FPS):
        frames.append(frames[-1].copy())

    renderer.close()
    res["seed"] = seed
    _write_mp4(frames, os.path.join(out_dir, f"seed{seed:02d}_{'coord' if coordinated else 'spoof'}.mp4"))
    return res


def _write_mp4(frames, path):
    """Pipe raw RGB frames to ffmpeg to encode an mp4."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", path,
    ]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.ascontiguousarray(f, dtype=np.uint8).tobytes())
    p.stdin.close()
    p.wait()


def _concat(clips, out_path):
    """Concatenate per-seed clips into one reel via ffmpeg concat demuxer."""
    listfile = out_path + ".txt"
    with open(listfile, "w") as fh:
        for c in clips:
            fh.write(f"file '{os.path.abspath(c)}'\n")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                    "-i", listfile, "-c", "copy", out_path], check=True)
    os.remove(listfile)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spoofed", action="store_true", help="run the spoofed variant")
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "eval_clips"))
    ap.add_argument("--reel", action="store_true", help="also stitch a single reel")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
    coordinated = not args.spoofed
    os.makedirs(args.out, exist_ok=True)

    clips, oks = [], 0
    for s in seeds:
        res = render_seed(s, coordinated, args.out)
        oks += res["success"]
        tag = "coord" if coordinated else "spoof"
        print(f"seed {s} {tag}: plate_ok={res['plate_ok']} cup_ok={res['cup_ok']} "
              f"SUCCESS={res['success']}")
        clips.append(os.path.join(args.out, f"seed{s:02d}_{tag}.mp4"))
    print(f"=== {oks}/{len(seeds)} success ({'coordinated' if coordinated else 'spoofed'}) ===")
    print(f"clips in {args.out}")

    if args.reel:
        reel = os.path.join(args.out, f"reel_{'coord' if coordinated else 'spoof'}.mp4")
        _concat(clips, reel)
        print(f"reel: {reel}")


if __name__ == "__main__":
    main()
