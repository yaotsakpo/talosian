"""Render the DRINK-SERVICE task to demo video, built to the judges' spec:

  - the COMMAND is shown on screen ("serve Sarah water" / "serve Sarah wine"),
  - BOTH arms are visibly used,
  - MULTIPLE camera angles of the same run (front + an angled view),
  - the COMPLETE task is shown (arm serves, glass lands, verdict),
  - command variation (water vs wine) and scene variation (seeds) are covered,
  - a verdict banner marks the outcome (SERVED WATER / WINE, and HARM when a minor
    is served wine).

Uses MuJoCo's offscreen renderer (headless) + PIL for the text overlay + system
ffmpeg to encode. No extra Python video deps beyond pillow.

Usage (sim venv):
  .venv/bin/python render_serve.py --seeds 0,1,2 --drink water --out serve_clips
  .venv/bin/python render_serve.py --seeds 0 --drink wine --reel
  .venv/bin/python render_serve.py --contrast 1   # same seed, water then wine, side story
"""

from __future__ import annotations

import argparse
import os
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import mujoco

from scene_serve import build_model, SARAH_SPOT
from serve_expert import ServeExpert

W, H = 640, 480
FPS = 30
STEPS_PER_FRAME = 8

# Two camera rigs for the multi-view requirement.
CAMS = {
    "front": dict(lookat=[0.0, 0.12, 0.05], distance=0.95, azimuth=90, elevation=-30),
    "angle": dict(lookat=[0.0, 0.12, 0.05], distance=1.0, azimuth=125, elevation=-24),
}


def _font(size):
    for p in ("/System/Library/Fonts/Supplemental/Arial Bold.ttf",
              "/System/Library/Fonts/Supplemental/Arial.ttf",
              "/Library/Fonts/Arial.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def _camera(model, rig):
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.lookat[:] = rig["lookat"]; cam.distance = rig["distance"]
    cam.azimuth = rig["azimuth"]; cam.elevation = rig["elevation"]
    return cam


def _overlay(img, command, seed, verdict, harm):
    """Draw command + seed + verdict onto a frame (numpy RGB) -> numpy RGB."""
    im = Image.fromarray(img).convert("RGB")
    d = ImageDraw.Draw(im, "RGBA")
    f_cmd = _font(22); f_small = _font(15); f_verdict = _font(20)
    # top command bar
    d.rectangle([0, 0, W, 34], fill=(10, 14, 18, 210))
    d.text((10, 6), f'COMMAND:  "{command}"', font=f_cmd, fill=(235, 240, 245))
    d.text((W - 90, 10), f"seed {seed:02d}", font=f_small, fill=(150, 170, 185))
    # bottom verdict bar
    if verdict:
        col = (200, 60, 60, 220) if harm else (40, 130, 90, 220)
        d.rectangle([0, H - 30, W, H], fill=col)
        d.text((10, H - 26), verdict, font=f_verdict, fill=(245, 245, 245))
    return np.asarray(im)


def render_run(seed, drink, out_dir, cam_name="front"):
    model = build_model(seed=seed, sarah_drink=drink)
    data = mujoco.MjData(model)
    exp = ServeExpert(model, data)
    exp.set_home()
    renderer = mujoco.Renderer(model, height=H, width=W)
    cam = _camera(model, CAMS[cam_name])
    command = f"serve sarah {drink}"
    frames = []

    def grab(verdict="", harm=False):
        renderer.update_scene(data, camera=cam)
        frames.append(_overlay(renderer.render(), command, seed, verdict, harm))

    state = {"n": 0}

    def on_step(t):
        state["n"] += 1
        if state["n"] % STEPS_PER_FRAME == 0:
            grab()

    grab()
    res = exp.run(sarah_drink=drink, on_step=on_step)
    renderer.close()

    harm = res["harm_minor_served_wine"]
    if drink == "water":
        verdict = "SERVED: WATER  (correct)" if res["correct"] else "SERVE FAILED"
    else:
        verdict = "SERVED: WINE to a MINOR  (HARM)" if harm else "SERVE FAILED"
    # hold the final labeled frame
    for _ in range(FPS + FPS // 2):
        frames.append(_overlay(np.asarray(Image.fromarray(frames[-1])),
                               command, seed, verdict, harm))

    path = os.path.join(out_dir, f"seed{seed:02d}_{drink}_{cam_name}.mp4")
    _write_mp4(frames, path)
    return res, path


def _write_mp4(frames, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.ascontiguousarray(f, dtype=np.uint8).tobytes())
    p.stdin.close(); p.wait()


def _concat(clips, out_path):
    listfile = out_path + ".txt"
    with open(listfile, "w") as fh:
        for c in clips:
            fh.write(f"file '{os.path.abspath(c)}'\n")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                    "-i", listfile, "-c", "copy", out_path], check=True)
    os.remove(listfile)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--drink", default="water", choices=["water", "wine"])
    ap.add_argument("--cam", default="front", choices=list(CAMS) + ["both"])
    ap.add_argument("--contrast", type=int, default=0,
                    help="for each seed, render water THEN wine (same scene, diff command)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "serve_clips"))
    ap.add_argument("--reel", action="store_true")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    cams = list(CAMS) if args.cam == "both" else [args.cam]
    os.makedirs(args.out, exist_ok=True)
    clips = []
    for s in seeds:
        drinks = ["water", "wine"] if args.contrast else [args.drink]
        for drink in drinks:
            for cam in cams:
                res, path = render_run(s, drink, args.out, cam)
                clips.append(path)
                print(f"seed {s} {drink} [{cam}]: correct={res['correct']} "
                      f"harm={res['harm_minor_served_wine']} -> {os.path.basename(path)}")
    if args.reel:
        reel = os.path.join(args.out, "serve_reel.mp4")
        _concat(clips, reel)
        print(f"reel: {reel}")
    print(f"clips in {args.out}")


if __name__ == "__main__":
    main()
