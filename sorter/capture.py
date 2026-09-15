"""Photograph cubes into the folders Anomalib trains from.

Anomalib learns what a good cube looks like from photographs, so before any of the
sorting works someone has to take them. This is that step. It is deliberately separate
from the teleoperation demos: those record arm movements for ACT and SmolVLA, and
nobody is necessarily photographing cubes for defect detection.

    python sorter/capture.py --label good        # then press SPACE for each cube
    python sorter/capture.py --label defect

Writes into:

    cubes/
      good/     good cubes
      defect/   defective cubes

CONSISTENCY MATTERS MORE THAN QUANTITY
    Same camera, same angle, same distance, same lighting, same background, every shot.
    A model trained on photos that vary in lighting learns the lighting, not the defect,
    and will then call a perfectly good cube anomalous because a cloud went past. This
    is also why the gate checks `calibrated`: a confident score means nothing if the
    scene is no longer the one the model learned.

    Use the SAME camera that will watch the arm during the demo.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def open_camera(index: int, width: int = 640, height: int = 480):
    """Open a camera by index. Works with the RealSense units in UVC mode and with any
    ordinary webcam, since both appear to OpenCV as numbered devices. Use
    `lerobot-find-cameras` to discover which index is which."""
    try:
        import cv2
    except ImportError:
        sys.exit("opencv is needed to capture: pip install opencv-python")

    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        sys.exit(f"could not open camera {index}. Run `lerobot-find-cameras` to list them.")
    return cap


def capture(label: str, out_root: str, camera: int, auto: int, delay: float,
            width: int, height: int) -> int:
    """Take photos until told to stop. Returns how many were written."""
    import cv2

    out = Path(out_root) / label
    out.mkdir(parents=True, exist_ok=True)
    existing = len(list(out.glob("*.png")))
    cap = open_camera(camera, width, height)

    print(f"saving to {out}  ({existing} already there)")
    if auto:
        print(f"auto mode: {auto} photos, {delay}s apart. Move the cube between shots.")
    else:
        print("SPACE to capture, Q to finish.")

    taken = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("dropped a frame from the camera, retrying")
                continue

            if auto:
                time.sleep(delay)
                shoot = True
                key = -1
            else:
                preview = frame.copy()
                cv2.putText(preview, f"{label}: {existing + taken} saved   SPACE=capture  Q=quit",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow("cube capture", preview)
                key = cv2.waitKey(1) & 0xFF
                shoot = key == ord(" ")

            if shoot:
                path = out / f"{label}_{existing + taken:04d}.png"
                cv2.imwrite(str(path), frame)
                taken += 1
                print(f"  {path.name}")
                if auto and taken >= auto:
                    break
            elif key == ord("q"):
                break
    finally:
        cap.release()
        if not auto:
            cv2.destroyAllWindows()

    print(f"\n{taken} photos this session, {existing + taken} total in {out}")
    return taken


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", required=True, choices=("good", "defect"),
                    help="which folder these cubes belong in")
    ap.add_argument("--out", default="cubes", help="dataset root (default: cubes/)")
    ap.add_argument("--camera", type=int, default=0,
                    help="camera index; run `lerobot-find-cameras` to list them")
    ap.add_argument("--auto", type=int, default=0,
                    help="take N photos automatically instead of on SPACE")
    ap.add_argument("--delay", type=float, default=1.5, help="seconds between auto shots")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    a = ap.parse_args()

    capture(a.label, a.out, a.camera, a.auto, a.delay, a.width, a.height)

    root = Path(a.out)
    counts = {d.name: len(list(d.glob("*.png"))) for d in root.iterdir() if d.is_dir()}
    print("\ndataset so far:", counts or "(empty)")
    if counts.get("good", 0) < 20:
        print("  aim for at least 20 good cubes before training")
    if counts.get("defect", 0) < 5:
        print("  aim for at least 5 defective cubes, with the defect visible to the camera")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
