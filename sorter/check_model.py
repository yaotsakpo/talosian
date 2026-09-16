"""Check a trained anomaly model, without Physical AI Studio.

Studio has no anomaly detection in it (its source contains no reference to anomalib), so
there is nothing in Studio that can open or test this model. This script does that job:
it finds the trained model, scores images with it, and tells you whether it actually
separates good cubes from defective ones.

    # find what training produced
    python sorter/check_model.py --find

    # score one image
    python sorter/check_model.py --model <path> --image cubes/defect/defect_0000.png

    # score the whole dataset, and write the CSV the experiment needs
    python sorter/check_model.py --model <path> --data cubes/ --csv scores.csv

The last one is the important one. It answers the question that decides whether any of
this works: **does the model give defective cubes higher scores than good ones, and by
how much?** If the two overlap completely, no threshold and no gate can help, and it is
better to find that out now than during the demo.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def find_models(root: str = ".") -> list[Path]:
    """Look for what Anomalib's export leaves behind."""
    hits: list[Path] = []
    for pattern in ("**/weights/openvino/*.xml", "**/weights/onnx/*.onnx",
                    "**/weights/torch/*.pt", "**/*.ckpt"):
        hits.extend(Path(root).glob(pattern))
    return sorted(set(hits))


def load(model_path: str, device: str = "CPU"):
    """Open a trained model, whichever format it was exported in."""
    p = Path(model_path)
    if p.suffix == ".xml":
        from anomalib.deploy import OpenVINOInferencer
        return OpenVINOInferencer(path=p, device=device)
    if p.suffix in (".pt", ".ckpt"):
        from anomalib.deploy import TorchInferencer
        return TorchInferencer(path=p)
    sys.exit(f"do not know how to open {p.suffix} files. Pass a .xml, .pt or .ckpt")


def score(inferencer, image_path: Path) -> float:
    """Score one image. Higher means more anomalous."""
    result = inferencer.predict(image=str(image_path))
    return float(getattr(result, "pred_score", 0.0))


def score_dataset(inferencer, data_root: str, csv_path: str | None):
    """Score every image and report whether the model separates the two classes."""
    root = Path(data_root)
    rows = []
    for folder, truth in (("good", "good"), ("defect", "defect"),
                          ("abnormal", "defect"), ("normal", "good")):
        d = root / folder
        if not d.is_dir():
            continue
        for img in sorted(d.glob("*.png")) + sorted(d.glob("*.jpg")):
            rows.append((score(inferencer, img), truth, img.name))

    if not rows:
        sys.exit(f"no images found under {root}. Expected {root}/good and {root}/defect")

    good = [s for s, t, _ in rows if t == "good"]
    bad = [s for s, t, _ in rows if t == "defect"]

    print(f"scored {len(rows)} images\n")
    if good:
        print(f"  good cubes    n={len(good):3}  scores {min(good):.3f} to {max(good):.3f}"
              f"  (average {sum(good)/len(good):.3f})")
    if bad:
        print(f"  defective     n={len(bad):3}  scores {min(bad):.3f} to {max(bad):.3f}"
              f"  (average {sum(bad)/len(bad):.3f})")

    if good and bad:
        print()
        gap = min(bad) - max(good)
        if gap > 0:
            print(f"  The classes do NOT overlap: every defective cube scored higher than")
            print(f"  every good one, with {gap:.3f} of clear air between them. Any")
            print(f"  threshold in that gap separates them perfectly.")
        else:
            overlapping = sum(1 for s in good if s >= min(bad)) + sum(1 for s in bad if s <= max(good))
            print(f"  The classes OVERLAP: {overlapping} of {len(rows)} images fall in the")
            print(f"  region where good and defective scores mix. That overlap is exactly")
            print(f"  what a two-bin threshold has to guess at, and what the gate holds")
            print(f"  instead. Run run_experiment.py on the CSV to size the tradeoff.")

    if csv_path:
        with open(csv_path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["score", "truth"])
            for s, t, _ in rows:
                w.writerow([f"{s:.6f}", t])
        print(f"\nwrote {csv_path}  ->  python sorter/run_experiment.py {csv_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--find", action="store_true", help="look for trained models on disk")
    ap.add_argument("--model", help="path to the trained model (.xml, .pt or .ckpt)")
    ap.add_argument("--image", help="score a single image")
    ap.add_argument("--data", help="score a whole dataset folder (good/ and defect/)")
    ap.add_argument("--csv", help="write score,truth rows here for run_experiment.py")
    ap.add_argument("--device", default="CPU", help="CPU, GPU or NPU")
    a = ap.parse_args()

    if a.find or not a.model:
        found = find_models()
        if not found:
            print("No trained model found under the current directory.")
            print("Train one first:  python sorter/anomaly.py train --data cubes/")
            return 1
        print("trained models found:")
        for f in found:
            print(f"  {f}")
        if not a.model:
            print("\nPass one with --model to score images with it.")
            return 0

    inferencer = load(a.model, a.device)

    if a.image:
        s = score(inferencer, Path(a.image))
        print(f"{a.image}: anomaly score {s:.4f}  (higher = more anomalous)")
    elif a.data:
        score_dataset(inferencer, a.data, a.csv)
    else:
        print("Nothing to do: pass --image or --data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
