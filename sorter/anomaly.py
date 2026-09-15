"""Anomalib, and turning its output into evidence the gate can use.

WHY THIS FILE EXISTS
    Anomalib is NOT part of Physical AI Studio. Studio does imitation learning (ACT,
    SmolVLA) and contains no anomaly detection at all: grepping its source for
    "anomalib" returns nothing. Intel's setup page lists them as separate installs
    alongside each other (Physical AI Studio, OpenVINO 2026.3, Anomalib v2.6.0, LeRobot),
    and the challenge slide shows them as three separate stages: ACT, Anomaly Lib,
    SmolVLA.

    So the anomaly step is something you run yourself. This is that step, plus the
    adapter that turns its prediction into the evidence dict the gate reads.

TRAINING (once, before the demo)
    Photograph cubes into two folders and train:

        cubes/
          good/       photos of good cubes
          defect/     photos of defective cubes

        python sorter/anomaly.py train --data cubes/

DURING THE DEMO
    reader = AnomalyReader("results/.../weights/onnx/model.onnx")
    ...
    GatedSource(inner=policy, evidence=lambda: reader.read(current_frame))
"""

from __future__ import annotations

import argparse
from pathlib import Path


def train(data_root: str, normal="good", abnormal="defect", model_name="Padim"):
    """Train an anomaly model on a folder of cube photos and export it for OpenVINO.

    Padim by default: it trains in minutes on a small set and needs no defective
    examples to learn from, which suits a hackathon where good cubes are plentiful and
    defective ones are whatever you can make.
    """
    from anomalib.data import Folder
    from anomalib.engine import Engine
    from anomalib import TaskType
    import anomalib.models as models

    datamodule = Folder(
        name="cubes",
        root=Path(data_root),
        normal_dir=normal,
        abnormal_dir=abnormal,
        task=TaskType.CLASSIFICATION,
    )
    model = getattr(models, model_name)()
    engine = Engine(task=TaskType.CLASSIFICATION)
    engine.fit(model=model, datamodule=datamodule)

    # scores on held-out cubes: this is what sets the gate's floor, see run_experiment
    results = engine.test(model=model, datamodule=datamodule)
    engine.export(model=model, export_type="openvino")
    return results


class AnomalyReader:
    """Reads one frame and reports it as evidence for the gate.

    Anomalib's prediction carries a `pred_score` (how anomalous, 0..1) and a
    `pred_label` (whether it crossed the model's own threshold). The gate needs both the
    verdict and how sure it is, so the score is converted into a distance from the
    decision boundary: a score right at the threshold is maximally unsure, and one far
    from it is confident.
    """

    def __init__(self, model_path: str, device: str = "CPU", threshold: float = 0.5):
        from anomalib.deploy import OpenVINOInferencer

        self._inferencer = OpenVINOInferencer(path=model_path, device=device)
        self._threshold = threshold

    def read(self, frame) -> dict:
        p = self._inferencer.predict(image=frame)
        score = float(getattr(p, "pred_score", 0.0))
        is_defective = bool(getattr(p, "pred_label", score >= self._threshold))

        # distance from the boundary, normalised to 0..1. At the threshold this is 0
        # (a coin flip); at either extreme it approaches 1.
        span = max(self._threshold, 1.0 - self._threshold) or 1.0
        confidence = min(1.0, abs(score - self._threshold) / span)

        return {
            "verdict": "defective" if is_defective else "good",
            "confidence": confidence,
            "calibrated": True,      # set False when lighting or camera has moved
            "seen_before": True,     # set False for a cube unlike the training set
            "raw_score": score,      # kept for the audit record
        }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train the cube anomaly model.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", required=True, help="folder with good/ and defect/ subfolders")
    t.add_argument("--model", default="Padim")
    a = ap.parse_args()
    if a.cmd == "train":
        print(train(a.data, model_name=a.model))
