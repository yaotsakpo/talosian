"""Run a Studio-trained policy on the arm, with the trust gate in the loop.

WHY THIS EXISTS RATHER THAN A STUDIO PATCH
    Studio's plugin system only registers robots (see application/backend/src/plugins:
    PluginRobot, role follower/leader). There is no supported extension point for
    governing actions, and the only seam is `self._policy = source` inside
    StudioActionSource._set_policy. Patching that means running a modified Studio.

    But Studio exports a trained policy to a directory on disk, and the code that turns
    that directory into a running policy is small and public. So the honest option is to
    leave Studio untouched, use it for what it is good at (setup, teleoperation,
    recording, training), and run the trained policy here with the gate in the loop.

    It also gives the demo something Studio's UI cannot: a protection on/off switch, so
    the same part can be shown going to the customer bin and then being held.

CONSTRUCTION
    Mirrors Studio's own, read from application/backend/src/runtime/config_builder.py
    (policy_source_from_fragment), so the policy runs exactly as Studio would run it.

USAGE
    python sorter/run_gated.py --export-dir  ~/models/<id>/exports/openvino \\
                              --follower-port /dev/ttyACM0 \\
                              [--protection off]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from gated_source import GatedSource


def build_policy_source(export_dir: str, backend: str = "openvino", device: str = "CPU",
                        task: str | None = None):
    """Build a PolicySource the way Studio does.

    Copied from Studio's policy_source_from_fragment rather than reimplemented, so a
    policy behaves here exactly as it does inside Studio.
    """
    from physicalai.inference import InferenceModel
    from physicalai.runtime import (ChunkedActionQueue, LerpSmoother, PolicySource,
                                    SyncExecution)

    kwargs = {}
    if task:
        kwargs["task"] = task
    return PolicySource(
        model=InferenceModel(export_dir=export_dir, policy_name=None,
                             backend=backend, device=device),
        execution=SyncExecution(request_threshold=1),
        action_queue=ChunkedActionQueue(smoother=LerpSmoother()),
        **kwargs,
    )


def anomaly_reading_stub():
    """Stand-in until Anomalib is wired in.

    Replace with the real reading:

        def anomaly_reading():
            r = anomalib_model.predict(current_frame)
            return {"verdict": "defective" if r.pred_label else "good",
                    "confidence": float(r.pred_score),
                    "calibrated": True, "seen_before": True}
    """
    return {"verdict": "good", "confidence": 0.99, "calibrated": True, "seen_before": True}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export-dir", required=True, help="Studio's exported policy directory")
    ap.add_argument("--follower-port", required=True, help="e.g. /dev/ttyACM0")
    ap.add_argument("--backend", default="openvino")
    ap.add_argument("--device", default="CPU", help="CPU, GPU or NPU")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--task", default=None, help="task string for a language-conditioned policy")
    ap.add_argument("--protection", choices=("on", "off"), default="on",
                    help="off reproduces today's pipeline: the claim reaches the motors unchecked")
    args = ap.parse_args()

    from physicalai.robot.so101 import SO101          # adjust if the import path differs
    from physicalai.runtime import RobotRuntime

    source = build_policy_source(args.export_dir, args.backend, args.device, args.task)

    decisions = []
    gated = GatedSource(
        inner=source,
        evidence=anomaly_reading_stub,
        on_decision=decisions.append,
        protection_on=(args.protection == "on"),
    )

    print(f"policy:     {args.export_dir}")
    print(f"backend:    {args.backend} on {args.device}")
    print(f"protection: {args.protection.upper()}"
          + ("" if args.protection == "on" else "   (claims reach the motors unchecked)"))

    runtime = RobotRuntime(
        fps=args.fps,
        robot=SO101(port=args.follower_port),
        action_source=gated,
    )
    try:
        runtime.run()
    except KeyboardInterrupt:
        pass
    finally:
        held = [d for d in decisions if not d["allowed"]]
        print(f"\n{len(decisions)} decisions, {len(held)} held")
        for d in held[:5]:
            print(f"  held: {d['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
