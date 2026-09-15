"""OpenVINO inference for the VLA policy, with heterogeneous device execution
across Intel Core Ultra's CPU / iGPU / NPU, and HMAC verification of the
cross-device handoff (our trust principle carried down to the accelerator
boundary).

Two ways to run the policy:

  1. WHOLE-MODEL on one device (or AUTO/HETERO): compile policy.onnx and infer.
     Device is any OpenVINO target: CPU, GPU (iGPU), NPU, AUTO, "HETERO:NPU,CPU".

  2. SPLIT across two devices (the heterogeneous optimization): the vision CNN
     encoder runs on one device (e.g. NPU/GPU, built for vision) and the fused
     action head on another (e.g. CPU, low-latency for a tiny MLP). The image
     embedding crosses the device boundary; we HMAC-TAG it on the producer side
     and VERIFY the tag on the consumer side, so a tampered/reordered handoff is
     caught before the control head acts on it. Same continuity-proof idea as the
     command gate, one layer down.

On a non-Intel dev box only CPU is present, so devices fall back to CPU and the
code still runs end-to-end (proving correctness); on Core Ultra it lights up
iGPU + NPU. Nothing here is Intel-only at build time.

Run with the ML venv:
  .venv-ml/bin/python infer_openvino.py --policy serve_policy --device AUTO
  .venv-ml/bin/python infer_openvino.py --policy serve_policy --split GPU CPU
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os

import numpy as np
import openvino as ov

# A per-session key for the handoff HMAC. In the real trust layer this comes from
# the agent-continuity ratchet; here a fixed dev key keeps inference reproducible.
_HANDOFF_KEY = b"talosian-xpu-handoff-v1"


def _tag(arr: np.ndarray) -> str:
    return hmac.new(_HANDOFF_KEY, arr.tobytes(), hashlib.sha256).hexdigest()


def _split_onnx(onnx_path: str, out_dir: str):
    """Split the VLA ONNX into (vision-encoder, action-head) sub-models so each can
    target a different device. The encoder outputs the flattened image embedding;
    the head takes [embedding, state, lang] -> action. Uses onnx utils."""
    import onnx
    from onnx.utils import Extractor
    model = onnx.load(onnx_path)
    # find the embedding tensor: output of the CNN Flatten (named by export order).
    # We locate the Concat that fuses [feat, state, lang]; its first input is feat.
    concat = next(n for n in model.graph.node if n.op_type == "Concat")
    feat_name = concat.input[0]
    # the head takes the image embedding plus every NON-image graph input
    # (state, lang, and prev if present), so it works whatever the input set is.
    non_image = [i.name for i in model.graph.input if i.name != "image"]
    ex = Extractor(model)
    enc = ex.extract_model(["image"], [feat_name])
    head = ex.extract_model([feat_name] + non_image, ["action"])
    os.makedirs(out_dir, exist_ok=True)
    enc_p = os.path.join(out_dir, "encoder.onnx")
    head_p = os.path.join(out_dir, "head.onnx")
    onnx.save(enc, enc_p); onnx.save(head, head_p)
    return enc_p, head_p, feat_name, non_image


class OVPolicy:
    """Whole-model OpenVINO policy on one device (or AUTO/HETERO)."""

    def __init__(self, policy_dir: str, device: str = "AUTO"):
        self.core = ov.Core()
        self.stats = json.load(open(os.path.join(policy_dir, "stats.json")))
        m = self.core.read_model(os.path.join(policy_dir, "policy.onnx"))
        self.compiled = self.core.compile_model(m, device)
        self.device = device

    def act(self, image, state, lang, prev=None):
        if prev is None:
            prev = np.zeros((1, 14), np.float32)
        out = self.compiled({"image": image, "state": state, "lang": lang, "prev": prev})
        return list(out.values())[0]


class OVSplitPolicy:
    """VLA split across two devices with a VERIFIED cross-device handoff.
    encoder -> (embedding, HMAC tag) -> [verify] -> head."""

    def __init__(self, policy_dir: str, enc_device: str = "GPU", head_device: str = "CPU"):
        self.core = ov.Core()
        self.stats = json.load(open(os.path.join(policy_dir, "stats.json")))
        onnx_path = os.path.join(policy_dir, "policy.onnx")
        split_dir = os.path.join(policy_dir, "split")
        enc_p, head_p, self.feat_name, self.head_inputs = _split_onnx(onnx_path, split_dir)
        avail = self.core.available_devices
        # gracefully fall back to CPU for any device not present on this box
        enc_device = enc_device if enc_device in avail else "CPU"
        head_device = head_device if head_device in avail else "CPU"
        self.enc = self.core.compile_model(self.core.read_model(enc_p), enc_device)
        self.head = self.core.compile_model(self.core.read_model(head_p), head_device)
        self.enc_device, self.head_device = enc_device, head_device
        self.tamper_detected = 0

    def act(self, image, state, lang, prev=None, tamper=False):
        if prev is None:
            prev = np.zeros((1, 14), np.float32)
        # producer (enc_device): run the vision encoder, TAG the embedding
        feat = list(self.enc({"image": image}).values())[0]
        tag = _tag(feat)
        if tamper:  # simulate a compromised handoff: flip the embedding after tagging
            feat = feat + 0.5
        # consumer (head_device): VERIFY the embedding before trusting it
        if _tag(feat) != tag:
            self.tamper_detected += 1
            raise RuntimeError("xpu handoff verification FAILED: vision embedding tampered")
        # feed the head its image embedding plus whichever non-image inputs it has
        avail = {"state": state, "lang": lang, "prev": prev}
        feed = {self.feat_name: feat}
        for name in self.head_inputs:
            feed[name] = avail[name]
        out = self.head(feed)
        return list(out.values())[0]


def _dummy_inputs(stats):
    h, w = stats["image_hw"]
    return (np.zeros((1, 3, h, w), np.float32),
            np.zeros((1, 12), np.float32),
            np.eye(stats["n_drinks"], dtype=np.float32)[0:1])  # one-hot water


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=os.path.join(os.path.dirname(__file__), "serve_policy"))
    ap.add_argument("--device", default="AUTO")
    ap.add_argument("--split", nargs=2, metavar=("ENC", "HEAD"), default=None,
                    help="split encoder/head across two devices, e.g. --split GPU CPU")
    args = ap.parse_args()

    print("OpenVINO", ov.__version__, "devices:", ov.Core().available_devices)
    img, st, lang = _dummy_inputs(json.load(open(os.path.join(args.policy, "stats.json"))))

    if args.split:
        pol = OVSplitPolicy(args.policy, args.split[0], args.split[1])
        print(f"SPLIT: encoder on {pol.enc_device}, head on {pol.head_device}, HMAC-verified handoff")
        a = pol.act(img, st, lang)
        print("action[0:3]", np.asarray(a).ravel()[:3].round(3))
        # demonstrate the verification catching a tampered handoff
        try:
            pol.act(img, st, lang, tamper=True)
        except RuntimeError as e:
            print("tamper test:", e)
    else:
        pol = OVPolicy(args.policy, args.device)
        print(f"WHOLE model on {args.device}")
        a = pol.act(img, st, lang)
        print("action[0:3]", np.asarray(a).ravel()[:3].round(3))


if __name__ == "__main__":
    main()
