"""Train the language-conditioned Vision-Language-Action (VLA) policy for the
drink-service task, then export it to ONNX (for OpenVINO conversion).

Inputs (VLA):
  Vision    : the camera frame (CNN encoder)
  Language  : the drink command, encoded as a 2-dim one-hot (water / wine)
  proprio   : the 12 joint positions (state)
Output:
  Action    : the 12 joint control targets.

The LANGUAGE input is what makes this a VLA and not just visuomotor BC: the SAME
scene with a DIFFERENT command must produce a different serve (water vs wine). The
command is fused with the visual + proprio features before the action head, so the
policy is conditioned on the instruction, exactly the surface a spoofed command
attacks, and exactly what the trust gate guards.

Small on purpose so ONNX export + OpenVINO conversion are fast and split cleanly
across CPU/iGPU/NPU. Normalization stats saved for inference denorm.

Run with the ML venv:
  .venv-ml/bin/python train_serve.py --data serve_dataset --epochs 30 --out serve_policy
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    import cv2
    def _load_rgb(path):
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
except Exception:
    from PIL import Image
    def _load_rgb(path):
        return np.asarray(Image.open(path).convert("RGB"))

N_DRINKS = 2  # water, wine


class ServeDataset(Dataset):
    def __init__(self, root: str, stride: int = 1):
        self.samples = []  # (frame_path, ep, t)
        self.states, self.actions, self.drinks = {}, {}, {}
        eps = sorted(glob.glob(os.path.join(root, "episode_*")))
        if not eps:
            raise RuntimeError(f"no episodes under {root}")
        for ep in eps:
            st = np.load(os.path.join(ep, "obs_state.npy"))
            ac = np.load(os.path.join(ep, "action.npy"))
            gr = np.load(os.path.join(ep, "grasp.npy"))   # (T,2) per-arm grasp bits
            dr = np.load(os.path.join(ep, "drink.npy"))
            frames = sorted(glob.glob(os.path.join(ep, "frames", "*.png")))
            n = min(len(st), len(ac), len(gr), len(dr), len(frames))
            # action = 12 joint targets + 2 grasp bits (so the policy learns to grasp)
            self.states[ep] = st[:n]
            self.actions[ep] = np.concatenate([ac[:n], gr[:n]], axis=1)
            self.drinks[ep] = dr[:n]
            for t in range(0, n, stride):   # subsample frames for faster CPU training
                self.samples.append((frames[t], ep, t))
        all_s = np.concatenate([self.states[e] for e in self.states])
        all_a = np.concatenate([self.actions[e] for e in self.actions])
        self.state_mean, self.state_std = all_s.mean(0), all_s.std(0) + 1e-6
        self.action_mean, self.action_std = all_a.mean(0), all_a.std(0) + 1e-6
        # the last 2 action dims are grasp bits (0/1): pass them through raw so
        # they stay interpretable (mean 0, std 1 => identity normalization).
        self.action_mean[12:] = 0.0
        self.action_std[12:] = 1.0

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        fp, ep, t = self.samples[i]
        img = np.transpose(_load_rgb(fp).astype(np.float32) / 255.0, (2, 0, 1))
        s = (self.states[ep][t] - self.state_mean) / self.state_std
        a = (self.actions[ep][t] - self.action_mean) / self.action_std
        # previous action (temporal context, so the policy can tell which PHASE of
        # the pick->carry->place sequence it is in and not stall on an averaged
        # action). At t=0 there is no previous action -> use the current (a no-op
        # bootstrap). Normalized like the action.
        t_prev = max(t - 1, 0)
        prev = (self.actions[ep][t_prev] - self.action_mean) / self.action_std
        drink = int(self.drinks[ep][t])
        lang = np.zeros(N_DRINKS, dtype=np.float32); lang[drink] = 1.0  # one-hot command
        return (torch.from_numpy(img).float(), torch.from_numpy(s).float(),
                torch.from_numpy(lang).float(), torch.from_numpy(prev).float(),
                torch.from_numpy(a).float())


class VLAPolicy(nn.Module):
    """Vision + Language + proprio -> action. Compact; OpenVINO/heterogeneous
    friendly (the CNN can run on NPU/iGPU, the fusion+head on CPU)."""

    def __init__(self, state_dim=12, lang_dim=N_DRINKS, action_dim=14):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),      # 512
        )
        # + action_dim for the previous-action input (temporal context)
        self.head = nn.Sequential(
            nn.Linear(512 + state_dim + lang_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, img, state, lang, prev):
        feat = self.cnn(img)
        return self.head(torch.cat([feat, state, lang, prev], dim=1))


def train(args):
    torch.manual_seed(0)
    ds = ServeDataset(args.data, stride=args.stride)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=2, drop_last=True)
    # MPS (Apple GPU) lacks non-divisible AdaptiveAvgPool2d; and the target is
    # Intel CPU/OpenVINO anyway. Use CUDA if present, else CPU (skip MPS).
    if args.device:
        dev = args.device
    elif torch.cuda.is_available():
        dev = "cuda"
    else:
        dev = "cpu"
    print(f"device={dev}  samples={len(ds)}  batches/epoch={len(dl)}")

    net = VLAPolicy().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    lossf = nn.MSELoss()
    net.train()
    for ep in range(args.epochs):
        tot, nb = 0.0, 0
        for img, st, lang, prev, ac in dl:
            img, st, lang, prev, ac = (img.to(dev), st.to(dev), lang.to(dev),
                                       prev.to(dev), ac.to(dev))
            opt.zero_grad()
            loss = lossf(net(img, st, lang, prev), ac)
            loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        print(f"epoch {ep+1}/{args.epochs}  loss={tot/max(nb,1):.4f}")

    os.makedirs(args.out, exist_ok=True)
    torch.save(net.state_dict(), os.path.join(args.out, "policy.pt"))
    h, w = _load_rgb(ds.samples[0][0]).shape[:2]
    stats = {
        "state_mean": ds.state_mean.tolist(), "state_std": ds.state_std.tolist(),
        "action_mean": ds.action_mean.tolist(), "action_std": ds.action_std.tolist(),
        "image_hw": [h, w], "n_drinks": N_DRINKS,
        "drink_id": {"water": 0, "wine": 1},
    }
    with open(os.path.join(args.out, "stats.json"), "w") as fh:
        json.dump(stats, fh, indent=2)

    net.eval().to("cpu")
    onnx_path = os.path.join(args.out, "policy.onnx")
    torch.onnx.export(
        net, (torch.zeros(1, 3, h, w), torch.zeros(1, 12), torch.zeros(1, N_DRINKS),
              torch.zeros(1, 14)),
        onnx_path, input_names=["image", "state", "lang", "prev"],
        output_names=["action"], opset_version=17,
    )
    print(f"saved policy.pt, stats.json, policy.onnx -> {args.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "serve_dataset"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "serve_policy"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default=None, help="torch device (default: cuda if present else cpu)")
    ap.add_argument("--stride", type=int, default=1, help="use every Nth frame (faster CPU training)")
    train(ap.parse_args())
