"""Train a small VLA / behavior-cloning policy on the collected table-setup
demonstrations, then export it to ONNX (for OpenVINO conversion).

Policy: image + joint state -> joint action.
  - a compact CNN encodes the 240x320 RGB camera frame,
  - the 12-dim joint state is concatenated with the image embedding,
  - a small MLP head regresses the 12-dim joint action (control targets).

Trained with MSE on the expert's actions (behavior cloning). Deliberately small so
ONNX export + OpenVINO conversion are quick and run on Intel Core Ultra. Actions
and states are normalized (z-score) using dataset statistics saved alongside the
model, so inference can denormalize.

Run with the ML venv (torch):
  .venv-ml/bin/python train_policy.py --data dataset --epochs 30 --out policy
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


class SetupDataset(Dataset):
    """Flat (image, state) -> action pairs across all episodes."""

    def __init__(self, root: str):
        self.samples = []  # (frame_path, ep_idx, t)
        self.states, self.actions = {}, {}
        eps = sorted(glob.glob(os.path.join(root, "episode_*")))
        if not eps:
            raise RuntimeError(f"no episodes under {root}")
        for ep in eps:
            st = np.load(os.path.join(ep, "obs_state.npy"))
            ac = np.load(os.path.join(ep, "action.npy"))
            frames = sorted(glob.glob(os.path.join(ep, "frames", "*.png")))
            n = min(len(st), len(ac), len(frames))
            self.states[ep] = st[:n]; self.actions[ep] = ac[:n]
            for t in range(n):
                self.samples.append((frames[t], ep, t))
        all_state = np.concatenate([self.states[e] for e in self.states])
        all_action = np.concatenate([self.actions[e] for e in self.actions])
        self.state_mean = all_state.mean(0); self.state_std = all_state.std(0) + 1e-6
        self.action_mean = all_action.mean(0); self.action_std = all_action.std(0) + 1e-6

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        fp, ep, t = self.samples[i]
        img = _load_rgb(fp).astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # CHW
        s = (self.states[ep][t] - self.state_mean) / self.state_std
        a = (self.actions[ep][t] - self.action_mean) / self.action_std
        return (torch.from_numpy(img).float(),
                torch.from_numpy(s).float(),
                torch.from_numpy(a).float())


class Policy(nn.Module):
    """Compact image+state -> action network. Small on purpose (OpenVINO-friendly)."""

    def __init__(self, state_dim=12, action_dim=12):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2, padding=2), nn.ReLU(),   # 120x160
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),  # 60x80
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),  # 30x40
            nn.AdaptiveAvgPool2d((4, 4)),                          # 32x4x4
            nn.Flatten(),                                          # 512
        )
        self.head = nn.Sequential(
            nn.Linear(512 + state_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, img, state):
        feat = self.cnn(img)
        return self.head(torch.cat([feat, state], dim=1))


def train(args):
    torch.manual_seed(0)
    ds = SetupDataset(args.data)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=2, drop_last=True)
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device={dev}  samples={len(ds)}  batches/epoch={len(dl)}")

    net = Policy().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    lossf = nn.MSELoss()

    net.train()
    for ep in range(args.epochs):
        tot, nb = 0.0, 0
        for img, st, ac in dl:
            img, st, ac = img.to(dev), st.to(dev), ac.to(dev)
            opt.zero_grad()
            pred = net(img, st)
            loss = lossf(pred, ac)
            loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        print(f"epoch {ep+1}/{args.epochs}  loss={tot/max(nb,1):.4f}")

    os.makedirs(args.out, exist_ok=True)
    torch.save(net.state_dict(), os.path.join(args.out, "policy.pt"))
    stats = {
        "state_mean": ds.state_mean.tolist(), "state_std": ds.state_std.tolist(),
        "action_mean": ds.action_mean.tolist(), "action_std": ds.action_std.tolist(),
        "image_hw": list(_load_rgb(ds.samples[0][0]).shape[:2]),
    }
    with open(os.path.join(args.out, "stats.json"), "w") as fh:
        json.dump(stats, fh, indent=2)

    # export ONNX (fixed batch 1) for OpenVINO
    net.eval().to("cpu")
    h, w = stats["image_hw"]
    dummy_img = torch.zeros(1, 3, h, w)
    dummy_state = torch.zeros(1, 12)
    onnx_path = os.path.join(args.out, "policy.onnx")
    torch.onnx.export(
        net, (dummy_img, dummy_state), onnx_path,
        input_names=["image", "state"], output_names=["action"],
        opset_version=17, dynamic_axes=None,
    )
    print(f"saved policy.pt, stats.json, policy.onnx -> {args.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "dataset"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "policy"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    train(ap.parse_args())
