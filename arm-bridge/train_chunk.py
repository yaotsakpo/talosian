"""Train the VLA with ACTION CHUNKING: from one observation predict the next K
actions (a chunk), executed open-loop before re-querying. This is the proven fix
for the compounding-error / freezing failure that single-step behavior cloning hit
here (ACT; arxiv 2507.09061): chunking shortens the effective horizon and gives
control-theoretic stability, so the policy stops stalling on an unstable fixed
point.

Inputs (VLA): Vision (image) + Language (drink one-hot) + proprio (12 joint pos).
Output: K x 14  (K future actions, each = 12 joint targets + 2 grasp bits).

At run time the first action of the chunk drives the arm; the chunk is executed for
CHUNK_EXEC steps, then the policy is re-queried. (Temporal ensembling optional; we
keep it simple: execute the chunk, re-query.)

Run with the ML venv:
  .venv-ml/bin/python train_chunk.py --data serve_dataset --epochs 20 --k 16 --out serve_chunk
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
    def _load_rgb(p): return cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
except Exception:
    from PIL import Image
    def _load_rgb(p): return np.asarray(Image.open(p).convert("RGB"))

N_DRINKS = 2
A_DIM = 14   # 12 joints + 2 grasp bits


class ChunkDataset(Dataset):
    """(image_t, state_t, lang) -> the next K actions [a_t .. a_{t+K-1}]."""

    def __init__(self, root: str, k: int, stride: int = 2):
        self.k = k
        self.samples = []       # (frame_path, ep, t)
        self.states, self.act_chunks, self.drinks = {}, {}, {}
        eps = sorted(glob.glob(os.path.join(root, "episode_*")))
        if not eps:
            raise RuntimeError(f"no episodes under {root}")
        for ep in eps:
            st = np.load(os.path.join(ep, "obs_state.npy"))
            ac = np.load(os.path.join(ep, "action.npy"))
            gr = np.load(os.path.join(ep, "grasp.npy"))
            dr = np.load(os.path.join(ep, "drink.npy"))
            frames = sorted(glob.glob(os.path.join(ep, "frames", "*.png")))
            n = min(len(st), len(ac), len(gr), len(dr), len(frames))
            actions = np.concatenate([ac[:n], gr[:n]], axis=1)  # (n,14)
            self.states[ep] = st[:n]; self.drinks[ep] = dr[:n]
            # for each t, the chunk is a[t:t+k] (last chunk repeats the final action)
            chunks = np.zeros((n, k, A_DIM), np.float32)
            for t in range(n):
                end = min(t + k, n)
                chunks[t, :end - t] = actions[t:end]
                if end - t < k:
                    chunks[t, end - t:] = actions[n - 1]  # pad by holding final action
            self.act_chunks[ep] = chunks
            for t in range(0, n, stride):
                self.samples.append((frames[t], ep, t))
        all_s = np.concatenate([self.states[e] for e in self.states])
        all_a = np.concatenate([self.act_chunks[e].reshape(-1, A_DIM) for e in self.act_chunks])
        self.state_mean, self.state_std = all_s.mean(0), all_s.std(0) + 1e-6
        self.action_mean, self.action_std = all_a.mean(0), all_a.std(0) + 1e-6
        self.action_mean[12:] = 0.0; self.action_std[12:] = 1.0   # grasp bits raw

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        fp, ep, t = self.samples[i]
        img = np.transpose(_load_rgb(fp).astype(np.float32) / 255.0, (2, 0, 1))
        s = (self.states[ep][t] - self.state_mean) / self.state_std
        chunk = (self.act_chunks[ep][t] - self.action_mean) / self.action_std  # (k,14)
        lang = np.zeros(N_DRINKS, dtype=np.float32); lang[int(self.drinks[ep][t])] = 1.0
        return (torch.from_numpy(img).float(), torch.from_numpy(s).float(),
                torch.from_numpy(lang).float(), torch.from_numpy(chunk).float())


class ChunkPolicy(nn.Module):
    def __init__(self, k, state_dim=12, lang_dim=N_DRINKS):
        super().__init__()
        self.k = k
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(512 + state_dim + lang_dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, k * A_DIM),
        )

    def forward(self, img, state, lang):
        feat = self.cnn(img)
        out = self.head(torch.cat([feat, state, lang], dim=1))
        return out.view(-1, self.k, A_DIM)


def train(args):
    torch.manual_seed(0)
    ds = ChunkDataset(args.data, args.k, stride=args.stride)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=2, drop_last=True)
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={dev}  samples={len(ds)}  k={args.k}  batches/epoch={len(dl)}")

    net = ChunkPolicy(args.k).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    lossf = nn.L1Loss()   # L1 (ACT uses L1) is robust for action regression
    net.train()
    for ep in range(args.epochs):
        tot, nb = 0.0, 0
        for img, st, lang, chunk in dl:
            img, st, lang, chunk = img.to(dev), st.to(dev), lang.to(dev), chunk.to(dev)
            opt.zero_grad()
            loss = lossf(net(img, st, lang), chunk)
            loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        print(f"epoch {ep+1}/{args.epochs}  loss={tot/max(nb,1):.4f}")

    os.makedirs(args.out, exist_ok=True)
    torch.save(net.state_dict(), os.path.join(args.out, "policy.pt"))
    h, w = _load_rgb(ds.samples[0][0]).shape[:2]
    stats = {"state_mean": ds.state_mean.tolist(), "state_std": ds.state_std.tolist(),
             "action_mean": ds.action_mean.tolist(), "action_std": ds.action_std.tolist(),
             "image_hw": [h, w], "n_drinks": N_DRINKS, "k": args.k,
             "drink_id": {"water": 0, "wine": 1}}
    json.dump(stats, open(os.path.join(args.out, "stats.json"), "w"), indent=2)

    net.eval().to("cpu")
    torch.onnx.export(
        net, (torch.zeros(1, 3, h, w), torch.zeros(1, 12), torch.zeros(1, N_DRINKS)),
        os.path.join(args.out, "policy.onnx"),
        input_names=["image", "state", "lang"], output_names=["chunk"], opset_version=17)
    print(f"saved policy.pt, stats.json, policy.onnx -> {args.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "serve_dataset"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "serve_chunk"))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--device", default=None)
    train(ap.parse_args())
