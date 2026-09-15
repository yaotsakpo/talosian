"""Dataset provenance (trust boundary 4): sign each recorded demonstration episode
so training accepts only UNTAMPERED demos. This defends the TRAINING boundary
against data poisoning, a corrupted or injected episode (e.g. one that teaches the
policy to serve wine to the minor) is rejected before it can influence the model.

Our principle at the data boundary: a training episode is not trusted because it is
present in the dataset folder; it is trusted only if it carries a valid signature
over its contents. Same verify-before-trust idea as the command and handoff gates.

  sign_episode(dir)   -> writes a provenance.json with an HMAC over the episode's
                         arrays (state, action, grasp, drink).
  verify_episode(dir) -> True iff the signature matches the current contents.
  verify_dataset(dir) -> list only the episodes that pass (poisoned/edited ones are
                         dropped), so training loads a clean set.

In a full system the key comes from the agent-continuity ratchet; here a dataset
key keeps it reproducible. This is lightweight and real, not full supply-chain
attestation.
"""

from __future__ import annotations

import glob
import hashlib
import hmac
import json
import os

import numpy as np

_DATASET_KEY = b"talosian-dataset-provenance-v1"
_ARRAYS = ("obs_state.npy", "action.npy", "grasp.npy", "drink.npy")


def _episode_digest(ep_dir: str) -> str:
    h = hmac.new(_DATASET_KEY, digestmod=hashlib.sha256)
    for name in _ARRAYS:
        p = os.path.join(ep_dir, name)
        if os.path.exists(p):
            h.update(name.encode())
            h.update(np.load(p).tobytes())
    # include the command text (the language label) in the signature
    cmd = os.path.join(ep_dir, "command.txt")
    if os.path.exists(cmd):
        h.update(open(cmd, "rb").read())
    return h.hexdigest()


def sign_episode(ep_dir: str):
    sig = _episode_digest(ep_dir)
    with open(os.path.join(ep_dir, "provenance.json"), "w") as fh:
        json.dump({"hmac_sha256": sig}, fh)
    return sig


def verify_episode(ep_dir: str) -> bool:
    prov = os.path.join(ep_dir, "provenance.json")
    if not os.path.exists(prov):
        return False
    want = json.load(open(prov)).get("hmac_sha256")
    return hmac.compare_digest(want or "", _episode_digest(ep_dir))


def sign_dataset(root: str):
    eps = sorted(glob.glob(os.path.join(root, "episode_*")))
    for ep in eps:
        sign_episode(ep)
    return len(eps)


def verify_dataset(root: str, verbose=False):
    """Return the list of episode dirs that pass provenance (clean set for training)."""
    good, bad = [], []
    for ep in sorted(glob.glob(os.path.join(root, "episode_*"))):
        (good if verify_episode(ep) else bad).append(ep)
    if verbose:
        print(f"provenance: {len(good)} verified, {len(bad)} rejected")
        for b in bad:
            print(f"  REJECTED (tampered/unsigned): {os.path.basename(b)}")
    return good


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "serve_dataset"
    if len(sys.argv) > 2 and sys.argv[2] == "sign":
        n = sign_dataset(root)
        print(f"signed {n} episodes in {root}")
    elif len(sys.argv) > 2 and sys.argv[2] == "poison-test":
        # demo: sign, then poison one episode's actions, show it is rejected
        sign_dataset(root)
        eps = sorted(glob.glob(os.path.join(root, "episode_*")))
        victim = eps[0]
        a = np.load(os.path.join(victim, "action.npy"))
        a[10] += 1.0  # tamper: nudge one action (a poisoned demonstration)
        np.save(os.path.join(victim, "action.npy"), a)
        print(f"poisoned {os.path.basename(victim)} (edited one action)")
        good = verify_dataset(root, verbose=True)
        print(f"=> training would load {len(good)}/{len(eps)} episodes (poisoned one dropped)")
    else:
        good = verify_dataset(root, verbose=True)
        print(f"clean episodes: {len(good)}")
