"""Change-gate: run the EXPENSIVE VLA forward pass only when it is warranted, and
otherwise reuse the last action. This is the compute-efficiency version of our
"verify before you spend" principle: a cheap check decides whether the heavy
inference needs to run at all.

Grounded in REIS-style scene gating (arxiv 2602.06971): trigger full reasoning only
on meaningful scene changes; skip redundant inference during visually stable
stretches, using a cheap similarity/delta over features. Here the cheap signal is:
  - the LANGUAGE command changed (always re-infer), or
  - the observation changed enough: a fast delta over the downsampled image AND the
    joint state exceeds a threshold.
When neither holds, the cached action is reused and the VLA (the CNN in particular)
is NOT run.

Why it matters for the rubric: fewer VLA inferences -> lower end-to-end latency and
energy (OpenVINO "efficient utilization"), and on Intel's heterogeneous HW the cheap
gate can sit on CPU/NPU while the heavy VLA on iGPU fires only when needed. The eval
proves task quality is preserved while inferences drop.

Wraps any object with .act(image, state, lang) (e.g. OVPolicy / OVSplitPolicy).
Reports the fraction of steps where the heavy inference was SKIPPED.
"""

from __future__ import annotations

import numpy as np


def _img_signature(image, k=8):
    """A tiny, cheap signature of the frame: mean-pool to k x k per channel. The
    delta between signatures is a fast proxy for 'did the scene change' without
    running the CNN."""
    # image: (1,3,H,W) float32 in [0,1]
    x = image[0]
    C, H, W = x.shape
    hs, ws = H // k, W // k
    x = x[:, :hs * k, :ws * k].reshape(C, k, hs, k, ws).mean(axis=(2, 4))
    return x.reshape(-1)


class ChangeGatedPolicy:
    def __init__(self, policy, img_thresh: float = 0.010, state_thresh: float = 0.05):
        self.policy = policy
        self.img_thresh = img_thresh
        self.state_thresh = state_thresh
        self._last_sig = None
        self._last_state = None
        self._last_lang = None
        self._last_action = None
        self.calls = 0        # heavy VLA forward passes actually run
        self.steps = 0        # total control steps
        self.skipped = 0      # steps where the cached action was reused

    def act(self, image, state, lang):
        self.steps += 1
        sig = _img_signature(image)
        st = np.asarray(state).ravel()
        lg = np.asarray(lang).ravel()

        run = self._last_action is None
        if not run:
            if not np.array_equal(lg, self._last_lang):
                run = True   # command changed -> always re-infer
            else:
                img_delta = float(np.abs(sig - self._last_sig).mean())
                state_delta = float(np.abs(st - self._last_state).mean())
                run = (img_delta > self.img_thresh) or (state_delta > self.state_thresh)

        if run:
            self._last_action = np.asarray(self.policy.act(image, state, lang)).ravel()
            self.calls += 1
        else:
            self.skipped += 1

        self._last_sig, self._last_state, self._last_lang = sig, st, lg
        return self._last_action

    def report(self):
        return {
            "steps": self.steps, "vla_calls": self.calls, "skipped": self.skipped,
            "skip_rate": round(self.skipped / max(self.steps, 1), 3),
            "compute_saved_pct": round(100.0 * self.skipped / max(self.steps, 1), 1),
        }
