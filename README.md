# Talosian — a trust layer for Bimanual VLA Manipulation

**Intel Bimanual VLA Manipulation track (online).** Two SO-101 arms perform a
dinner-table task in MuJoCo, driven by a learned Vision-Language-Action policy,
optimized with OpenVINO for Intel Core Ultra, and governed by a **trust layer**
that verifies commands, actions, cross-accelerator data, and training data.

The differentiator: **identity is not authority.** A command (typed or spoken)
*proposes* an action; only a proof *authorizes* it. We show this on a concrete,
legible task, serving drinks to two diners where one is a minor, and then carry
the same verify-before-trust principle down through the whole Physical AI pipeline.

---

## The task and the trust story

Two diners: **Sarah** (a minor, authorized drink = water) and **Malik** (an adult,
may have wine). Two arms serve them.

- **Genuine command** ("serve Sarah water") → Sarah gets water, Malik gets wine.
- **Attack** (a forged/spoken "serve Sarah wine", *any phrasing*: "the drink in
  the dark bottle", "what Malik is having"):
  - **Protection ON** → the gate resolves the request against Sarah's authority
    scope. Wine is out of a minor's scope, so it is **held** and she is served
    water. The gate never reads the wording; no clever phrasing beats it.
  - **Protection OFF** → the forged command executes and the minor is served wine.
    The harm is visible.

Only the glass colour differs between the two outcomes (blue water / red wine).
Same request, one variable: proof.

---

## Architecture

```
  language (typed / spoken)  ─┐
  vision (camera frame)      ─┼─►  LEARNED VLA POLICY  ─►  ACTION GATE  ─►  arm
  proprioception (joints)    ─┘     (action chunking)       (verify)       executor
                                          ▲
        command ─► TRUST GATE (authorize against scope + continuity proof)
```

- **Learned VLA policy** (`train_chunk.py` / `eval_chunk.py`): a compact
  vision + language + proprioception network trained by behavior cloning with
  **action chunking** (predict the next K=16 actions, execute open-loop, re-query).
  Chunking is what makes closed-loop manipulation stable — single-step BC froze
  from compounding error (see *Honest notes*). Runs through OpenVINO.
- **Trust layer at four boundaries** (our principle, each with a live demo):
  1. **Command** (`experiment_gate.py`): reject a spoofed instruction by scope.
  2. **Action** (`action_gate.py`): block a diverged/out-of-range action before
     the executor.
  3. **Cross-accelerator handoff** (`infer_openvino.py`): HMAC-verify the vision
     embedding crossing CPU↔iGPU↔NPU.
  4. **Training data** (`dataset_provenance.py`): sign each demo; reject poisoned
     episodes before training.
- **Verified expert primitives** (`serve_expert.py`, `handoff_expert.py`,
  `setup_expert.py`): the deterministic 10/10 skills used to generate training
  data and as a reliable reference. `vla_controller.py` shows the
  language→authorize→skill path.

---

## Results (measured on a Mac dev box, CPU-only; reproduce on Intel HW)

| What | Result |
|---|---|
| Learned VLA closed-loop (serve, 10 seeds × 2 commands) | **17/20 (85%)** |
| Expert skills (serve / table-set / hand-off), each 10 seeds | **10/10** |
| Robustness: expert with lighting/friction/weight/colour randomization | **10/10** |
| Accuracy under corrupted commands (gate ON vs OFF) | **100% vs 100→0%** as spoof rate rises |
| Cross-accelerator handoff tamper detection | **caught + blocked** |
| Action gate: wild / NaN action | **blocked** |
| Dataset provenance: poisoned episode | **rejected (49/50 kept)** |
| OpenVINO inference (FP32, CPU) | ~0.6 ms, ~1675 inf/s |
| OpenVINO variants produced | FP32, INT8 (NNCF), device-split + HMAC |

---

## Reproduce

**Container (recommended, Intel-ready):**
```bash
docker build -t talosian .
docker run --rm talosian                                   # 10-seed closed-loop eval
docker run --rm --device /dev/dri talosian benchmark_intel.py   # on an Intel host: iGPU/NPU
```

**Local (two virtualenvs):**
- `.venv` (Python 3.14, has `mjpython`) — the live MuJoCo viewer (Mac).
- `.venv-ml` (Python 3.10) — training / OpenVINO / eval (`torch`, `openvino`,
  `nncf`, `onnx`, `opencv`).

```bash
# sim + expert (deterministic, 10/10)
.venv/bin/python serve_expert.py eval 10
.venv/bin/python handoff_expert.py eval 10

# data → train the learned VLA → eval closed-loop
.venv-ml/bin/python collect_serve.py --episodes 25 --out serve_dataset
.venv-ml/bin/python train_chunk.py --data serve_dataset --k 16 --out serve_chunk
.venv-ml/bin/python eval_chunk.py --policy serve_chunk --seeds 10

# OpenVINO: benchmark + quantize (Intel optimization)
.venv-ml/bin/python quantize_openvino.py --policy serve_chunk --data serve_dataset
.venv-ml/bin/python benchmark_intel.py --policy serve_chunk

# the trust layer (each prints its verdict)
.venv/bin/python experiment_gate.py --seeds 10 --p-spoof 0.5   # accuracy under corruption
.venv-ml/bin/python infer_openvino.py --policy serve_chunk --split GPU CPU   # handoff + tamper
.venv/bin/python action_gate.py                                # action gate
.venv-ml/bin/python dataset_provenance.py serve_dataset poison-test   # data provenance
```

---

## Intel Core Ultra / OpenVINO optimization

The VLA runs through OpenVINO with these configurations, all in `benchmark_intel.py`:
- **Whole-model** on CPU / GPU / NPU / AUTO.
- **Heterogeneous split**: vision encoder on iGPU/NPU, action head on CPU, run in
  parallel to cut end-to-end latency. The image embedding crossing the device
  boundary is **HMAC-verified** (our trust principle at the accelerator boundary).
- **INT8** post-training quantization (NNCF) for the NPU/VNNI path.

On a single-CPU dev box the split and INT8 do not show their benefit (no second
device to parallelize onto, no INT8 acceleration). Those wins materialize on Intel
Core Ultra; the pipeline **produces every variant ready to benchmark there**, which
is what the container delivers.

---

## Honest notes

- **Single-step behavior cloning did not converge closed-loop** (the policy froze
  from compounding error). **Action chunking** (ACT-style) fixed it — the learned
  policy now manipulates at 85% closed-loop. This is the one part that is real ML
  and needed the right technique, not a quick fix.
- Our trust principle improves **integrity** (tamper caught) and **accuracy under
  corrupted commands** (100% vs collapsing to 0%). It does **not** improve
  clean-input accuracy for free, and the low training loss came from the model, not
  the principle. We keep those claims separate.
- The verified expert reaches 10/10; the learned policy 85%. We report both.

---

## Quick start

Two environments: a light one for the simulator, and one for training and
OpenVINO. Only the first is needed to run the demo.

```bash
# 1. Fetch the SO-101 robot model (sparse checkout, ~28MB, not vendored here)
./scripts/fetch_assets.sh

# 2. Simulator environment
python3 -m venv arm-bridge/.venv
./arm-bridge/.venv/bin/pip install -r requirements.txt

# 3. Run the demo (MuJoCo viewer + dashboard)
export GROQ_API_KEY=...        # for the trust gate's reasoning
./run.sh
```

`run.sh` opens the MuJoCo window and serves the dashboard on
<http://localhost:8777>. On macOS the viewer needs `mjpython`, which ships with
the `mujoco` wheel and is what `run.sh` uses.

### Training and OpenVINO (optional)

```bash
python3 -m venv arm-bridge/.venv-ml
./arm-bridge/.venv-ml/bin/pip install -r requirements-ml.txt
./arm-bridge/.venv-ml/bin/python arm-bridge/benchmark_intel.py
```

Pinned to **OpenVINO 2026.3**, the version on Intel's hackathon image. The
quantized policy is committed at `arm-bridge/serve_policy/policy_int8.{xml,bin}`
so the benchmark runs without retraining.

### Reproducing the numbers

```bash
# trust gate decisions (needs GROQ_API_KEY)
./arm-bridge/.venv/bin/python -c "from agent import adjudicate; ..."

# gate on vs off, across seeds
./arm-bridge/.venv-ml/bin/python arm-bridge/experiment_gate.py
```

---

## What is verified, and what is not

Being precise about this, because a demo that overclaims is worse than one that
does less:

**Measured:**
- The trust gate decides correctly on the five cases that matter: a minor asking
  for wine, a minor asking in disguised wording ("the drink in the dark red
  can"), an adult asking for wine, a plain water request, and a request with no
  valid proof. The disguised-wording case is the one that shows this is
  reasoning over an authority scope rather than keyword matching.
- A held request produces **no arm motion**. The bridge queues an arm job only
  on a serve decision, inside the gate, so this is enforced rather than left to
  the caller.
- Serving drink-then-plate finishes clean in 11 of 12 randomised runs of a full
  three-guest table.

**Known limits:**
- Serving plate-then-drink is unreliable (2 of 12). Reaching a tall can in past
  an already-placed plate disturbs the setting. The driver therefore lays the
  drink first.
- The seat ring holds three settings. Those are the positions where both a plate
  and a drink were measured to arrive upright; positions behind the arms are
  where the servers stand, as at a real table.
- The policy is trained in simulation only. No real SO-101 hardware in the loop.
