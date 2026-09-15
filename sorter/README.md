# Talosian for the Intel on-site challenge

**Challenge:** sort parts with anomalies using Intel edge compute. A camera sees a
part, an anomaly model judges it good or defective, an arm places it in the matching
bin.

**What is missing from that pipeline:** the model does not output "defective". It
outputs a number. Someone picks a threshold, and above it the part is rejected. At 0.51
and at 0.99 the arm behaves identically, and nothing records that one of them was a coin
flip. The verdict *is* the authority.

### The threshold problem

This is a known and unsolved tension in visual inspection, not something invented for a
demo. Any threshold dividing accept from reject creates a direct tradeoff: tighten it to
catch more defects and you scrap more good parts; loosen it and defects reach the
customer, which is [usually the more serious
error](https://www.unitxlabs.com/blog/what-is-false-acceptance-fa-and-false-rejection-fr-in-ai-inspection/).
A missed defect is not one wasted part: it is everything later built on top of it.

It is also hard to see, because the usual metrics hide it. On the standard benchmark,
[pixel-level AUROC can exceed 0.99 by predicting that nothing is
defective](https://arxiv.org/pdf/2401.01984). A headline number can look excellent while
defects pass.

The tension is unavoidable *as posed*, because the score distributions for good and
defective parts genuinely overlap. No tuning recovers information that is not there.

### What changes here

The premise is that every part must go to one of two bins, so the overlap has to be
guessed at. This adds a third outcome:

> **If the score does not clear the bar, the placement action is skipped.**

The part is not sorted into either bin. SmolVLA is never asked to place it. The arm
makes no sorting motion, and the reason is recorded.

The threshold does not disappear. It stops being the line between shipping a defect and
scrapping a good part, and becomes the line between acting and deferring, which is a far
less costly thing to get slightly wrong.

This is the same argument the restaurant demo on `main` makes with a different task:
**a claim proposes, only a proof establishes.** A classifier's output is a claim, not a
warrant to act.

## Where the gate sits

```
camera -> ACT grasps -> Anomalib scores -> [ verdict + confidence ]
                                                    |
                                            THE GATE (sort_gate.py)
                                        derives scope from evidence
                                                    |
                            allowed ----------------+---------------- held
                               |                                        |
                     SmolVLA places it                     the placement is SKIPPED:
                       in good / reject                    no sorting motion at all,
                                                           the reason is recorded
```

The gate sits between Anomalib and SmolVLA. It does not touch ACT, does not re-run
perception, and does not make any model more accurate. It decides whether a claim may
become a motion.

The gate does not re-run perception and does not second-guess the model's opinion. It
asks a different question: **has this claim earned the authority to move a part into a
customer bin?** Authority is derived at the boundary, from evidence:

- confidence below the floor
- the scene does not match training conditions (lighting, camera moved)
- the part is unlike anything in the training set

Any of those and the verdict is a claim without standing. The part is held, and the arm
makes no sorting motion, which is the same property the restaurant bridge has: a held
request queues no motion.

## The demo

One part, one model, one variable:

| | bin | arm moves | why |
|---|---|---|---|
| protection **off** | good | yes | acting on the model's claim unchecked |
| protection **on** | review | **no** | confidence 0.62 below floor; scene does not match training |

With the gate off, a defective part the detector was fooled about goes to the customer.
With it on, the same part is stopped, and the reason is legible.

## Why this is the right place for it

Intel's own Physical AI Studio documentation describes "future-ready hooks for action
clamps and emergency stops, guarding against unsafe movements caused by model errors or
unexpected inputs". The hook is acknowledged; the policy that belongs in it is not
supplied. This is that policy.


## Using it

One line changes in whatever script runs the trained policy on the arm.

**Before**, the policy's action goes straight to the motors:

```python
from physicalai.inference import InferenceModel

policy = InferenceModel("./policy")
obs, info = env.reset()
while not done:
    action = policy.select_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
```

**After**, the same loop, with the policy wrapped:

```python
from physicalai.inference import InferenceModel
from gated_policy import GatedPolicy

policy = GatedPolicy(
    InferenceModel("./policy"),
    evidence=anomaly_reading,        # returns the current part's score, see below
)

obs, info = env.reset()
while not done:
    action = policy.select_action(obs)      # None when the gate withholds
    if action is None:                      # placement skipped: leave the part alone
        continue
    obs, reward, terminated, truncated, info = env.step(action)
```

Nothing else changes. `GatedPolicy` exposes the same interface and forwards every other
attribute to the wrapped policy.

### Supplying the evidence

`evidence` is a callable returning what is known about the part currently in the
gripper:

```python
def anomaly_reading():
    result = anomalib_model.predict(current_frame)
    return {
        "verdict":     "defective" if result.pred_label else "good",
        "confidence":  float(result.pred_score),   # 0..1
        "calibrated":  True,     # is the scene the one the model was trained under
        "seen_before": True,     # does this part resemble the training set
    }
```

Only `verdict` and `confidence` are required; the other two default to True. They exist
because a confident score means nothing if the lighting changed or the part is unlike
anything the model has seen.

### Setting the bar

`CONFIDENCE_FLOOR` in `sort_gate.py` is **0.75 as a placeholder**. Set it from the
trained model's own score distribution, not by guessing:

1. Run the model over a test set containing known defective parts.
2. Keep every score, not just the labels.
3. Plot the scores for good and defective parts. The overlap is the region where a
   two-bin threshold has to guess.
4. Put the floor at the edge of that overlap.

That also produces the number worth reporting: *of the defective parts that scored above
the accept threshold, how many did the gate hold instead of passing to the customer?*

### Showing the difference

`protection_on=False` runs the pipeline as it works today, the model's claim reaching
the motors unchecked, so the contrast can be demonstrated rather than asserted:

| | placement | arm moves | why |
|---|---|---|---|
| protection **off** | good bin | yes | acting on the claim unchecked |
| protection **on** | skipped | **no** | confidence 0.62 below floor; scene does not match training |

Same part, same model, same hardware. One variable.

## How it attaches to Physical AI Studio

Studio runs a trained policy with a plain loop:

```python
from physicalai.inference import InferenceModel
policy = InferenceModel("./policy")
obs, info = env.reset()
while not done:
    action = policy.select_action(obs)          # <- the boundary
    obs, reward, terminated, truncated, info = env.step(action)
```

Whatever `select_action` returns becomes what the motors do, and nothing in between
asks whether the claim behind that action earned the authority to move a part into a
customer's bin. `GatedPolicy` wraps the policy and answers that question at the
boundary, exposing the same interface, so the loop does not change:

```python
policy = GatedPolicy(InferenceModel("./policy"), evidence=anomaly_reading)
```

That is the whole integration. No fork of Studio, no patched framework.

### The other deployment path

Intel also ships a separate deployment framework (`openvinotoolkit/physicalai`) with a
`RobotRuntime` that takes a pluggable `action_source`. `gated_source.py` is the same gate
shaped for that seam, for a project deploying through the runtime rather than Studio's
loop. The two files are the same policy wearing the interface each framework expects.

<details>
<summary>The runtime version</summary>

The OpenVINO Physical AI runtime is assembled as:

```python
runtime = RobotRuntime(
    fps=30,
    robot=SO101(port="/dev/ttyACM0"),
    action_source=PolicySource(model=InferenceModel("./exports/act_policy")),
    cameras={...},
)
```

`action_source` is a documented extension point: anything implementing the ActionSource
protocol. The runtime pulls an action from it each tick and hands it to the robot's
`send_action`. That seam, between what the policy wants and what the motors do, is
where authority should be established, so that is where the gate stands:

```python
runtime = RobotRuntime(
    fps=30,
    robot=SO101(port="/dev/ttyACM0"),
    action_source=GatedSource(
        inner=PolicySource(model=InferenceModel("./exports/act_policy")),
        evidence=anomaly_reading,          # verdict + confidence from Anomalib
    ),
    cameras={...},
)
```

A withheld action is not a modified action. The arm is simply not commanded to sort.

</details>

Intel describes the framework as having "future-ready hooks for action clamps and
emergency stops, guarding against unsafe movements caused by model errors or unexpected
inputs". The seam is real. What is not shipped is a policy to put in it: there are no
documented validation layers or interception points between inference output and motor
commands. This is that policy.

## Status

- `sort_gate.py` — scope derivation and the allow/hold decision. Done, tested.
- `gated_policy.py` — wrapper for **Physical AI Studio's** `select_action` loop. Done,
  tested against a stub policy. This is the one to use with Studio.
- `gated_source.py` — the same gate shaped for the separate OpenVINO Physical AI
  runtime's `action_source` seam.
- Not yet done, and both need the trained model to exist first:
  - `evidence` is not wired to Anomalib. It expects
    `{verdict, confidence, calibrated, seen_before}`; the real output format has to be
    read off the model.
  - `CONFIDENCE_FLOOR = 0.75` is a placeholder. It should be set from the score
    distribution of the actual trained model, not guessed.
  - Import paths are taken from the documentation and have not been run against the
    installed package.
