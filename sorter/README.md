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

### What is not new here, and what is

Abstaining when unsure is **well established**. It is called the reject option or
selective prediction, and it has [a 2024 survey in *Machine
Learning*](https://link.springer.com/article/10.1007/s10994-024-06534-x). The two checks
used below are textbook categories from it: low confidence is *ambiguity rejection*, and
an unfamiliar part is *novelty rejection*. Nothing about deferring to a human is invented
here, and claiming otherwise would be wrong.

What is different is **where the decision lives and what it governs**:

|  | selective prediction | this gate |
|---|---|---|
| lives | inside the classifier | at the boundary before the actuators |
| governs | whether to answer | whether a claim becomes a motion |
| scope | the model's own score | the score, plus whether conditions match training |
| applies to | a model you trained | any model, including one you did not |

A reject option improves the quality of the answers a model gives. This governs whether
an answer is permitted to move a robot. In a pipeline of three models, where ACT grasps,
Anomalib judges and SmolVLA places, the thing that needs governing is not any one
model's confidence but the seam where a judgement becomes an act. Intel's framework has
that seam and ships no policy for it.

## Where the gate sits

```
ACT grasps the cube          <- never gated: picking it up to look at it
        |                       should happen whatever the confidence
Anomalib scores it
        |
   THE GATE  (only the placement is governed)
        |
   allowed --+-- held
      |          |
 SmolVLA      the cube goes BACK to the input area,
 places it    not into either bin
 blue / pink
```

**Only the third step is governed.** The gate does not touch ACT, does not re-run
perception, and does not make any model more accurate. It decides whether a judgement
may become a placement.

Studio holds one policy at a time and switches phase with a task string
(`StartTaskCommand` -> `policy.set_task(task)`), so the gate reads the task to tell a
grasp from a placement.

### Where a held cube goes

Anomalib decides *after* the arm already has the cube, so "hold" cannot mean "freeze":
the arm is holding something and must put it somewhere. There are two bins, blue for
good and pink for bad, and neither is right for a cube nobody is sure about. So a held
cube is **put back where it came from**.

That has a useful property. The arm works through the pile taking only what it is sure
about, and what remains in the input area is exactly the set the model could not commit
to. The gate's output is a physical pile, and its size measures where the model lacks
competence on this batch.

It measures *uncertainty*, not defectiveness: the residue mixes genuinely ambiguous
defects with good cubes that photographed badly. The honest claim is "the arm sorted
what it was sure about and left the rest", not "everything left is defective".

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

Read from Studio's own backend, not from documentation:
`application/backend/src/runtime/action_source.py`. `StudioActionSource.update()` ends:

```python
policy_action = self._policy_action(robot_state, camera_frames, step)

if self._follower_source == "teleop" and leader_action is not None:
    return leader_action
if self._follower_source == "policy" and policy_action is not None:
    return policy_action
return self._hold_target.copy()          # <- the arm holds station
```

Two things follow:

1. **The seam is `PolicySource.update(robot_state, camera_frames, step)`.** Whatever it
   returns becomes the joint command for that tick.

2. **Studio already knows how to hold.** If the policy action is `None`, the runtime
   commands `_hold_target`, the pose taken from the robot's own state, and the arm stays
   put. Studio does this itself when a policy errors (`_drop_policy_to_hold`).

So "skip the placement" needs no invention. Returning `None` already means it, and the
runtime does the safe thing. What is missing from the pipeline is anything that decides
*whether* an action has earned the right to be sent. That is the gate:

```python
from gated_source import GatedSource

source = GatedSource(
    inner=loaded_policy_source,       # what Studio loaded
    evidence=anomaly_reading,         # the current part's score, see below
)
```

`GatedSource` exposes the same `update(...)` and forwards everything else, so Studio
calls it exactly as it would the original.

## Where the gate sits

```
ACT grasps the cube          <- never gated: picking it up to look at it
        |                       should happen whatever the confidence
Anomalib scores it
        |
   THE GATE  (only the placement is governed)
        |
   allowed --+-- held
      |          |
 SmolVLA      the cube goes BACK to the input area,
 places it    not into either bin
 blue / pink
```

**Only the third step is governed.** The gate does not touch ACT, does not re-run
perception, and does not make any model more accurate. It decides whether a judgement
may become a placement.

Studio holds one policy at a time and switches phase with a task string
(`StartTaskCommand` -> `policy.set_task(task)`), so the gate reads the task to tell a
grasp from a placement.

### Where a held cube goes

Anomalib decides *after* the arm already has the cube, so "hold" cannot mean "freeze":
the arm is holding something and must put it somewhere. There are two bins, blue for
good and pink for bad, and neither is right for a cube nobody is sure about. So a held
cube is **put back where it came from**.

That has a useful property. The arm works through the pile taking only what it is sure
about, and what remains in the input area is exactly the set the model could not commit
to. The gate's output is a physical pile, and its size measures where the model lacks
competence on this batch.

It measures *uncertainty*, not defectiveness: the residue mixes genuinely ambiguous
defects with good cubes that photographed badly. The honest claim is "the arm sorted
what it was sure about and left the rest", not "everything left is defective".

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

## Does it actually help? (the experiment)

A trust gate cannot make a model more accurate, so it cannot reduce both kinds of error
for free. What it does is convert some errors into holds. Whether that trade is worth it
is a judgement about cost, and `experiment.py` produces the numbers to judge with.

It needs one thing: **(score, truth) pairs from a held-out test set**, the anomaly score
the model gave each part and whether that part was really good or defective. No robot,
no Studio, no retraining.

```bash
python -c "
import sys; sys.path.insert(0,'sorter')
from experiment import sweep
sweep(load_my_scores(), accept_at=0.5)
"
```

On a synthetic distribution with a realistic overlap (400 good, 100 defective):

```
  floor  defects shipped  good scrapped    held  held %
   0.00               16             54       0    0.0%     <- today: two bins
   0.10                8             40      53   10.6%
   0.20                4             26     107   21.4%
   0.40                0             11     228   45.6%
```

Read it as a tradeoff, not a win. A floor of 0.10 halves the defects reaching the
customer for a 10% hold rate. A floor of 0.40 stops them entirely but sends nearly half
the parts to a human. **No setting is simply best**: it depends on what a shipped defect
costs relative to a person's time, and a missed defect is
[usually the more expensive error](https://www.unitxlabs.com/blog/what-is-final-acceptance-fa-and-false-rejection-fr-in-ai-inspection/)
because everything later built on it inherits the fault.

The number worth reporting for this challenge is the same table run on **the real
model's scores**, which is why the test set has to keep scores and not just labels.

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

## How this actually runs (no Studio patch)

Studio's plugin system registers **robots only** (`application/backend/src/plugins`:
`PluginRobot`, `role: follower/leader`). There is no supported extension point for
governing actions, and the only seam is `self._policy = source` inside
`StudioActionSource._set_policy`. Patching that means running a modified Studio, which
gets lost on the next reinstall.

There is no need to. Studio **exports a trained policy to a directory on disk**, and the
code that turns that directory into a running policy is small and public
(`config_builder.py: policy_source_from_fragment`). So:

1. Use Studio, unmodified, for setup, teleoperation, recording and training.
2. Studio exports the policy to `models/<id>/exports/<backend>/`.
3. Run it with `run_gated.py`, which builds the policy exactly the way Studio does and
   puts the gate in the loop.

```bash
python sorter/run_gated.py \
    --export-dir ~/models/<model-id>/exports/openvino \
    --follower-port /dev/ttyACM0 \
    --device NPU

# the same part, the same model, ungoverned:
python sorter/run_gated.py ... --protection off
```

Nothing is forked and nothing is pulled back onto the Studio install. It also gives the
demo something Studio's UI cannot: a protection on/off switch, so the same part can be
shown reaching the customer bin and then being held.

## Running it

```bash
# the experiment, on the reference distribution
python sorter/run_experiment.py

# the experiment, on your model's real scores (score,truth per line)
python sorter/run_experiment.py scores.csv

# the tests
python -m pytest sorter/ -q
```

## Status

**Done and tested (17 tests passing):**

- `sort_gate.py` — derives what the evidence supports, and the allow/hold decision.
- `gated_source.py` — wraps Studio's `PolicySource`, matching the real
  `update(robot_state, camera_frames, step)` interface read from Studio's backend at
  `application/backend/src/runtime/action_source.py`. **This is the one to use.**
- `experiment.py` / `run_experiment.py` — measures what the gate changes.
- `gated_policy.py` — an earlier version written against a `select_action(obs)` loop
  described in a blog post. Studio does not work that way; kept only for a project that
  drives a policy directly rather than through Studio's runtime.

**Not done, and honest about why:**

- **`evidence` is not wired to Anomalib.** It expects
  `{verdict, confidence, calibrated, seen_before}`. The real output format has to be read
  off the trained model, which does not exist yet.
- **`CONFIDENCE_FLOOR = 0.75` is a placeholder.** It must come from the model's own score
  distribution, using the sweep above. Guessing it would undo the point of measuring.
- **The numbers above are from a reference distribution, not from real parts.** They show
  the harness works and what shape the answer takes. They are not a result.
- **The import path and the wrap point have not been run against the installed package.**
  Studio's own type check (`policy_loader.py:155`) runs before the source is returned, so
  wrapping the returned value should be fine, but that is reasoning from source, not a
  test on the machine.

**What is needed to turn this into a result:** a trained Anomalib model and a held-out
test set that keeps **scores, not just labels**. Without the scores there is no way to
set the floor and no experiment to run.
