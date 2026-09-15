# Talosian for the Intel on-site challenge

**Challenge:** sort parts with anomalies using Intel edge compute. A camera sees a
part, an anomaly model judges it good or defective, an arm places it in the matching
bin.

**What is missing from that pipeline:** the model's verdict *is* the authority. It says
"good at 0.51" and the part goes in the customer's box. A wrong verdict silently becomes
a wrong physical action, and nothing records that anything was ever uncertain.

That is the thing this project argues against, and it is the same argument the
restaurant demo on `main` makes with a different task: **a claim proposes, only a proof
establishes.** A classifier's output is a claim, not a warrant to act.

## Where the gate sits

```
camera -> anomaly model -> [ verdict + confidence ]
                                    |
                            THE GATE (sort_gate.py)
                        derives scope from evidence
                                    |
                    allowed ---------+--------- held
                       |                          |
                 arm sorts to bin        no sorting motion at all;
                                          part goes to review
```

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

## How it attaches to Physical AI Studio

No fork and no patched framework. The OpenVINO Physical AI runtime is assembled as:

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

Intel describes the framework as having "future-ready hooks for action clamps and
emergency stops, guarding against unsafe movements caused by model errors or unexpected
inputs". The seam is real and pluggable. What the framework does not ship is a policy to
put in it: there are no documented validation layers or interception points between
inference output and motor commands. This is that policy.

## Status

- `sort_gate.py` — scope derivation and the allow/hold decision. Done, tested.
- `gated_source.py` — the ActionSource wrapper. Done, tested against a stub policy.
- Wiring `evidence` to Anomalib's real output, and running it on the arm — next.
