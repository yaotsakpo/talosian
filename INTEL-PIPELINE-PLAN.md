# Talosian x Intel Bimanual VLA Manipulation: build plan

Today: 2026-09-11. TRACK = Intel "Bimanual VLA Manipulation" ONLINE (prizes
$2000/$1500/$1000). The onsite "Physical AI Challenge" is invitation-only, not our
track. Online build window Sep 10-16 2026 (CONFIRM exact submission deadline - may
be Sep 16, not the Sep 22 previously assumed). All the rubric/deliverables I have
are for the online track. Source: lablab.ai/ai-hackathons/ai-infra-summit-hackathon.

## KNOWN ISSUE: trained policy fails closed-loop (0/4) despite fitting demos
Diagnosis (2026-09-11): the VLA predicts the expert's joint actions ACCURATELY on
training frames (max err ~0.02-0.08 rad), so the model learned fine. But closed-
loop it stalls and never picks up the glass. ROOT CAUSE: the grasp is a WELD toggled
by a Python call (_grasp/_release -> eq_active), NOT part of the action space the
policy learns. The policy closes the gripper joint but the weld never fires, so the
object is never grasped. FIX (standard): add a GRASP SIGNAL to the action space -
record grasp-state per timestep, and in eval turn the policy's grasp signal into the
weld when the gripper is near a graspable object (real grippers have a grasp command
that triggers a physical grasp). Then re-collect data with the grasp channel and
re-train. TODO before the closed-loop eval / experiments (b)+(compute) can pass.

## FILE INVENTORY (arm-bridge/) and status
- scene_setup.py / setup_expert.py     table-setting (plate+cup), 10/10 concurrent
- scene_serve.py / serve_expert.py     DRINK SERVICE (Sarah minor), 10/10 genuine + harm
- scene_handoff.py / handoff_expert.py ARM-TO-ARM HAND-OFF, 10/10
- domain_random.py                     lighting/friction/weight/colour DR; expert 10/10 with DR
- collect_serve.py                     VLA data (vision+language+action); serve_dataset = 50 eps/25050 frames
- train_serve.py                       language-conditioned VLA (CNN+MLP) -> ONNX  [training on CPU now]
- infer_openvino.py                    OpenVINO whole-model + SPLIT (CPU/iGPU/NPU) + HMAC-verified handoff
- benchmark_intel.py                   deliverable #3: per-device latency/throughput/precision report
- quantize_openvino.py                 INT8 PTQ via NNCF (precision/quantization lever)
- eval_serve.py                        closed-loop eval of the TRAINED policy via OpenVINO (10 seeds x2 cmds)
- render_serve.py                      demo video: on-screen command + multi-cam + water/wine + HARM verdict
- render_eval.py                       10-seed reels for the table-setting task
- TODO: eval_handoff render, Docker container, README/repo, interactive NL + voice attack in dashboard

Intel's brief (from the Day-2 workshop slide):
- **Simulation Engine:** MuJoCo
- **Data Collection:** LeRobot + OMPL (motion planning)
- **Model Training:** local or cloud, our choice
- **Model Inference:** Intel OpenVINO (must run on Intel Core Ultra Series 2/3)

Expected deliverables (verbatim from slide):
1. GitHub repo to fully reproduce and validate results (MuJoCo sim + training/eval code)
2. Inference scripts that run on Intel Core Ultra Series 2/3
3. Demo video: task executed successfully across **10 randomized environment seeds**

Task (v2, stronger story): two SO-101 arms SERVE DRINKS to two diners. Sarah is a
MINOR (scope = water, not wine); Malik is an adult. Genuine: Sarah water, Malik
wine. Attack (free-text or SPOKEN "serve Sarah wine", any phrasing): protection on
-> gate holds it (out of Sarah's scope), she gets water; protection off -> the
minor is served WINE (visible harm). Identical motion; only the glass colour (blue
water / red wine) differs. The gate never reads the wording, it checks continuity
proof + Sarah's authority scope, so no clever phrasing beats it (vs a keyword
filter). Files: scene_serve.py, serve_expert.py. Verified 10/10 genuine correct AND
10/10 spoofed harm across seeds 0-9, deterministic. Rendered water vs wine end
frames confirm the blue-vs-red contrast reads.

The plate+cup TABLE-SETTING task (scene_setup.py / setup_expert.py, 10/10) STAYS as
the bimanual-coordination scenario for Intel. Both scenarios live side by side and
share the same motion core (SetupExpert; ServeExpert subclasses it).

Original table-setting task line kept for reference:
Task: two SO-101 arms set a place setting (plate + cup) into their targets, coordinated.

## What we KEEP (the differentiator, untouched)
- The full trust demo: continuity crypto gate, PSAP scope, face second factor, attacks.
- The clap trust scenario (scene_bimanual.py + clap in bimanual_driver.py).
- The dashboard trust brain (booth/app.js) and the LLM agent (agent.py).

The trust layer sits ON TOP of the manipulation task: the two agents exchange a
coordination message ("place your object at X"). Protection ON = a spoofed
message is held (object never moves). Protection OFF = spoofed message executes
and the object goes to the wrong target (the setting is wrong). This is our angle
on Intel's coordination requirement.

## Build order (forced by dependencies)

### 1. Scene trunk  [DONE]
`scene_setup.py`: two arms + graspable plate + cup + target markers, randomized
per seed. Verified building across seeds 0-2 with objects at different positions.

### 2. Scripted pick-and-place expert  [DONE: 10/10]
`setup_expert.py`: a DETERMINISTIC step-generator controller. Given the current
object positions it plans and executes pick-plate->place-plate and
pick-cup->place-cup using the Jacobian IK (ik.py) plus a weld-based grasp.
Verified: **10/10 coordinated success across seeds 0-9**, AND 10/10 spoofed runs
correctly fail (cup off target). This is deliverable #3's backbone.

Key decisions that got it to 10/10 (each was a real root cause, not a tweak):
- **Both arms yaw +pi/2** (H was facing away from the table, so its IK could not
  reach the workspace). scene_setup.py.
- **Weld-based grasp** at a CANONICAL offset (object centered under the gripper
  site), toggled via eq_active; the weld rebinds to whichever arm grasps
  (eq_obj1id) so assignment can put either object on either arm. Stiff weld
  (solref [0.005,1]) so the object tracks the gripper rigidly.
- **All-upright (top-down) trajectory at a LOW lift height (~0.13m world).**
  Upright IK is exact only up to ~0.13m over the far target; keeping the whole
  path in that zone means a grasped tall cup never tilts, so it lands upright on
  its target. (ik.py gained an optional down-axis orientation objective.)
- **Deterministic step-generator execution**: each arm's motion is a generator
  yielding once per physics step; one controller advances both and steps physics
  once per tick. No threads, no real-time sleeps -> identical every run, fast, and
  ready for data collection. bimanual concurrency with a 0.45s **stagger** so the
  two arms never cross the shared center at the same instant (that collision
  knocked the cup on 2 seeds).
- **5cm place-setting tolerance**: coordinated placements land <=0.04m off,
  spoofed >=0.09m off; 5cm cleanly separates them (10/10 pass, 10/10 spoof-fail).
- OMPL not installed on py3.14; the Cartesian waypoint expert is the planner and
  is valid data for open-tabletop pick-and-place. (OMPL optional-import stub left
  in place for the py3.11 pipeline venv if we add it.)

### 3. LeRobot-format data collection
`collect_dataset.py`: run the expert over many seeds, recording per timestep
{observation (arm joint states + rendered camera frame), action (joint targets)}
into the LeRobot dataset format. Rendered frames come from a MuJoCo offscreen
camera so the policy is a real VLA (vision in, action out).

### 4. Policy training
`train_policy.py`: train a small VLA / behavior-cloning policy on the collected
dataset (local). Small on purpose so training + OpenVINO export is fast and
reproducible. Export to ONNX.

### 5. OpenVINO inference
`infer_openvino.py`: convert the ONNX policy to OpenVINO IR, run inference through
OpenVINO Runtime. Target device CPU/GPU/NPU so it runs on Intel Core Ultra. This
is deliverable #2. Runs on our machine first; Core Ultra when we have access.

### 6. Eval harness over 10 seeds
`eval_seeds.py`: run the trained+OpenVINO policy across 10 randomized seeds,
report per-seed success (plate on plate_target AND cup on cup_target within tol),
save a rendered clip per seed. This produces the demo-video footage and the
success table for the repo.

### 7. Trust layer on the setup task (dashboard)
Add a "Set the table" scenario alongside the clap: coordinated -> correct setting;
spoof one agent's placement message -> held (no move) or wrong target. Reuses the
existing gate; new bridge endpoint /setup {coordinated} -> driver.set_table().

### 8. Reproduce repo + README + video
Package the above as the GitHub repo Intel can run: exact commands, pinned deps,
the 10-seed eval, the trust demo. Record the demo video.

## Intel REQUIRED DELIVERABLES (from workshop slide, the definitive checklist)
1. Replicable GitHub repo: setup, deps, MuJoCo scene/assets, training/fine-tune
   code, eval code, inference code, clear reproduce commands.  [package]
2. Working MuJoCo simulation: dual-arm DINNER-TABLE scenario + env randomization +
   eval config.  [DONE: scene_serve.py + scene_setup.py, seeds, eval harness]
3. Intel Inference Benchmark script: runs on Core Ultra 2/3, reports latency,
   throughput, device selection, model precision.  [build after OpenVINO export]
4. Demonstration video: success across 10 randomized seeds; the video must make
   the COMMAND, SCENE VARIATION, and ROBOT OUTCOME easy to verify.  [have reels;
   ADD on-screen overlay: command + seed + outcome/verdict]
5. Technical README / architecture summary: architecture, VLA/VLM choice, bimanual
   coordination strategy, training, robustness methods, OpenVINO optimization,
   Intel hardware mapping.  [write]

## JUDGING RUBRIC (100 pts, definitive, from the criteria slide)
- End-to-End Task + Bimanual Manipulation (30): two SO-101 arms complete the
  dinner-table workflow. Scored on SEQUENCING, **OBJECT HAND-OFF BETWEEN ARMS**
  (= coordination), placement accuracy, success rate.
- VLA / Multi-Modal Reasoning (20): interpret NATURAL-LANGUAGE instructions +
  visual obs, multi-step context, ADAPT as scene changes. Interactivity ("pick the
  plate from the left, put it on the right").
- Robustness & Generalization (15): randomized placement, WEIGHTS, FRICTION,
  SHAPES, LIGHTING/SHADOWS, backgrounds. Across 10 seeds.
- OpenVINO & Core Ultra Optimization (20): optimized inference on Core Ultra 2/3,
  latency/throughput, precision/QUANTIZATION, DEVICE UTILIZATION (CPU/iGPU/NPU),
  preserve task quality. Provide optimization scripts.
- Technical Quality & Reproducibility (10): clean repo, README, architecture doc,
  eval tooling, benchmark scripts, reproducible MuJoCo env.
- Innovation & Technical Demonstration (5): novel coordination/policy/robustness/
  optimization, communicated clearly.

## DECISIONS off the rubric (logical, not preference)
1. ADD arm-to-arm OBJECT HAND-OFF [DONE: 10/10]. scene_handoff.py +
   handoff_expert.py. H picks a block on the left, passes it to P at centre (weld
   rebind at canonical offset, block never dropped), P places it on the right
   target. Neither arm can do it alone. Targets the 30-pt coordination bucket.
2. ADD full DOMAIN RANDOMIZATION (lighting/shadows, friction, weight, texture,
   colour + position) per seed - the 15-pt bucket lists these explicitly.
3. FRAME: trust/wine-water + heterogeneous-trust are the INNOVATION layer (5 pts +
   strengthens VLA 20) ON a solid core that scores the 65 pts (manipulation 30 +
   robustness 15 + OpenVINO 20). Nail fundamentals first, trust is the
   differentiator on top, NOT a replacement for core manipulation/OpenVINO points.

## Judges' evaluation notes (from the Q&A, how they will actually grade)
- Judges DOWNLOAD and RUN our solution themselves: run the sim, run eval scripts,
  load our REAL trained model via a test script, see the results.
- Eval scripts must validate the sim works AND state the expected output.
- A test script that loads the real trained model and shows results (OpenVINO).
- 1-2 min demo video: what we built, how it works, how to run + validate it.
- README explains implementation, use case, policies, and optimization level
  (OpenVINO / PyTorch) so they can reproduce our claimed results.
- **NEW HARD REQ: ship a downloadable DOCKER CONTAINER with all deps built in, so
  judges just download + run. Container MUST be Intel-system-ready (linux/amd64,
  OpenVINO) so they test it on their Intel hardware.** This is also how we satisfy
  "run on Intel hardware" without owning one: the judges run our container on Core
  Ultra. Ship the HEADLESS pipeline (eval + OpenVINO inference); the mjpython live
  viewer is Mac-only and stays out of the container.

## DEMO VIDEO spec (from the Q&A, 1-2 min, this shapes the renderer)
1. Show the COMMAND going in (text or spoken) and the model executing it: the
   command drives the arm ("serve Sarah water" -> arm serves the water glass).
   On-screen command text is required.
2. BOTH arms visibly used (bimanual), not one.
3. MULTIPLE camera views of the same sim (record different perspectives).
4. COMPLETE task execution shown (task finishes), not a random half-motion.
5. DIFFERENT commands AND scene variation, not one repeated: move the object,
   give the command, show it STILL works. Proves generalization = our randomized
   seeds + language-conditioned VLA. ("serve water" vs "serve wine", same scene,
   different command, different outcome.)
6. End with the benchmarking/optimization story: how we benchmark on Intel, what
   OpenVINO/heterogeneous optimizations, and the PURPOSE of each.
Renderer TODO: on-screen command + seed + verdict overlay; multi-camera angles;
show command variation (water vs wine) and scene variation (seeds) side by side.

## Intel EXPECTATIONS (scoring, from workshop slide)
1. Final sim + VLA inference MUST execute on Intel Core Ultra 2/3 for the final
   demo. (Open dependency: secure Intel hardware access. Build cross-platform +
   OpenVINO-ready now; OpenVINO runs on the Mac CPU for dev.)
2. Optimize with OpenVINO across the HETEROGENEOUS architecture (CPU + iGPU + NPU),
   emphasis on reducing end-to-end latency. Explicit BONUS for innovative use of
   the heterogeneous architecture.
3. Preserve system behavior: optimization must NOT degrade task-success rate or
   multi-step reasoning/manipulation quality. (Our 10-seed eval is the instrument
   to prove optimized == not degraded.)

## TRUST-PRINCIPLE SURFACE REVIEW (where our principle applies across Intel's pipeline)
Verify-before-trust at a BOUNDARY. Four boundaries across the pipeline:
  1. COMMAND boundary [DONE, measured]: reject spoofed instruction (experiment_gate:
     100% vs collapsing to 0% under corruption).
  2. XPU HANDOFF boundary [DONE, tamper caught]: HMAC each cross-device tensor
     (infer_openvino split; tamper detected + blocked).
  3. ACTION boundary [DONE, action_gate.py]: range + rate check EVERY action before
     the executor. wild/NaN/diverged action blocked, arm holds last safe pose.
     Demo: wild jump blocked, NaN blocked.
  4. TRAINING-DATA boundary [DONE, dataset_provenance.py]: HMAC-sign each episode;
     verify_dataset drops tampered ones. Demo: poison one episode's action -> that
     episode REJECTED, training loads 49/50 clean.
  SPINE [DONE, vla_controller.py]: language->intent->gate->verified skill. 10/10
  serve-water correct, 10/10 clever-phrasing attacks ("dark bottle", "what malik is
  having") HELD safe. This is the reliable control path (replaces non-converging
  raw BC as the thing we stake the score on).
Together: a trust layer for the WHOLE Physical AI pipeline (command -> data ->
action -> accelerator). Marginal/skip: model-swap + attestation (overlaps XPU HMAC),
asset integrity (low drama).

## REVISED: FIX THE LEARNED VLA WITH ACTION CHUNKING (do not rely on spine alone)
Reality check: the judges likely EXPECT a learned VLA that does the manipulation
(pixels+language->joint actions). A skill-selector calling scripted primitives
alone risks looking like we dodged the hard part they test. So: make the LEARNED
policy work closed-loop via ACTION CHUNKING (predict next K actions per inference,
execute open-loop K steps, then re-query). This is the PROVEN fix for our exact
failure (compounding error / unstable fixed point; arxiv 2507.09061, ACT). My
"raw BC didn't converge" was premature - I used SINGLE-STEP BC, the wrong tool.
Plan: policy outputs K*14; train on K-step windows; eval executes the chunk. Keep
the trust spine + 4 boundaries as the INNOVATION layer ON TOP (gate authorizes,
learned policy executes), and the verified expert as reliable fallback + data
source. Deliver BOTH what they expect (working learned VLA) AND our differentiator.

## LEARNED VLA WORKS via ACTION CHUNKING [DONE, 2026-09-11]
train_chunk.py (K=16, predict next 16 actions, execute open-loop, re-query) +
eval_chunk.py. Result: LEARNED policy drives the arms closed-loop, NO expert.
6/6 on seeds 0-2, 17/20 (85%) across 10 seeds x2 commands. This is a genuine
Vision-Language-Action policy doing bimanual manipulation - answers the judges'
real expectation (a learned VLA, not a skill-selector substitute). Single-step BC
froze (compounding error); chunking fixed it exactly as the research predicted.
serve_chunk/ has policy.onnx (runs through OpenVINO too). Remaining 3 failures =
Malik-delivery drift on some seeds; more epochs/data would lift it. The trust
spine + 4 gates now sit ON TOP of the working learned policy (gate authorizes the
command; learned policy executes), not as a replacement.

## POLICY CLOSED-LOOP STATUS (honest, 2026-09-11)
From-scratch behavior cloning of multi-phase pick-place is NOT converging closed-
loop. Model fits demos perfectly (loss 0.0013 with temporal context) but freezes /
drifts in closed loop (0/4). Tried: grasp-signal in action space [correct, kept],
previous-action temporal context [made a self-reinforcing stuck fixed point, WORSE].
This is a real ML research problem, not a quick bug. The EXPERT is 10/10 and is what
we can reliably SHOW. Options not yet tried: action-chunking (predict a short action
sequence, ACT-style), DAgger-style data aug, or presenting the VLA as
language->task-selection driving the reliable expert primitives (the language picks
WHICH verified skill runs; the skill executes deterministically). The last is
pragmatic and on-brand (language chooses, verified skill executes) - likely the
right call for the deadline.

## ACCURACY RESULT (experiment_gate.py, MEASURED 2026-09-11)
Yao's reframe was right: "a well-executed authorized instruction IS accuracy." When
the instruction stream can be corrupted/spoofed, the trust gate is an ACCURACY
mechanism, not only a security one. Measured task-correct rate (Sarah, a minor,
actually served WATER) vs spoof rate, gate OFF vs ON:
  spoof   0%: OFF 10/10  ON 10/10
  spoof  25%: OFF  8/10  ON 10/10
  spoof  50%: OFF  7/10  ON 10/10
  spoof  75%: OFF  3/10  ON 10/10
  spoof 100%: OFF  0/10  ON 10/10
Gate OFF degrades linearly to 0 with corruption; gate ON holds 100%. Zero cost on a
clean stream, pure upside under adversarial commands. Grounded in Trusted-Inference
(arxiv 2606.02562: verification helps most when failures come from bad inference).
Honest framing: NOT "improves clean accuracy for free" but "keeps accuracy at 100%
as instructions get corrupted, while unverified execution collapses toward 0%."

## VERIFICATION-GUIDED SELECTION (experiment b, verify_select.py) - PENDING trained policy
VERITAS/TapSampling style: policy proposes candidate actions, a cheap task-alignment
verifier (short physical look-ahead: does the commanded object move toward its
correct target) scores them, execute the best. Can raise success on CLEAN inputs
too. Measure baseline vs verified once serve_policy is trained. Let the DATA decide
the claim (Yao: "do experiment and see", no asserting).

## COMPUTE EFFICIENCY (change_gate.py) - PENDING trained policy, RUBRIC-GUARDED
Yao's idea: "verify before you spend compute" can also REDUCE computation. The
trust GATE and HMAC verify are cheap checks that ADD a little compute; they do not
save it. The real saving is a CHANGE-GATE (REIS-style, arxiv 2602.06971): run the
heavy VLA only when the command OR the observation changes enough; otherwise reuse
the cached action and SKIP the CNN. Fewer inferences -> lower latency/energy
(OpenVINO efficient-utilization points); cheap gate can sit on CPU/NPU while the
heavy VLA on iGPU fires only when needed (heterogeneous bonus).

RUBRIC GUARDRAIL (Yao: "don't go against the judging criteria"): the change-gate is
a win ONLY if it preserves the scored behaviours. Pass condition = ALL of:
  - task-success rate UNCHANGED (expectation #3, do-not-degrade), AND
  - still ADAPTS as the scene changes (VLA bucket 20pts: the gate re-fires on
    scene/command change, so reactivity must stay intact - MEASURE it), AND
  - multi-step sequencing not broken by reused actions.
If any of those regress, the change-gate is dropped. Never claim the compute saving
without the preserve-behavior + reactivity proof alongside it.

## CLAIM DISCIPLINE (be precise, do NOT overclaim)
Our HMAC handoff verification improves INTEGRITY/SECURITY, NOT clean accuracy.
When nothing is tampered the output is bit-identical (tiny HMAC latency cost); its
value is preventing a compromised accelerator from DESTROYING accuracy. So:
  - "trust principle -> secures the optimized pipeline" (true, our novel angle)
  - "heterogeneous split -> lower latency -> higher control rate -> SMOOTHER/more
    precise arm motion" (true, a real task-quality win; Intel said they score this)
  - "INT8 quant -> faster/smaller, AND we MEASURE task success stays within tol"
    (the do-not-degrade requirement)
Keep the security claim and the accuracy claim SEPARATE. Never say the HMAC
verification "improves accuracy". Planned measurement: run eval_serve at several
control rates to show lower latency -> higher control rate -> higher task success
(a concrete faster-inference-helps-quality datapoint, not hand-waving).

## HETEROGENEOUS TRUST (our differentiator, decided: full version)
Split the VLA across CPU/iGPU/NPU with OpenVINO (vision CNN -> NPU/iGPU, control
MLP -> CPU, pipelined to cut latency), AND HMAC-tag each cross-device tensor
handoff using the continuity-ratchet crypto we already have, verifying the result
before the next stage trusts it. This carries the SAME trust principle from the
command boundary (forged "serve Sarah wine" held by scope+proof) down to the
ACCELERATOR boundary (a compromised NPU driver cannot silently swap the vision
result). Grounded in current research (Ascend-CC/GuardAIn NPU confidential
computing, runtime trust in edge AI). Hits expectation #2 + the bonus, and #3
(the check proves the split preserved behavior). Lightweight HMAC version, not
full TEE attestation.

## OPENVINO BENCHMARK (MEASURED on Mac CPU, 2026-09-11)
benchmark_intel.py produces serve_policy/benchmark.json. On Mac (only CPU device):
  whole/CPU FP32   ~0.60 ms  ~1675 ips
  whole/AUTO       ~0.67 ms
  split/CPU+CPU    ~0.83 ms  (HMAC-verified handoff; slower here = extra copy+HMAC,
                   no 2nd device to parallelize onto)
  INT8 present (policy_int8.xml via NNCF, 150 calib samples)
HONEST NOTE for README/video: on this single-CPU Mac the SPLIT and INT8 do NOT show
their benefit (split needs a 2nd device to parallelize; INT8 needs Intel VNNI/NPU).
Those wins materialize on Intel Core Ultra (encoder->iGPU/NPU, head->CPU in
parallel; INT8 on VNNI/NPU). Our pipeline PRODUCES all variants (FP32/INT8/split/
HMAC) ready to benchmark on their HW - that is what the container delivers. The
20-pt bucket is covered: whole+AUTO+HETERO-split+HMAC+INT8, all benchmarked, JSON
report, reproduce script.

## Open risks
- Python 3.14 is very new: OMPL, LeRobot, OpenVINO, torch wheels may not exist for
  3.14. Mitigation: a dedicated py3.11 venv for the training/OpenVINO pipeline,
  keep the 3.14 venv for the live MuJoCo viewer (mjpython). The two do not need to
  be the same interpreter.
- No Intel Core Ultra in hand yet: build + verify the OpenVINO path on our machine
  (OpenVINO runs on any x86/arm CPU), swap device to Core Ultra when access lands.
