# Running this, start to finish

Written to be followed literally. Every command says which machine it runs on and what
you should see. Where something might differ on your setup, it says so rather than
pretending.

---

## 0. What you need before starting

- The Intel laptop, with the arms plugged in and calibrated (`hack_follower`, `hack_leader`)
- Both conda environments: `hack_lerobot`, `hack_physica_ai`
- A camera pointed at the cube area
- Blue bin (good), pink bin (bad), and an input area where cubes start

---

## 1. Get this repo onto the Intel laptop

```bash
git clone -b intel-onsite-anomaly-sort https://github.com/yaotsakpo/talosian.git
cd talosian
```

---

## 2. Check what is already installed

Anomalib is **not** part of Physical AI Studio (Studio does imitation learning only), so
it may or may not be on the machine. Find out before installing anything:

```bash
conda activate hack_physica_ai

python -c "import anomalib; print('anomalib', anomalib.__version__)"
python -c "import openvino; print('openvino', openvino.__version__)"
python -c "import physicalai; print('physicalai ok')"
```

- **`anomalib` prints a version** → skip step 3.
- **`ModuleNotFoundError: anomalib`** → do step 3.

Do not `pip install` anything that already imports. The environments were prepared for
you, and reinstalling can pull a different version and break Studio.

---

## 3. Install Anomalib (only if step 2 said it is missing)

```bash
conda activate hack_physica_ai
pip install anomalib
```

Intel's own materials name **v2.6.0**, so if you want to match them exactly:

```bash
pip install "anomalib==2.6.0"
```

Verify:

```bash
python -c "import anomalib; print(anomalib.__version__)"
```

---

## 4. Photograph the cubes

This is **separate from the teleoperation demos**. Your teammates are recording arm
movements for ACT and SmolVLA; nobody is necessarily photographing cubes for defect
detection. Someone has to.

### Find the camera

```bash
conda activate hack_lerobot
lerobot-find-cameras
```

That lists the cameras with their indices. Note the index of the one **pointed at the
cube area**, since that is the one the model must learn from.

### Take the photos

```bash
conda activate hack_physica_ai
cd talosian

# good cubes: press SPACE for each one, Q when done
python sorter/capture.py --label good --camera 0

# defective cubes
python sorter/capture.py --label defect --camera 0
```

A preview window opens with a counter. Put a cube down, press SPACE, swap it, press
SPACE again. The script writes into `cubes/good/` and `cubes/defect/`, numbering files
so you can stop and come back without overwriting anything.

Hands-free alternative, if you would rather move cubes than reach for the keyboard:

```bash
python sorter/capture.py --label good --camera 0 --auto 30 --delay 2
```

That takes 30 photos, two seconds apart. Move or swap the cube between shots.

### What actually matters

- **Consistency beats quantity.** Same camera, same angle, same distance, same lighting,
  same background, every shot. A model trained on photos that vary in lighting learns
  the lighting, not the defect, and will call a good cube anomalous because a cloud went
  past.
- **Use the same camera** that will watch the arm during the demo.
- **20 to 30 good cubes** to start with. The script tells you if you are short.
- **5 to 10 defective cubes**, with the defect **visible from the camera angle you will
  actually use**. A chip on the underside teaches the model nothing.
- Vary the cube's position and rotation within the frame, but not the camera.

## 5. Train the anomaly model

```bash
cd talosian
conda activate hack_physica_ai
python sorter/anomaly.py train --data cubes/
```

This trains and exports for OpenVINO. It takes minutes, not hours, on CPU.

**If it errors:** the script is written against Anomalib's documented API but has not
been run against your installed version, so a call may need adjusting. Paste the error;
it is likely a renamed argument rather than anything structural.

The exported model lands under `results/`. Note the path to the `.xml` file, you need it
in step 7.

---

## 6. Set the threshold from real scores

Do not guess this. `CONFIDENCE_FLOOR` in `sort_gate.py` currently says `0.75`, which is
a placeholder.

Run the trained model over cubes it has **not** seen, and save one line per cube:

```
score,truth
0.12,good
0.88,defect
0.47,good
```

Then:

```bash
python sorter/run_experiment.py scores.csv
```

You get a table like this:

```
  floor  defects shipped  good scrapped    held  held %
   0.00               16             54       0    0.0%   <- today, no gate
   0.10                8             40      53   10.6%
   0.20                4             26     107   21.4%
```

Pick a floor from that table and put it in `sort_gate.py`. **This table is also your
evidence**: it is the difference the gate makes, measured on your own cubes rather than
asserted.

---

## 7. Run the demo

```bash
conda activate hack_physica_ai

python sorter/run_gated.py \
    --export-dir ~/models/<model-id>/exports/openvino \
    --follower-port /dev/ttyACM0 \
    --device NPU
```

- `--export-dir` is where **Studio** exported the trained ACT/SmolVLA policy.
- `--follower-port` is the follower arm, from `lerobot-find-port`.
- `--device` can be `CPU`, `GPU` or `NPU`.

What you should see: the arm picks up a cube, and confident cubes go to the blue or pink
bin while uncertain ones are **put back on the table**. At the end it prints how many
decisions were made and how many were held.

### The comparison that makes the point

Run the same thing again with the gate switched off:

```bash
python sorter/run_gated.py ... --protection off
```

Now every cube goes into a bin, including the ones it is unsure about. Same arm, same
models, same cubes. One switch.

---

## What is still stubbed

`run_gated.py` currently uses a placeholder for the anomaly reading (it always returns
"good, 0.99"). Before the demo, wire it to the real model:

```python
# in run_gated.py, replace anomaly_reading_stub with:
from anomaly import AnomalyReader
reader = AnomalyReader("results/.../weights/openvino/model.xml")
# then pass: evidence=lambda: reader.read(current_frame)
```

`current_frame` is the camera image of the cube in the gripper. Where that comes from
depends on how the cameras are wired in your run, which is why it is not filled in here.

---

## Troubleshooting

**`command not found: lerobot-find-port`** — you did not activate the environment.
`conda activate hack_lerobot` first.

**`Permission denied: /dev/ttyACM0`** — `sudo chmod 666 /dev/ttyACM0 /dev/ttyACM1`

**The find-port tool says it could not detect the port** — you must press Enter while the
cable is still unplugged. Unplug, wait two seconds, then press Enter.

**Anomalib API errors in step 5** — the script follows the documented API but was not run
against your version. Paste the error.

**The arm lurches when teleoperation starts** — put both arms in a similar pose before
starting; the follower jumps to match the leader.
