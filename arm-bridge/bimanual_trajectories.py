"""Table-setting trajectories for the two arms (H = Holder, P = Pourer).

Gesture-level and legible rather than pixel-perfect grasps: the demo's point is
that the pourer pours only after a PROVEN 'glass secure' message, not a flawless
IK grasp. Values are conservative joint targets (degrees) per arm, expanded to
full poses that start and end at a safe HOME. On-site these get polished against
the real arms; the structure and the trust logic do not change.

Joint order matches the SO-101: shoulder_pan, shoulder_lift, elbow_flex,
wrist_flex, wrist_roll, gripper. The bimanual driver prefixes these with H_/P_.
"""

from __future__ import annotations

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")

# Safe resting pose, both arms return here.
HOME = {
    "shoulder_pan": 0.0, "shoulder_lift": -20.0, "elbow_flex": 40.0,
    "wrist_flex": 0.0, "wrist_roll": 0.0, "gripper": 20.0,
}


def _resolve(frames):
    """Expand partial frames into full poses carried forward from HOME, starting
    and ending at HOME. Returns [(pose, dwell_s)]."""
    cur = dict(HOME)
    out = [(dict(cur), 0.3)]
    for pos, dwell in frames:
        cur = {**cur, **pos}
        out.append((dict(cur), dwell))
    out.append((dict(HOME), 0.5))
    return out


# H reaches toward the glass and closes its gripper to "hold" it steady.
HOLD_GLASS = [
    ({"shoulder_pan": 10.0, "shoulder_lift": 25.0, "elbow_flex": 55.0, "gripper": 45.0}, 0.5),
    ({"shoulder_lift": 35.0, "elbow_flex": 65.0}, 0.5),   # reach in
    ({"gripper": 8.0}, 0.4),                               # close on the glass
    ({}, 0.6),                                             # hold steady
]

# H places the plate: reach low, open, set down, withdraw.
PLACE_PLATE = [
    ({"shoulder_pan": -15.0, "shoulder_lift": 30.0, "elbow_flex": 60.0, "gripper": 40.0}, 0.5),
    ({"shoulder_lift": 40.0}, 0.4),
    ({"gripper": 20.0}, 0.4),
]

# P pours: reach over the glass and tilt the wrist (the "pour"). If the glass is
# NOT being held, this motion nudges it and it tips, which is the whole point.
POUR = [
    ({"shoulder_pan": -10.0, "shoulder_lift": 20.0, "elbow_flex": 50.0}, 0.5),  # bring over the glass
    ({"wrist_flex": 45.0}, 0.5),                                                # tilt to pour
    ({"wrist_flex": 55.0}, 0.4),                                                # pour more
    ({"wrist_flex": 0.0}, 0.4),                                                 # upright
]

# P places a fork.
PLACE_FORK = [
    ({"shoulder_pan": 20.0, "shoulder_lift": 15.0, "elbow_flex": 30.0, "gripper": 30.0}, 0.5),
    ({"gripper": 20.0}, 0.4),
]

TRAJECTORIES = {
    "hold_glass": HOLD_GLASS,
    "place_plate": PLACE_PLATE,
    "pour": POUR,
    "place_fork": PLACE_FORK,
}

# Which arm performs which action, and its scope predicate (mirrors ARM_SCOPES).
ACTION_ARM = {"hold_glass": "H", "place_plate": "H", "pour": "P", "place_fork": "P"}

KNOWN_ACTIONS = frozenset(TRAJECTORIES.keys())


def resolved(action: str):
    return _resolve(TRAJECTORIES[action])
