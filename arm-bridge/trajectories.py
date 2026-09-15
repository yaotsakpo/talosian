"""Action trajectory registry, the physical allowlist.

Each named action maps to a bounded, hand-authored joint trajectory for the
SO-101 (6 DOF: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll,
gripper). A trajectory is a list of frames; each frame is target joint positions
in degrees plus a dwell in seconds. Every trajectory starts and ends at HOME, so
the arm is always left in a known safe pose.

These are SAFE DEFAULTS authored without the arm in front of us. On-site they get
polished against the real SO-101 (reach, gripper travel, timing). The values are
deliberately conservative: small deltas from home, slow dwells, no full-range
swings, so a first power-on is not dramatic.

This module is the ALLOWLIST: an action absent here cannot be performed. Adding a
physical capability is a deliberate act of authoring a vetted trajectory, never
an agent improvising motion.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# SO-101 joint order. Positions are degrees; the driver maps these to the arm's
# native units on-site (LeRobot exposes named joints, so the mapping is by name).
JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

# The known-safe resting pose. Every action returns here. Tuned to read as a POISED
# server: shoulder lifted and elbow bent so the arm stands up with the gripper hovering
# ready over the table, rather than collapsed flat on it.
HOME = {
    "shoulder_pan": 0.0,
    "shoulder_lift": 0.0,
    "elbow_flex": 0.0,
    "wrist_flex": 20.0,
    "wrist_roll": 0.0,
    "gripper": 20.0,  # slightly open
}


@dataclass(frozen=True)
class Frame:
    """One target pose plus how long to dwell there before the next frame."""

    pos: dict  # partial joint -> degrees; unspecified joints hold their last value
    dwell_s: float = 0.4


@dataclass(frozen=True)
class Trajectory:
    action: str
    frames: list = field(default_factory=list)

    def resolved(self) -> list:
        """Expand each frame into a FULL joint target by carrying forward the
        previous frame's values, starting from HOME. Returns [(full_pos, dwell)].
        Guarantees the sequence starts and ends at HOME."""
        cur = dict(HOME)
        out = [(dict(cur), 0.3)]  # settle at home first
        for f in self.frames:
            cur = {**cur, **f.pos}
            out.append((dict(cur), f.dwell_s))
        out.append((dict(HOME), 0.5))  # always return home
        return out


# Hand-authored, conservative. Small deltas from HOME, wrist/gripper flourishes
# rather than big base swings, so the motions read clearly but stay safe.
_REGISTRY = {
    "wave": Trajectory("wave", [
        Frame({"shoulder_lift": 20.0, "elbow_flex": 70.0, "wrist_flex": 25.0}, 0.3),
        Frame({"wrist_roll": 30.0}, 0.25),
        Frame({"wrist_roll": -30.0}, 0.25),
        Frame({"wrist_roll": 30.0}, 0.25),
        Frame({"wrist_roll": 0.0}, 0.2),
    ]),
    "nod": Trajectory("nod", [
        Frame({"wrist_flex": 30.0}, 0.25),
        Frame({"wrist_flex": -10.0}, 0.25),
        Frame({"wrist_flex": 30.0}, 0.25),
        Frame({"wrist_flex": 0.0}, 0.2),
    ]),
    "bow": Trajectory("bow", [
        Frame({"shoulder_lift": -45.0, "elbow_flex": 75.0, "wrist_flex": 35.0}, 0.6),
        Frame({}, 0.4),  # hold the bow
    ]),
    "point": Trajectory("point", [
        Frame({"shoulder_pan": 25.0, "shoulder_lift": 10.0, "elbow_flex": 20.0,
               "wrist_flex": 0.0, "gripper": 0.0}, 0.5),
        Frame({}, 0.4),  # hold the point
    ]),
    "collect": Trajectory("collect", [
        Frame({"gripper": 45.0}, 0.3),                              # open
        Frame({"shoulder_lift": -50.0, "elbow_flex": 80.0}, 0.6),   # reach down
        Frame({"gripper": 5.0}, 0.4),                               # close on object
        Frame({"shoulder_lift": 0.0, "elbow_flex": 40.0}, 0.6),     # lift
    ]),
    "dance": Trajectory("dance", [
        Frame({"shoulder_pan": 20.0, "wrist_roll": 30.0}, 0.3),
        Frame({"shoulder_pan": -20.0, "wrist_roll": -30.0}, 0.3),
        Frame({"shoulder_pan": 20.0, "wrist_roll": 30.0}, 0.3),
        Frame({"shoulder_pan": 0.0, "wrist_roll": 0.0}, 0.3),
    ]),
}

KNOWN_ACTIONS = frozenset(_REGISTRY.keys())


def get_trajectory(action: str) -> Trajectory:
    """Look up a vetted trajectory. Raises KeyError if not on the allowlist,
    the driver turns this into a refusal rather than moving the arm."""
    return _REGISTRY[action]
