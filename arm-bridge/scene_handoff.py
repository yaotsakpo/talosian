"""Build the HAND-OFF scene: a single object that must be passed from one arm to
the other to reach its target. This is the explicit arm-to-arm object hand-off the
judging rubric rewards under "bimanual manipulation / coordination".

Layout: the object (a block) starts on the LEFT, in a zone the LEFT arm (H) can
pick but the RIGHT arm (P) cannot comfortably place from. Its target is on the
RIGHT, reachable by P. So the task genuinely requires a hand-off: H picks it, the
two grippers meet at the centre, the object transfers H -> P, and P places it on
the target. Neither arm can complete the task alone.

Randomized per seed (position; domain randomization added in scene_random.py).

One block, one weld per arm (rebound to whichever arm holds it at the moment):
  block           the object being handed off
targets/markers:
  handoff_target  where P finally places the block (right side)
"""

from __future__ import annotations

import math
import os

import numpy as np
import mujoco

_HERE = os.path.dirname(os.path.abspath(__file__))
_ARM = os.path.join(_HERE, "menagerie", "robotstudio_so101", "so101.xml")

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")

TABLE_TOP_Z = 0.04
BLOCK_HALF = 0.022

# The block starts on the left (H picks), target is on the right (P places).
HANDOFF_TARGET = np.array([0.12, 0.17])
# meeting point where the two grippers exchange the block (both arms reach it).
MEET_XY = np.array([0.0, 0.11])

_PICK_X = (-0.14, -0.06)   # left side, H's pick zone
_PICK_Y = (0.00, 0.06)

BLOCK_RGBA = [0.85, 0.7, 0.3, 1.0]


def _yaw(a: float):
    return [math.cos(a / 2), 0.0, 0.0, math.sin(a / 2)]


def sample_start_pose(seed: int):
    rng = np.random.default_rng(seed)
    return (rng.uniform(*_PICK_X), rng.uniform(*_PICK_Y))


def build_model(seed: int = 0) -> mujoco.MjModel:
    bx, by = sample_start_pose(seed)
    scene = mujoco.MjSpec()
    scene.modelname = "talosian-handoff"
    wb = scene.worldbody
    wb.add_light(pos=[0, 0, 3.5], dir=[0, 0, -1])

    floor = wb.add_geom()
    floor.name = "floor"; floor.type = mujoco.mjtGeom.mjGEOM_PLANE; floor.size = [0, 0, 0.05]
    floor.rgba = [0.2, 0.28, 0.34, 1]

    table = wb.add_geom()
    table.name = "table"; table.type = mujoco.mjtGeom.mjGEOM_BOX
    table.pos = [0, 0, 0.02]; table.size = [0.5, 0.32, 0.02]; table.rgba = [0.35, 0.28, 0.22, 1]

    tgt = wb.add_site()
    tgt.name = "handoff_target"; tgt.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    tgt.size = [0.045, 0.0005, 0]; tgt.pos = [HANDOFF_TARGET[0], HANDOFF_TARGET[1], TABLE_TOP_Z + 0.0006]
    tgt.rgba = [0.4, 0.8, 0.5, 0.35]

    block = wb.add_body()
    block.name = "block"; block.pos = [bx, by, TABLE_TOP_Z + BLOCK_HALF]
    block.add_freejoint()
    g = block.add_geom()
    g.name = "block_geom"; g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.size = [BLOCK_HALF, BLOCK_HALF, BLOCK_HALF]; g.rgba = BLOCK_RGBA; g.mass = 0.04

    armH = mujoco.MjSpec.from_file(_ARM)
    fh = wb.add_frame(); fh.pos = [-0.14, -0.02, 0.04]; fh.quat = _yaw(math.pi / 2)
    scene.attach(armH, prefix="H_", frame=fh)
    armP = mujoco.MjSpec.from_file(_ARM)
    fp = wb.add_frame(); fp.pos = [0.14, -0.02, 0.04]; fp.quat = _yaw(math.pi / 2)
    scene.attach(armP, prefix="P_", frame=fp)

    # ONE weld for the block, rebound to whichever arm currently holds it (the
    # hand-off flips eq_obj1id from H_gripper to P_gripper at the meeting point).
    w = scene.add_equality()
    w.name = "grasp_block"; w.type = mujoco.mjtEq.mjEQ_WELD
    w.name1 = "H_gripper"; w.name2 = "block"
    w.objtype = mujoco.mjtObj.mjOBJ_BODY; w.active = False
    w.solref = [0.005, 1.0]; w.solimp = [0.99, 0.999, 1e-4, 0.5, 2.0]

    return scene.compile()


def object_xy(model, data, name: str):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    return np.array(data.xpos[bid][:2])


def on_target(model, data, tol: float = 0.05) -> bool:
    return float(np.linalg.norm(object_xy(model, data, "block") - HANDOFF_TARGET)) <= tol


if __name__ == "__main__":
    for s in range(3):
        m = build_model(seed=s); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
        print(f"seed {s}: nu={m.nu} block={object_xy(m, d, 'block').round(3)} target={HANDOFF_TARGET}")
