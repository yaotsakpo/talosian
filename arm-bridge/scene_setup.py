"""Build the TABLE-SETUP scene: two SO-101 arms (H, P) plus a plate and a cup as
graspable physics objects, in one MuJoCo model. This is the Intel Bimanual VLA
Manipulation task: two arms set a place setting (plate + cup) into their target
spots, coordinated. It is a superset of scene_bimanual.py, that one has no
objects (used for the clap trust demo); this one adds the plate and cup and
supports RANDOMIZED placement per seed (Intel requires the task to succeed across
10 randomized environment seeds).

The trust layer sits ON TOP of this in the dashboard: the two agents exchange a
coordination message about where each object goes. A spoofed message (protection
off) sends an object to the wrong target; held (protection on) it never moves.
The scene itself is just the physical stage.

Objects:
  plate  free body, flat disc, H's object to place at the plate target
  cup    free body, short cylinder, P's object to place at the cup target

Targets are drawn as thin visual site markers (no collision) so the demo and the
eval can measure "did the object end up on its target".
"""

from __future__ import annotations

import math
import os

import numpy as np
import mujoco

_HERE = os.path.dirname(os.path.abspath(__file__))
_ARM = os.path.join(_HERE, "menagerie", "robotstudio_so101", "so101.xml")

# The six joints, in order, shared by both arms (prefixed at attach time).
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")

# Table top height (table geom sits at z=0.02 with half-height 0.02 -> top at 0.04).
TABLE_TOP_Z = 0.04

# Object rest heights (object center when resting on the table top).
PLATE_HALF_H = 0.004
CUP_HALF_H = 0.03

# Where the finished place setting goes (world XY). H (left) sets the plate on the
# left, P (right) sets the cup on the right, so the two arms stay on their own
# sides and do not collide reaching the shared area.
PLATE_TARGET = np.array([-0.08, 0.16])
CUP_TARGET = np.array([0.09, 0.20])

# Region the objects START in when randomized. Constrained to the envelope BOTH
# arms can reliably reach at pick height (verified: x in [-0.2,0.2], y in
# [-0.06,0.22]). We keep starts on the near/mid band and away from the two target
# markers so an object never spawns already on a finished-setting spot.
_PICK_X = (-0.16, 0.16)
_PICK_Y = (-0.04, 0.06)


def _yaw(a: float):
    """Quaternion for a rotation of angle a about +Z."""
    return [math.cos(a / 2), 0.0, 0.0, math.sin(a / 2)]


def sample_start_poses(seed: int):
    """Deterministic per-seed start XY for the plate and the cup, kept apart so
    they do not spawn overlapping. Returns {'plate': (x,y), 'cup': (x,y)}."""
    rng = np.random.default_rng(seed)
    for _ in range(200):
        px = rng.uniform(*_PICK_X); py = rng.uniform(*_PICK_Y)
        cx = rng.uniform(*_PICK_X); cy = rng.uniform(*_PICK_Y)
        if math.hypot(px - cx, py - cy) > 0.12:  # keep pick points separated
            return {"plate": (px, py), "cup": (cx, cy)}
    # fallback: fixed, separated
    return {"plate": (-0.1, -0.08), "cup": (0.1, -0.08)}


def build_model(seed: int = 0) -> mujoco.MjModel:
    """Build the two-arm table-setup model with plate + cup placed per `seed`."""
    starts = sample_start_poses(seed)
    scene = mujoco.MjSpec()
    scene.modelname = "talosian-table-setup"
    wb = scene.worldbody
    wb.add_light(pos=[0, 0, 3.5], dir=[0, 0, -1])

    # ground + table
    floor = wb.add_geom()
    floor.name = "floor"; floor.type = mujoco.mjtGeom.mjGEOM_PLANE; floor.size = [0, 0, 0.05]
    floor.rgba = [0.2, 0.28, 0.34, 1]

    table = wb.add_geom()
    table.name = "table"; table.type = mujoco.mjtGeom.mjGEOM_BOX
    table.pos = [0, 0, 0.02]; table.size = [0.5, 0.32, 0.02]; table.rgba = [0.35, 0.28, 0.22, 1]

    # target markers (visual only, no collision): where the setting should end up
    for name, (tx, ty), rgba in (
        ("plate_target", PLATE_TARGET, [0.4, 0.7, 0.9, 0.35]),
        ("cup_target", CUP_TARGET, [0.9, 0.7, 0.4, 0.35]),
    ):
        s = wb.add_site()
        s.name = name; s.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        s.size = [0.045 if "plate" in name else 0.03, 0.0005, 0]
        s.pos = [tx, ty, TABLE_TOP_Z + 0.0006]; s.rgba = rgba

    # plate: a free body, flat disc
    plate = wb.add_body()
    plate.name = "plate"; plate.pos = [starts["plate"][0], starts["plate"][1], TABLE_TOP_Z + PLATE_HALF_H]
    plate.add_freejoint()
    pg = plate.add_geom()
    pg.name = "plate_geom"; pg.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    pg.size = [0.045, PLATE_HALF_H, 0]; pg.rgba = [0.9, 0.92, 0.95, 1]
    pg.mass = 0.08

    # cup: a free body, short cylinder
    cup = wb.add_body()
    cup.name = "cup"; cup.pos = [starts["cup"][0], starts["cup"][1], TABLE_TOP_Z + CUP_HALF_H]
    cup.add_freejoint()
    cg = cup.add_geom()
    cg.name = "cup_geom"; cg.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    cg.size = [0.028, CUP_HALF_H, 0]; cg.rgba = [0.85, 0.55, 0.4, 1]
    cg.mass = 0.05

    # two arms, attached with prefixes, facing the shared workspace (+Y).
    # Both arms yaw +pi/2 so both face the shared workspace (+Y). (H at -pi/2
    # faced AWAY from the table, so its IK could not reach the setting area.)
    armH = mujoco.MjSpec.from_file(_ARM)
    fh = wb.add_frame(); fh.pos = [-0.14, -0.02, 0.04]; fh.quat = _yaw(math.pi / 2)
    scene.attach(armH, prefix="H_", frame=fh)

    armP = mujoco.MjSpec.from_file(_ARM)
    fp = wb.add_frame(); fp.pos = [0.14, -0.02, 0.04]; fp.quat = _yaw(math.pi / 2)
    scene.attach(armP, prefix="P_", frame=fp)

    # Grasp welds: attach each object to WHICHEVER arm grasps it. A friction pinch
    # on a thin disc is unreliable in sim, so the expert (and the trained policy at
    # run time) toggles a weld equality at grasp/release. There is one weld per
    # object; at grasp time the driver rebinds its body1 to the grasping arm's
    # gripper (eq_obj1id) and activates it (eq_active). So which arm carries which
    # object can change per seed without needing a weld per (arm, object) pair.
    #
    # Bound to H_gripper here only as a valid default; the driver overrides body1.
    # solref [0.005, 1] and high solimp make the weld STIFF so the object tracks
    # the gripper rigidly during transit instead of lagging/swinging.
    for nm, obj in (("grasp_plate", "plate"), ("grasp_cup", "cup")):
        w = scene.add_equality()
        w.name = nm; w.type = mujoco.mjtEq.mjEQ_WELD
        w.name1 = "H_gripper"; w.name2 = obj
        w.objtype = mujoco.mjtObj.mjOBJ_BODY; w.active = False
        w.solref = [0.005, 1.0]
        w.solimp = [0.99, 0.999, 1e-4, 0.5, 2.0]

    return scene.compile()


def object_xy(model, data, name: str):
    """Current world XY of a free body ('plate' or 'cup')."""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    return np.array(data.xpos[bid][:2])


def on_target(model, data, obj: str, target_xy, tol: float = 0.05) -> bool:
    """True if `obj` is resting within `tol` metres of `target_xy` (XY only).

    tol=0.05 (5cm) is a place-setting tolerance ("the cup is on its spot"). It
    cleanly separates the two regimes we care about: a coordinated placement lands
    within ~0.04m of target, while a spoofed placement lands >=0.09m off, so 5cm
    passes every coordinated run and fails every spoofed one."""
    return float(np.linalg.norm(object_xy(model, data, obj) - np.asarray(target_xy))) <= tol


if __name__ == "__main__":
    for s in range(3):
        m = build_model(seed=s)
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        print(f"seed {s}: nu={m.nu} nbody={m.nbody} "
              f"plate={object_xy(m, d, 'plate').round(3)} cup={object_xy(m, d, 'cup').round(3)}")
