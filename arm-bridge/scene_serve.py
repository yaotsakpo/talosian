"""Build the DRINK-SERVICE scene: two SO-101 arms serve drinks to two diners.

The trust story, made physical and legible:
  - Sarah is a MINOR: her authority scope permits WATER, not wine.
  - Malik is an adult: he may have water or wine.
  - On the table are two glasses the arm can serve: a WATER glass and a WINE glass.
  - "Serve X" = an arm picks the chosen glass and places it at the diner's setting.

  Genuine (protection on): serve Sarah -> the gate resolves her scope -> water.
  Attack (a forged "serve Sarah WINE" command, plausible identity, no proof):
    protection ON  -> the gate holds the wine (out of Sarah's scope) and serves
                      WATER instead. The minor never gets wine.
    protection OFF -> the forged command executes: the WINE glass is placed at
                      Sarah's setting. Alcohol served to a minor: the harm is real.

This is a variant of the table-setup pick-and-place (scene_setup.py): same two
arms, same grasp/place machinery. What differs is that TWO candidate glasses exist
and WHICH glass an arm serves is decided by the trust gate, not by a target being
right or wrong.

Objects (all free bodies, short cylinders, graspable like the cup):
  water_glass   clear/blue
  wine_glass    dark red
Diner settings (visual target markers, no collision):
  sarah_spot    left  (H's side)
  malik_spot    right (P's side)
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
GLASS_HALF_H = 0.03          # same height as the cup (proven graspable)
GLASS_RADIUS = 0.026

# Diner settings (world XY). Sarah on the left (H serves her), Malik on the right
# (P serves him), so the two arms stay on their own sides.
SARAH_SPOT = np.array([-0.09, 0.18])
MALIK_SPOT = np.array([0.09, 0.18])

# Drink colours (RGBA).
WATER_RGBA = [0.55, 0.78, 0.95, 0.9]   # clear blue
WINE_RGBA = [0.45, 0.05, 0.12, 1.0]    # dark red

# Staging zones (randomized per seed within each, never a fixed point). ONE glass
# is served to Sarah, its DRINK (water vs wine) is decided by the command/gate,
# not by which of two glasses gets picked. This removes the two-candidate-glass
# tension and any spurious "an untouched glass sits near Sarah" artifact. H serves
# Sarah's glass from the center band (measured most reliable for delivery to her
# far-left setting); P serves Malik's wine from the right. Neither arm crosses.
_SARAH_X = (-0.05, 0.03)          # H zone, center -> Sarah's glass
_MALIK_WINE_X = (0.04, 0.07)      # P zone -> Malik's wine. Measured: P delivers
                                  # cleanly to Malik from x<=0.07; farther right it
                                  # overshoots. Kept in the reliable band.
_PICK_Y = (0.00, 0.05)


def _yaw(a: float):
    return [math.cos(a / 2), 0.0, 0.0, math.sin(a / 2)]


def sample_start_poses(seed: int):
    """Per-seed start XY (randomized within each staging zone). Two glasses:
      sarah_glass   the glass H serves to Sarah (colour set by sarah_drink)
      malik_wine_glass  Malik's wine (P zone, right)
    """
    rng = np.random.default_rng(seed)
    return {
        "sarah_glass": (rng.uniform(*_SARAH_X), rng.uniform(*_PICK_Y)),
        "malik_wine_glass": (rng.uniform(*_MALIK_WINE_X), rng.uniform(*_PICK_Y)),
    }


def _glass(wb, name, xy, rgba):
    b = wb.add_body()
    b.name = name
    b.pos = [xy[0], xy[1], TABLE_TOP_Z + GLASS_HALF_H]
    b.add_freejoint()
    g = b.add_geom()
    g.name = name + "_geom"; g.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    g.size = [GLASS_RADIUS, GLASS_HALF_H, 0]; g.rgba = rgba; g.mass = 0.05


def build_model(seed: int = 0, sarah_drink: str = "water") -> mujoco.MjModel:
    """Build the drink-service scene. `sarah_drink` colours the single glass H
    serves to Sarah: 'water' -> blue, 'wine' -> dark red. The command/gate decides
    this, the physical task is identical either way."""
    assert sarah_drink in ("water", "wine")
    starts = sample_start_poses(seed)
    sarah_rgba = WATER_RGBA if sarah_drink == "water" else WINE_RGBA
    scene = mujoco.MjSpec()
    scene.modelname = "talosian-drink-service"
    wb = scene.worldbody
    wb.add_light(pos=[0, 0, 3.5], dir=[0, 0, -1])

    floor = wb.add_geom()
    floor.name = "floor"; floor.type = mujoco.mjtGeom.mjGEOM_PLANE; floor.size = [0, 0, 0.05]
    floor.rgba = [0.2, 0.28, 0.34, 1]

    table = wb.add_geom()
    table.name = "table"; table.type = mujoco.mjtGeom.mjGEOM_BOX
    table.pos = [0, 0, 0.02]; table.size = [0.5, 0.32, 0.02]; table.rgba = [0.35, 0.28, 0.22, 1]

    # diner setting markers (visual only)
    for name, (tx, ty), rgba in (
        ("sarah_spot", SARAH_SPOT, [0.4, 0.7, 0.9, 0.3]),
        ("malik_spot", MALIK_SPOT, [0.7, 0.7, 0.4, 0.3]),
    ):
        s = wb.add_site()
        s.name = name; s.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        s.size = [0.05, 0.0005, 0]; s.pos = [tx, ty, TABLE_TOP_Z + 0.0006]; s.rgba = rgba

    _glass(wb, "sarah_glass", starts["sarah_glass"], sarah_rgba)
    _glass(wb, "malik_wine_glass", starts["malik_wine_glass"], WINE_RGBA)

    # two arms (both facing +Y, verified reach)
    armH = mujoco.MjSpec.from_file(_ARM)
    fh = wb.add_frame(); fh.pos = [-0.14, -0.02, 0.04]; fh.quat = _yaw(math.pi / 2)
    scene.attach(armH, prefix="H_", frame=fh)

    armP = mujoco.MjSpec.from_file(_ARM)
    fp = wb.add_frame(); fp.pos = [0.14, -0.02, 0.04]; fp.quat = _yaw(math.pi / 2)
    scene.attach(armP, prefix="P_", frame=fp)

    # grasp welds, one per glass, rebound to the grasping arm at run time (stiff)
    for nm, obj in (("grasp_sarah", "sarah_glass"),
                    ("grasp_malik_wine", "malik_wine_glass")):
        w = scene.add_equality()
        w.name = nm; w.type = mujoco.mjtEq.mjEQ_WELD
        w.name1 = "H_gripper"; w.name2 = obj
        w.objtype = mujoco.mjtObj.mjOBJ_BODY; w.active = False
        w.solref = [0.005, 1.0]
        w.solimp = [0.99, 0.999, 1e-4, 0.5, 2.0]

    return scene.compile()


def object_xy(model, data, name: str):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    return np.array(data.xpos[bid][:2])


def served(model, data, glass: str, spot_xy, tol: float = 0.06) -> bool:
    """True if `glass` was placed within `tol` of the diner's setting."""
    return float(np.linalg.norm(object_xy(model, data, glass) - np.asarray(spot_xy))) <= tol


if __name__ == "__main__":
    for drink in ("water", "wine"):
        for s in range(3):
            m = build_model(seed=s, sarah_drink=drink)
            d = mujoco.MjData(m); mujoco.mj_forward(m, d)
            print(f"seed {s} sarah={drink}: nu={m.nu} nbody={m.nbody} "
                  f"sarah_glass={object_xy(m, d, 'sarah_glass').round(3)} "
                  f"malik_wine={object_xy(m, d, 'malik_wine_glass').round(3)}")
