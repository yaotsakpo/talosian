"""Verify a hand-tuned explicit ring: (x,y,arm) list, each served by its arm."""
from __future__ import annotations

import math

import numpy as np
import mujoco

from scene_restaurant import build_model
from restaurant_driver import RestaurantDriver


def serve_probe(arm, tx, ty):
    diners = ["probe"]
    m = build_model(diners)
    d = mujoco.MjData(m)
    drv = RestaurantDriver(m, d, diners)
    drink = "wine" if arm == "H" else "water"
    picked = drv._pick_can(arm, drink)
    body_name, k, pick_xy = picked
    spot = np.array([tx, ty])
    for _ in drv._serve_can(arm, body_name, pick_xy, spot):
        mujoco.mj_step(m, d)
    for _ in range(300):
        mujoco.mj_step(m, d)
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body_name)
    up = d.xmat[bid].reshape(3, 3)[:, 2]
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(up[2])))))
    dist = float(np.linalg.norm(d.xpos[bid][:2] - spot))
    return tilt, dist


# Hand-tuned ring, left to right. H owns far-left, P owns near-left-of-center,
# center and right. The one left-of-center seat is pulled to a P-reachable spot.
RING = [
    (-0.247, 0.110, "H"),
    (-0.193, 0.204, "H"),
    (-0.090, 0.250, "P"),   # left-of-center, inside P's reach (measured)
    ( 0.000, 0.290, "P"),   # front-center, P holds it upright (measured)
    ( 0.106, 0.268, "P"),
    ( 0.193, 0.204, "P"),
    ( 0.247, 0.110, "P"),
]

allok = True
for i, (x, y, arm) in enumerate(RING):
    r = serve_probe(arm, x, y)
    ok = r[0] <= 25 and r[1] <= 0.06
    allok = allok and ok
    print(f"{i}: ({x:+.3f},{y:+.3f}) {arm}  tilt={r[0]:5.1f} dist={r[1]*100:4.1f}cm ok={ok}")
print(f"ALL OK: {allok}")
