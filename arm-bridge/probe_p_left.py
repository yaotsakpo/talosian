"""Find P's reachable left-of-center front seats: sweep the left-of-center angular
band at several radii, P only, report tilt+dist. We want the largest radius where
P stays upright AND within 6cm at x in roughly [-0.13,-0.05]."""
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


# sweep left-of-center points P must own (between H's left block and center)
targets = []
for x in (-0.13, -0.11, -0.09, -0.07):
    for y in (0.20, 0.22, 0.24, 0.26):
        targets.append((x, y))
for x, y in targets:
    r = serve_probe("P", x, y)
    ok = r[0] < 15 and r[1] <= 0.06
    print(f"P ({x:+.3f},{y:+.3f})  tilt={r[0]:5.1f} dist={r[1]*100:4.1f}cm  clean={ok}")
