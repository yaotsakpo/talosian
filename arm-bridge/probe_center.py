"""Fine probe of the front-center band to find where each arm's upright zone ends."""
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
    if picked is None:
        return None
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


xs = [round(x, 3) for x in np.arange(-0.12, 0.121, 0.03)]
ys = [0.24, 0.27, 0.30]
for y in ys:
    print(f"\ny={y}")
    for x in xs:
        rH = serve_probe("H", x, y)
        rP = serve_probe("P", x, y)
        def okf(r):
            return r is not None and r[0] < 15 and r[1] <= 0.06
        print(f"  x={x:+.3f}  H:t{rH[0]:.0f}/d{rH[1]*100:.0f}{'*' if okf(rH) else ' '}"
              f"  P:t{rP[0]:.0f}/d{rP[1]*100:.0f}{'*' if okf(rP) else ' '}")
