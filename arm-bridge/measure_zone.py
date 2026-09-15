"""Measure each arm's clean-UPRIGHT delivery zone by REAL serves.

For a grid of target (x,y) on the table front, build a scene, make the given arm
pick a can from ITS OWN pile (H=wine/left, P=water/right), call _serve_can to that
target, settle, and record final tilt and distance. A target is CLEAN for that arm
if the delivered can ends tilt<15 and within 0.06.
"""
from __future__ import annotations

import math
import sys

import numpy as np
import mujoco

from scene_restaurant import build_model, TABLE_TOP_Z, CAN_HALF_H
from restaurant_driver import RestaurantDriver


def measure_one(arm, tx, ty):
    """Return (tilt_deg, dist) of a real serve by `arm` to (tx,ty)."""
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


def main():
    xs = [round(x, 3) for x in np.arange(-0.34, 0.341, 0.06)]
    ys = [round(y, 3) for y in np.arange(0.08, 0.361, 0.04)]
    arms = sys.argv[1:] or ["H", "P"]
    for arm in arms:
        print(f"\n=== ARM {arm} (pile={'wine/left' if arm=='H' else 'water/right'}) ===")
        print("clean = tilt<15 AND dist<=0.06.  cells: 'OK' or tilt value")
        header = "  y\\x  " + "".join(f"{x:>7.2f}" for x in xs)
        print(header)
        clean = []
        for y in ys:
            row = f"{y:>6.2f} "
            for x in xs:
                r = measure_one(arm, x, y)
                if r is None:
                    row += f"{'--':>7}"
                    continue
                tilt, dist = r
                ok = tilt < 15.0 and dist <= 0.06
                if ok:
                    clean.append((x, y))
                    row += f"{'OK':>7}"
                else:
                    tag = f"{tilt:.0f}"
                    if dist > 0.06:
                        tag = f"d{dist*100:.0f}"
                    row += f"{tag:>7}"
            print(row)
        print(f"CLEAN[{arm}] = {clean}")


if __name__ == "__main__":
    main()
