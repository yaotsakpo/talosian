"""Probe candidate front-arc seat points with BOTH arms and report, per seat,
which arm (if any) delivers clean-upright. Lets us pick a real even ring."""
from __future__ import annotations

import math
import sys

import numpy as np
import mujoco

from scene_restaurant import build_model, SEAT_CENTER
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


def main():
    # candidate arcs: centre at SEAT_CENTER, sweep radius, span across the front.
    cx, cy = float(SEAT_CENTER[0]), float(SEAT_CENTER[1])
    radius = float(sys.argv[1]) if len(sys.argv) > 1 else 0.24
    nseats = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    a0, a1 = 15, 165
    angs = np.linspace(a1, a0, nseats)  # left to right
    print(f"ARC radius={radius} nseats={nseats} centre=({cx},{cy})")
    print("seat: (x,y)  Htilt/Hd  Ptilt/Pd  -> pick")
    for i, a in enumerate(angs):
        x = cx + radius * math.cos(math.radians(a))
        y = cy + radius * math.sin(math.radians(a))
        rH = serve_probe("H", x, y)
        rP = serve_probe("P", x, y)
        def fmt(r):
            if r is None:
                return "  --  "
            t, dd = r
            return f"t{t:.0f}/d{dd*100:.0f}"
        okH = rH is not None and rH[0] < 15 and rH[1] <= 0.06
        okP = rP is not None and rP[0] < 15 and rP[1] <= 0.06
        # prefer the clean one; if both, the one with lower tilt; if neither, mark X
        pick = "X"
        if okH and okP:
            pick = "H" if rH[0] <= rP[0] else "P"
        elif okH:
            pick = "H"
        elif okP:
            pick = "P"
        print(f"{i}: ({x:+.3f},{y:+.3f})  H:{fmt(rH)}  P:{fmt(rP)}  -> {pick}")


if __name__ == "__main__":
    main()
