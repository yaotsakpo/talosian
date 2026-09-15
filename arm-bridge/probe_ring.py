"""Probe a concrete candidate ring with an explicit per-seat arm assignment,
each seat served by its assigned arm, and report tilt/dist and OK."""
from __future__ import annotations

import math
import sys

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


# candidate ring: (x, y, arm). H = left seats, P = center+right.
# built on an arc centre (0,0.03) radius R, span, then assign by measured zone.
def build_ring(R, a_lo, a_hi, n, hcut):
    """hcut: seats whose angle >= hcut (leftmost, high angle) go to H, rest to P."""
    cx, cy = 0.0, 0.03
    angs = np.linspace(a_hi, a_lo, n)
    ring = []
    for a in angs:
        x = cx + R * math.cos(math.radians(a))
        y = cy + R * math.sin(math.radians(a))
        arm = "H" if a >= hcut else "P"
        ring.append((round(x, 4), round(y, 4), arm))
    return ring


def main():
    R = float(sys.argv[1]) if len(sys.argv) > 1 else 0.26
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 7
    hcut = float(sys.argv[3]) if len(sys.argv) > 3 else 120.0
    a_lo, a_hi = 18, 162
    ring = build_ring(R, a_lo, a_hi, n, hcut)
    print(f"R={R} n={n} hcut={hcut}")
    allok = True
    for i, (x, y, arm) in enumerate(ring):
        r = serve_probe(arm, x, y)
        ok = r is not None and r[0] <= 25 and r[1] <= 0.06
        strict = r is not None and r[0] < 15 and r[1] <= 0.06
        allok = allok and ok
        print(f"{i}: ({x:+.3f},{y:+.3f}) {arm}  tilt={r[0]:5.1f} dist={r[1]*100:4.1f}cm "
              f"ok(<=25)={ok} clean(<15)={strict}")
    print(f"ALL OK: {allok}")


if __name__ == "__main__":
    main()
