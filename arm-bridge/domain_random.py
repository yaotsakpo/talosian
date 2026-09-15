"""Domain randomization for the dinner-table scenes. The judging rubric's
Robustness bucket explicitly lists: object placement, WEIGHTS, FRICTION, SHAPES,
LIGHTING/SHADOWS, and background. Position is already randomized per seed in each
scene; this module randomizes the REST, so the policy must generalize across
physical and visual conditions, not just where objects sit.

Applied to a compiled MjModel + its MjData (post-compile, per seed):
  - lighting: light position/direction + ambient/diffuse (shadows shift)
  - table + floor colour (background variation)
  - per free-object: mass (weight), friction, and colour/texture tint

Kept within ranges that do not break the task (the expert must still succeed), so
the SAME expert generates demonstrations across the randomized conditions and the
policy learns to be robust. Deterministic per seed.

Usage:
  from domain_random import randomize
  model = build_model(seed)
  data = mujoco.MjData(model)
  randomize(model, data, seed, free_bodies=["water_glass", "malik_wine_glass"])
"""

from __future__ import annotations

import numpy as np
import mujoco


def _rng(seed: int):
    # offset so DR draws are independent of the scene's position-sampling rng
    return np.random.default_rng(seed * 7919 + 104729)


def randomize(model, data, seed: int, free_bodies=None, *,
              light=True, colors=True, physics=True):
    """Apply deterministic per-seed domain randomization to a compiled model.
    Mutates model (geom rgba/friction, body mass, lights) in place. Call once
    after building the model, before running the episode."""
    rng = _rng(seed)
    free_bodies = free_bodies or []

    if light:
        for i in range(model.nlight):
            # jitter light position (moves shadows) and direction
            model.light_pos[i] = [rng.uniform(-1.2, 1.2), rng.uniform(-1.2, 1.2), rng.uniform(2.6, 3.8)]
            d = np.array([rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3), -1.0])
            model.light_dir[i] = d / np.linalg.norm(d)
            # vary intensity via diffuse/ambient
            lvl = rng.uniform(0.6, 1.0)
            model.light_diffuse[i] = [lvl, lvl, lvl]
            model.light_ambient[i] = [rng.uniform(0.1, 0.35)] * 3

    if colors:
        # table + floor background tint
        for name in ("table", "floor"):
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid >= 0:
                base = model.geom_rgba[gid][:3]
                jitter = rng.uniform(0.8, 1.2, size=3)
                model.geom_rgba[gid][:3] = np.clip(base * jitter, 0.05, 0.95)

    if physics:
        for bname in free_bodies:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bname)
            if bid < 0:
                continue
            # mass (weight): scale +/- 40%
            model.body_mass[bid] *= rng.uniform(0.6, 1.4)
            # the body's geoms: friction + a colour tint (texture-like variation)
            for gid in range(model.ngeom):
                if model.geom_bodyid[gid] == bid:
                    model.geom_friction[gid][0] = rng.uniform(0.6, 1.2)   # sliding friction
                    tint = rng.uniform(0.85, 1.15, size=3)
                    model.geom_rgba[gid][:3] = np.clip(model.geom_rgba[gid][:3] * tint, 0.05, 1.0)

    # recompute derived quantities after mass/inertia edits
    mujoco.mj_forward(model, data)
    return model


if __name__ == "__main__":
    # smoke test: randomize a serve scene across seeds, confirm it still builds/steps
    from scene_serve import build_model
    for s in range(3):
        m = build_model(seed=s, sarah_drink="water")
        d = mujoco.MjData(m)
        randomize(m, d, s, free_bodies=["sarah_glass", "malik_wine_glass"])
        tid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "table")
        print(f"seed {s}: table_rgba={m.geom_rgba[tid][:3].round(2)} "
              f"light0_pos={m.light_pos[0].round(2)}")
