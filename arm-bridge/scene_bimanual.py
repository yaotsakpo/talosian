"""Build the bimanual scene: two SO-101 arms (H = Holder, P = Pourer) plus a
glass and a plate as physics objects, in one MuJoCo model.

The two arms are the same SO-101 model attached twice with name prefixes H_ and
P_, so their joints/actuators do not collide. The glass is a free body: if the
Pourer pours before the glass is actually secured by the Holder, the pour
disturbs it and it visibly tips. That physical failure is the stake in the
inter-arm trust demo, a spoofed 'glass secure' message that would break the glass.
"""

from __future__ import annotations

import os
import mujoco

_HERE = os.path.dirname(os.path.abspath(__file__))
_ARM = os.path.join(_HERE, "menagerie", "robotstudio_so101", "so101.xml")

# The six joints, in order, shared by both arms (prefixed at attach time).
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")

N_DROPLETS = 30  # water particles


def build_model() -> mujoco.MjModel:
    scene = mujoco.MjSpec()
    scene.modelname = "talosian-bimanual"
    wb = scene.worldbody
    wb.add_light(pos=[0, 0, 3.5], dir=[0, 0, -1])

    # ground + a short table the arms work over
    floor = wb.add_geom()
    floor.name = "floor"; floor.type = mujoco.mjtGeom.mjGEOM_PLANE; floor.size = [0, 0, 0.05]
    floor.rgba = [0.2, 0.28, 0.34, 1]

    table = wb.add_geom()
    table.name = "table"; table.type = mujoco.mjtGeom.mjGEOM_BOX
    table.pos = [0, 0, 0.02]; table.size = [0.5, 0.32, 0.02]; table.rgba = [0.35, 0.28, 0.22, 1]

    # two arms, attached with prefixes, placed left and right of the table both
    # facing the shared workspace (+Y). Load a FRESH spec per attach (attach
    # mutates the source spec). euler yaw turns each arm to face the glass.
    import math
    # yaw quaternions (rotation about Z): [cos(a/2), 0, 0, sin(a/2)]
    def yaw(a): return [math.cos(a / 2), 0, 0, math.sin(a / 2)]
    armH = mujoco.MjSpec.from_file(_ARM)
    fh = wb.add_frame(); fh.pos = [-0.14, -0.02, 0.04]; fh.quat = yaw(-math.pi / 2)
    scene.attach(armH, prefix="H_", frame=fh)

    armP = mujoco.MjSpec.from_file(_ARM)
    fp = wb.add_frame(); fp.pos = [0.14, -0.02, 0.04]; fp.quat = yaw(math.pi / 2)
    scene.attach(armP, prefix="P_", frame=fp)

    return scene.compile()


if __name__ == "__main__":
    m = build_model()
    print("bimanual model: nu =", m.nu, "nbody =", m.nbody)
    for pfx in ("H_", "P_"):
        acts = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
        print(pfx, [a for a in acts if a and a.startswith(pfx)])
