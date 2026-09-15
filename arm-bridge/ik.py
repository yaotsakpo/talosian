"""Inverse kinematics for the SO-101 arms: drive an arm's gripper to a target
world position by solving for joint angles, instead of hand-guessing angles.

Jacobian damped-least-squares IK (the standard MuJoCo approach): iteratively
nudge the arm's joints so its gripper site moves toward the target. Operates on
one arm at a time (the H_ or P_ joint set) and returns joint targets in radians,
which the driver ramps to. Because it targets a POSITION in space, the same
solution transfers to the real arm.
"""

from __future__ import annotations

import numpy as np
import mujoco


class ArmIK:
    def __init__(self, model, prefix: str, site_name: str):
        self.model = model
        self.prefix = prefix
        self.site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if self.site < 0:
            raise RuntimeError(f"IK site {site_name} not found")
        # This arm's 6 joints (position + velocity address) and actuator ids.
        self.jnt_qpos = []
        self.jnt_dof = []
        self.act = []
        for j in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}{j}")
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{prefix}{j}")
            self.jnt_qpos.append(model.jnt_qposadr[jid])
            self.jnt_dof.append(model.jnt_dofadr[jid])
            self.act.append(aid)
        # the arm joints we actually solve for (exclude the gripper from reaching)
        self.reach_dof = self.jnt_dof[:5]
        self.reach_qpos = self.jnt_qpos[:5]

    def solve(self, data, target_xyz, iters=120, tol=1e-3, damping=0.15,
              down_axis=None, down_weight=0.4):
        """Return joint angles (radians, 6 values incl. gripper unchanged) that
        put the gripper site at target_xyz. Works on a scratch copy so it does
        not disturb the live sim.

        If `down_axis` is given (a local site axis index, 0=x 1=y 2=z), the solver
        also drives THAT site axis to point down (world -Z). Constraining the
        approach orientation keeps the wrist from flopping between pick and place,
        so a grasped object does not swing sideways as the arm moves. `down_weight`
        scales the orientation error relative to position (position dominates)."""
        m = self.model
        d = mujoco.MjData(m)
        d.qpos[:] = data.qpos  # start from the current pose
        mujoco.mj_forward(m, d)
        target = np.array(target_xyz, dtype=float)
        want_down = np.array([0.0, 0.0, -1.0])
        jacp = np.zeros((3, m.nv))
        jacr = np.zeros((3, m.nv))
        for _ in range(iters):
            mujoco.mj_forward(m, d)
            perr = target - d.site_xpos[self.site]
            if down_axis is None:
                if np.linalg.norm(perr) < tol:
                    break
                mujoco.mj_jacSite(m, d, jacp, None, self.site)
                J = jacp[:, self.reach_dof]
                err = perr
            else:
                # current world direction of the chosen local site axis
                axis = d.site_xmat[self.site].reshape(3, 3)[:, down_axis]
                # orientation error as a rotation vector: axis x want (small-angle)
                oerr = np.cross(axis, want_down) * down_weight
                if np.linalg.norm(perr) < tol and np.linalg.norm(oerr) < 5e-3:
                    break
                mujoco.mj_jacSite(m, d, jacp, jacr, self.site)
                J = np.vstack([jacp[:, self.reach_dof], jacr[:, self.reach_dof]])
                err = np.concatenate([perr, oerr])
            # damped least squares: dq = J^T (J J^T + λI)^-1 err
            JJt = J @ J.T + (damping ** 2) * np.eye(J.shape[0])
            dq = J.T @ np.linalg.solve(JJt, err)
            for k, qadr in enumerate(self.reach_qpos):
                d.qpos[qadr] += dq[k]
            mujoco.mj_forward(m, d)
        # read back the solved joint angles, clamp to actuator ctrl range
        out = {}
        names = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
        for k, name in enumerate(names[:5]):
            aid = self.act[k]
            val = float(d.qpos[self.jnt_qpos[k]])
            lo, hi = m.actuator_ctrlrange[aid]
            out[name] = float(np.degrees(min(max(val, lo), hi)))
        return out
