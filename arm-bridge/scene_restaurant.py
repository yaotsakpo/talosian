"""Restaurant scene: two SO-101 arms (the Server) sit in the MIDDLE of a round table
and serve a menu to diners seated around it. DYNAMIC: the diner list is passed in, and
new guests are added at runtime via rebind (see restaurant_bridge).

PILE model: there is ONE central STACK of plate bodies and ONE central STACK of cup
bodies near the arms. A serve = the serving arm picks the TOP item off the matching
pile, carries it, and places it UPRIGHT at the guest's seat on the ring. The placed
item STAYS at the seat, so the pile visibly shrinks by one. There are NO parked
off-table bodies: at rest the only carriable bodies are the two piles.

Menu items and how they LOOK / their category:
  serve_water   blue cup        (drink)
  serve_wine    dark-red cup    (drink)
  serve_shrimp  pink food plate (food)
  serve_cake    tan food plate  (food)

A drink is served in a cup (recoloured to the drink at serve time), food on a plate,
no shape morphing.
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
CUP_HALF_H = 0.03
# Narrow enough for the SO-101 pincer to clamp its wall and hold by friction on a
# top-down grasp (measured: the fingertip pads close to a ~0.5cm gap, so a cup this
# slim gets pinched, a wide one just slips). Still reads as a cup.
CUP_RADIUS = 0.016

ITEM_RGBA = {
    "serve_water": [0.55, 0.78, 0.95, 0.9],
    "serve_wine":  [0.45, 0.05, 0.12, 1.0],
    "serve_shrimp":[0.95, 0.55, 0.55, 1.0],
    "serve_cake":  [0.85, 0.72, 0.45, 1.0],
}
ITEM_KIND = {"serve_water": "drink", "serve_wine": "drink",
             "serve_shrimp": "food", "serve_cake": "food"}
ITEMS = list(ITEM_RGBA)

# neutral start colours for the source piles (before a serve gives an item its colour).
CUP_NEUTRAL = [0.85, 0.87, 0.92, 1.0]
PLATE_NEUTRAL = [0.92, 0.92, 0.95, 1.0]

# default seed diners; the bridge can pass any list.
DEFAULT_DINERS = ["sarah", "malik"]


def _yaw(a):
    return [math.cos(a / 2), 0.0, 0.0, math.sin(a / 2)]


# The two arms are the servers, standing in the middle of the table. The table is SET
# the normal way: identical place settings spaced EVENLY around the table, at one radius,
# like any set table. Nothing lopsided, no gaps, every setting the same.
#
# The radius is measured, not guessed: a sweep of an even 8-setting ring found that at
# 0.20 from the table centre, seven of the eight positions are reachable by one of the
# two arms (only the position directly BEHIND the arms is not, which is simply where the
# server stands, exactly as at a real table). Larger radii lose most of the ring.
ARM_MOUNT = {"H": np.array([-0.14, -0.02]), "P": np.array([0.14, -0.02])}
TABLE_CENTER = np.array([0.0, 0.06])
SETTING_RADIUS = 0.20          # every place setting sits this far from the table centre
N_SETTINGS = 8                 # positions on the even ring
# The servers stand at the BACK of the table, so the back positions are not guest seats.
# Measured: a place set at the back (the -0.08 y row) cannot be served at all (the plate
# lands 18 to 26cm away and flips over), because it is behind the arms. Those positions
# are the servers' own side of the table, exactly as at a real table.
_SERVER_SIDE_DEG = {225, 270, 315}

# Which arm actually serves each ring position, MEASURED by running real serves (a plate
# and a can) to each position with each arm and keeping the one that lands both correctly.
# It is not simply "the arm on that side": the left side position is served correctly by
# the RIGHT arm reaching across (7cm, upright), while the left arm drops the can (23cm, on
# its side). A position absent from this table is one neither arm can serve, so it is not
# laid as a guest seat.
_SERVING_ARM = {
    (0.00, 0.26): "P",
    (0.14, 0.20): "P",
    (-0.14, 0.20): "H",
}


def _seat_ring():
    """Place settings evenly spaced around the table, one radius, skipping only the spot
    behind the arms where the servers stand. Returns (x, y, arm) per setting, going around
    the table. Guests fill from the front and around, so the table always looks set."""
    seats = []
    for i in range(N_SETTINGS):
        ang = 90 - 360.0 * i / N_SETTINGS          # start at the front, go around
        if int(ang) % 360 in _SERVER_SIDE_DEG:     # the servers' side of the table
            continue
        x = float(TABLE_CENTER[0] + SETTING_RADIUS * math.cos(math.radians(ang)))
        y = float(TABLE_CENTER[1] + SETTING_RADIUS * math.sin(math.radians(ang)))
        arm = _SERVING_ARM.get((round(x, 2), round(y, 2)))
        if arm is None:                            # not servable by either arm, not a seat
            continue
        seats.append((x, y, arm))
    # The settings stay evenly spaced around the table (above). What changes here is only
    # the ORDER guests take them: alternate P side, H side, P side, so both servers work
    # and neither one's stack runs dry while the other sits idle.
    right = [s for s in seats if s[2] == "P"]
    left = [s for s in seats if s[2] == "H"]
    order = []
    for i in range(max(len(right), len(left))):
        if i < len(right): order.append(right[i])
        if i < len(left): order.append(left[i])
    return order


SEAT_RING = _seat_ring()
MAX_SEATS = len(SEAT_RING)


def diner_spots(diners):
    """Map each diner onto a fixed seat on the ring (first come, first seat), so a
    guest's place never moves when a later guest joins. Returns {name: np.array([x,y])}."""
    return {nm: np.array(SEAT_RING[i][:2]) for i, nm in enumerate(list(diners)[:MAX_SEATS])}


def _seat_arm(i):
    return SEAT_RING[i][2]


def arm_for_spot(xy):
    """Which arm serves this seat: match against the ring (each seat carries its arm);
    fall back to the nearer mount if an arbitrary xy is passed."""
    for sx, sy, arm in SEAT_RING:
        if abs(sx - float(xy[0])) < 1e-6 and abs(sy - float(xy[1])) < 1e-6:
            return arm
    dH = np.linalg.norm(np.array(xy) - ARM_MOUNT["H"])
    dP = np.linalg.norm(np.array(xy) - ARM_MOUNT["P"])
    return "H" if dH <= dP else "P"


# SOURCE PILES, clearly shown on the table: canned WINE and canned WATER, standing
# upright in rows. Each can is a small cylinder in its drink colour, so a viewer sees at
# a glance which row is wine and which is water.
#
# EACH ARM GETS ITS OWN PAIR OF ROWS (a wine row and a water row on its own side). That
# is measured, not decorative: an arm picking from a pile on its OWN side delivers within
# 2.5 to 5cm of the seat, but reaching ACROSS to the far pile lands the can 15 to 20cm
# off (still upright, but nowhere near the guest). Since either arm must be able to serve
# either drink (the whole point of the demo is the right drink to the right person), both
# drinks have to be within each arm's own reach.
# THE ARMS ARE SPECIALISED, and the supplies sit with their specialist: the plate
# stack lives beside arm H (left), the drink cans beside arm P (right). Each arm
# serves EVERY seat with its own item type, which is what makes this bimanual: one
# arm plates the table, the other pours. Measured over the three settings: H places
# plates within 2.1 to 3.0cm (flat), P places cans within 7.4cm (nothing tipped).
CAN_RADIUS = 0.018
CAN_HALF_H = 0.033
CAN_RGBA = {"wine": [0.50, 0.05, 0.10, 1.0], "water": [0.30, 0.55, 0.90, 1.0]}
# Row centre per (arm, drink). The place settings ring the table at SETTING_RADIUS, so the
# source rows live INSIDE that ring, beside the arms in the middle, where the servers can
# reach them without crossing the settings. H works the left, P the right.
# Rows are separated in x (wine inboard, water outboard) so the two rows of one arm never
# overlap. They used to share an x with only a small y offset, which made the end cans of
# the two rows spawn in the same spot, inside each other, shoving one another over.
# Measured pick zone: an arm picks cleanly from y about 0.02 to 0.06 at x 0.035 to 0.075
# (1.8 to 3.0cm delivery). At y = -0.02 the pick fails completely (the can ends 24 to 70cm
# away, on its side), and x beyond 0.10 also fails. So BOTH drink rows sit inside that
# good band: wine at the front of it, water just behind, spread in X (not Y, which would
# push the end cans out of the band).
# Both drink rows sit at the SAME x, separated in y. That is not cosmetic: the arm
# approaches along x, so two rows side by side in x make it sweep across one row to
# reach the other and knock a can over (measured: side-by-side tips a can on a single
# serve, stacked in y tips nothing).
# The piles are SHARED: the cans sit on the left of the table, the plates on the right,
# and EITHER arm can fetch from EITHER pile (both are within measured reach of both
# mounts). That is what lets the driver pick, per request, whichever arm can do the job
# without its path running through something already on the table.
# Both piles sit well clear of the arm mounts (16cm+, measured) so a moving arm does not
# shove the stacks around, while staying within reach of BOTH arms so either can fetch
# from either pile. Cans on the left, plates on the right, as a table is actually laid.
# Both piles sit clear of the arm mounts (measured) so a moving arm does not shove the
# stacks around, while staying within reach of BOTH arms so either can fetch from either
# pile. Cans on the left, plates on the right, as a table is actually laid. The two drink
# rows share an x and are separated in y by more than the row's own span, so the rows
# never overlap and the arm never sweeps across one to reach the other.
# Both piles sit clear of the arm mounts (14.5cm, up from 9.4cm) so a moving arm no
# longer shoves the stacks around, while staying within reach of BOTH arms so either can
# fetch from either pile. Cans on the left, plates on the right, as a table is laid.
# One can of each drink: a second can per row cannot be placed without either crowding a
# mount again or falling outside the other arm's reach (searched, no solution exists).
PILE = {
    "water": (-0.045, 0.090),
    "wine":  (-0.045, 0.150),
}
# Two cans per drink, spaced 0.04 apart in y, keeps every can on one of the measured good
# pick spots (y = 0.00, 0.04, 0.08, 0.12 work; 0.06 and 0.12+ do not). Three per row would
# push the last can out of the band, where the pick fails.
PILE_COUNT = 1              # two of each drink per arm: enough for the seats it serves

# The PLATE pile: a stack of plates in the middle, the source the servers take a plate
# from when setting a place. Plates stack flat on each other, which is how plates pile.
PLATE_RADIUS = 0.038
PLATE_HALF_H = 0.005
PLATE_RGBA = [0.93, 0.93, 0.95, 1.0]
# Clear of every placemat (they used to sit ON two of the settings) and clear of the
# can rows. Measured: plates land within 5.7cm of the setting from here, flat.
PLATE_PILE = (0.060, 0.130)
PLATE_COUNT = 4             # plates per stack: one per guest that arm serves
# centre-to-centre spacing of adjacent cans. The gripper jaws (pre-rolled) sweep wider
# than a can, so cans sit apart in a SINGLE ROW and are picked outer-first, so the can
# being grabbed never has an un-picked neighbour in the jaw's swing (measured: a tight
# grid knocked adjacent cans; a spaced single row picked from the end does not).
_CAN_PITCH = 0.040


def can_slot_xy(drink, k):
    """Table (x, y) of the k-th can in this arm's row of that drink: a single row spread
    in X, kept inside the measured good pick band, so every can in the row is pickable."""
    cx, cy = PILE[drink]
    # spread along y inside the arm's measured good pick band (each can stays pickable),
    # with a tight pitch so a three-can row does not run out of the band.
    return cx, cy + k * _CAN_PITCH


def can_z(k):
    """World z of the CENTRE of an upright can standing on the table (every can rests on
    the table, they are clustered side by side, not stacked)."""
    return TABLE_TOP_Z + CAN_HALF_H


def _weld_seed_body(arm):
    """A body name that exists, just so each arm's weld compiles. The driver retargets
    the weld onto whatever it actually picks, so this is only a placeholder."""
    return "wine_0"


def make_spec(diners=None):
    """Build and return the MjSpec (kept by the bridge so it can recompile when a
    diner joins). diners: list of names."""
    diners = diners or list(DEFAULT_DINERS)
    spec = mujoco.MjSpec()
    spec.modelname = "talosian-restaurant"
    wb = spec.worldbody
    wb.add_light(pos=[0, 0, 3.5], dir=[0, 0, -1])
    wb.add_light(pos=[0.0, -0.6, 1.2], dir=[0, 0.4, -1], diffuse=[0.4, 0.4, 0.4])

    # A fixed scene camera framing the table from the front, slightly above, looking
    # toward the seated guests (+y). Placed so the arms (behind, at y=-0.02), the source
    # piles (y~0.10), the row of place settings (y~0.15..0.22) and the served items are
    # all in frame. xyaxes: camera right = world +x, camera up tilts back so the lens
    # looks forward and down onto the table.
    cam = wb.add_camera()
    cam.name = "scene"
    # (pos/quat computed from a free cam az=270 el=-30 dist=0.85 lookat=[0,0.10,0.10].)
    cam.pos = [0.0, 0.836, 0.525]
    cam.quat = [0.0, 0.0, 0.5, 0.86603]

    floor = wb.add_geom()
    floor.name = "floor"; floor.type = mujoco.mjtGeom.mjGEOM_PLANE; floor.size = [0, 0, 0.05]
    floor.rgba = [0.2, 0.28, 0.34, 1]
    # ROUND table: the arms sit near its back, guests ring the front. A cylinder geom
    # gives the round top; radius covers the seat ring plus a margin.
    table = wb.add_geom()
    table.name = "table"; table.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    # Table sized to the setting ring: the place settings sit at SETTING_RADIUS from the
    # centre, so the top ends just outside them, giving a normal edge margin instead of a
    # wide band of bare wood.
    table.pos = [float(TABLE_CENTER[0]), float(TABLE_CENTER[1]), 0.02]
    table.size = [SETTING_RADIUS + 0.07, 0.02, 0]; table.rgba = [0.36, 0.27, 0.20, 1]

    spots = diner_spots(diners)
    # A placemat at EVERY seat on the ring, always visible, so the table always looks
    # evenly set (a real table lays all its settings out, seated or not). A seat taken by
    # a current guest is lit bright; an empty seat is a faint placemat (still there,
    # evenly spaced). A late guest just brightens their seat's mat (site_rgba, no
    # recompile). Seats are laid out evenly around the ring, so the row always reads even.
    for i, (sx, sy, _arm) in enumerate(SEAT_RING):
        seated = i < len(diners)
        s = wb.add_site()
        s.name = f"slot_{i}"; s.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        s.size = [0.042, 0.0006, 0]; s.pos = [sx, sy, TABLE_TOP_Z + 0.0006]
        s.rgba = [0.90, 0.86, 0.76, 0.95] if seated else [0.72, 0.70, 0.66, 0.45]

    # SOURCE ROWS, clearly shown: a row of canned WINE and a row of canned WATER for EACH
    # arm, on that arm's own side (so either arm can serve either drink without reaching
    # across, which measurably ruins delivery accuracy). Every can is an upright freejoint
    # cylinder in its drink colour. A serve takes the outermost still-standing can off the
    # matching row and carries it to a seat, where it stays, so the row visibly shrinks.
    # Body names: {drink}_{arm}_{k}. These are the ONLY carriable bodies.
    for drink in PILE:
            for k in range(PILE_COUNT):
                sx, sy = can_slot_xy(drink, k)
                nm = f"{drink}_{k}"
                b = wb.add_body(); b.name = nm
                b.pos = [sx, sy, can_z(k)]; b.add_freejoint()
                g = b.add_geom(); g.name = f"{nm}_geom"; g.type = mujoco.mjtGeom.mjGEOM_CYLINDER
                g.size = [CAN_RADIUS, CAN_HALF_H, 0]; g.rgba = list(CAN_RGBA[drink]); g.mass = 0.05

    # THE PLATE PILE: a stack of plates beside each arm, the source a plate is taken from
    # when the server sets a place. Plates rest flat on each other, which is how a stack of
    # plates actually sits. The server takes the TOP plate off the stack.
    cx, cy = PLATE_PILE
    if True:
        for k in range(PLATE_COUNT):
            nm = f"plate_{k}"
            b = wb.add_body(); b.name = nm
            b.pos = [cx, cy, TABLE_TOP_Z + PLATE_HALF_H + k * (2 * PLATE_HALF_H + 0.0005)]
            b.add_freejoint()
            g = b.add_geom(); g.name = f"{nm}_geom"; g.type = mujoco.mjtGeom.mjGEOM_CYLINDER
            g.size = [PLATE_RADIUS, PLATE_HALF_H, 0]; g.rgba = list(PLATE_RGBA); g.mass = 0.04

    armH = mujoco.MjSpec.from_file(_ARM)
    fh = wb.add_frame(); fh.pos = [-0.14, -0.02, 0.04]; fh.quat = _yaw(math.pi / 2)
    spec.attach(armH, prefix="H_", frame=fh)
    armP = mujoco.MjSpec.from_file(_ARM)
    fp = wb.add_frame(); fp.pos = [0.14, -0.02, 0.04]; fp.quat = _yaw(math.pi / 2)
    spec.attach(armP, prefix="P_", frame=fp)

    # ONE weld per arm (grasp_H, grasp_P), initially inactive. At pick time the driver
    # RETARGETS the arm's weld onto whichever can it grabbed (eq_obj1id = the arm gripper,
    # eq_obj2id = the picked can, eq_data = the live relative pose) and activates it.
    # name2 is seeded to wine_0 only so the weld compiles with a valid second body; the
    # driver overwrites both ids at pick time.
    for arm in ("H", "P"):
        w = spec.add_equality()
        w.name = f"grasp_{arm}"; w.type = mujoco.mjtEq.mjEQ_WELD
        w.name1 = f"{arm}_gripper"; w.name2 = _weld_seed_body(arm)
        w.objtype = mujoco.mjtObj.mjOBJ_BODY; w.active = False
        w.solref = [0.005, 1.0]; w.solimp = [0.99, 0.999, 1e-4, 0.5, 2.0]

    return spec, spots


def build_model(diners=None):
    spec, spots = make_spec(diners)
    return spec.compile()


def object_xy(model, data, name):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    return np.array(data.xpos[bid][:2])


if __name__ == "__main__":
    for ds in (["sarah", "malik"], ["sarah", "malik", "yao"], ["a", "b", "c", "d"]):
        m = build_model(ds); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
        print(f"diners={ds}: nu={m.nu} nbody={m.nbody} spots={ {k: v.round(2).tolist() for k,v in diner_spots(ds).items()} }")
