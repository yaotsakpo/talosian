"""Watch the bimanual table-setting demo live. Run under mjpython (macOS viewer).

  mjpython bimanual_viewer.py            # genuine: table set, glass safe
  mjpython bimanual_viewer.py spoof-on   # spoofed message, protection ON: held, safe
  mjpython bimanual_viewer.py spoof-off  # spoofed message, protection OFF: glass tips

The viewer runs on the main thread (macOS requirement); the scenario runs on a
background thread and drives the shared MjData the viewer renders.
"""

from __future__ import annotations

import sys
import threading
import time

import mujoco
import mujoco.viewer

from bimanual_driver import BimanualDriver


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "genuine"
    protection = arg != "spoof-off"
    spoof = arg.startswith("spoof")

    drv = BimanualDriver()
    print(f"[bimanual] scenario={arg} protection={protection} spoof={spoof}")
    print("[viewer] two SO-101s + glass. Close the window to quit.")

    def run_scenario():
        time.sleep(1.5)  # let the viewer open and the scene settle
        drv.set_the_table(protection=protection, spoof=spoof)

    threading.Thread(target=run_scenario, daemon=True).start()

    with mujoco.viewer.launch_passive(drv.model, drv.data) as viewer:
        while viewer.is_running():
            with drv._lock:
                mujoco.mj_step(drv.model, drv.data)
            viewer.sync()
            time.sleep(drv.model.opt.timestep)


if __name__ == "__main__":
    main()
