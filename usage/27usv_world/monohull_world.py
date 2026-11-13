"""Monohull keyboard demo (vessels/monohull.yaml).

Same Fossen 3DOF as the WAM-V demos; hull.type selects a single-hull outline.

Keys (control_mode: keyboard):
  W     both thrusters (surge)
  A / D differential yaw
  S     ignored (twj_min = 0)

Or pass thrust directly::

    env.step([20.0, 20.0])  # port, stbd [N]
"""

import irsim

env = irsim.make("monohull_world.yaml")

for _i in range(2000):
    env.step()
    env.render()
    if env.done():
        break

env.end()
