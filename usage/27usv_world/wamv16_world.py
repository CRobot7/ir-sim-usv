"""WAM-V 16 keyboard demo (vessels/wamv16.yaml).

Larger catamaran than USV14; waterjets may reverse.

Keys (control_mode: keyboard):
  W     both waterjets (surge)
  A / D differential thrust (yaw)
  S     reverse (twj_min = -500 N)

Or pass thrust directly::

    env.step([-200.0, -200.0])  # astern [N]
"""

import irsim

env = irsim.make("wamv16_world.yaml")

for _i in range(2000):
    env.step()
    env.render()
    if env.done():
        break

env.end()
