"""WAM-V USV keyboard demo.

Keys (control_mode: keyboard):
  W     both waterjets (surge)
  A / D differential thrust (yaw)
  S     ignored - jets cannot reverse (Klinger 2017 waterjet model)

Or pass thrust directly::

    env.step([150.0, 150.0])  # port, stbd [N]
"""

import irsim

env = irsim.make("usv_world.yaml")

for _i in range(2000):
    env.step()
    env.render()
    if env.done():
        break

env.end()
