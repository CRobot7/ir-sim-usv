"""Five WAM-Vs with a 2D map of the shared WaveField elevation.

The background is ``env.wave.elevation(x, y, t)`` at the current time,
the same field used for Froude-Krylov wrenches.
"""

import irsim

env = irsim.make("2D_multi_usv_world.yaml")

for _i in range(2000):
    env.step()
    env.render()
    if env.done():
        break

env.end()
