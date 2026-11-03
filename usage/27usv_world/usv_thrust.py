"""Open-loop WAM-V: equal waterjet thrust, then a yaw pulse."""

import irsim

env = irsim.make("usv_world.yaml", display=False)
robot = env.robot

# 4 s of equal thrust (surge), then 2 s of port-only (yaw)
for i in range(120):
    env.step([160.0, 160.0])

for i in range(40):
    env.step([204.0, 0.0])

state = robot.state.flatten()
print("state [x, y, psi, u, v, r] =", state)
env.end()
