"""Five WAM-Vs in one shared irregular sea.

Waterjets stay at 0 N. Each hull queries the same WaveField at the same t,
so the body-frame wrenches differ only by pose.
"""

import irsim

env = irsim.make("multi_usv_world.yaml")

print("same t, five hull wrenches [X, Y, N] N / N m")
for robot in env.robot_list:
    tau = env.wave_wrench(robot)
    print(f"  {robot.name}: {tau}")

for _i in range(2000):
    env.step()
    env.render()
    if env.done():
        break

env.end()
