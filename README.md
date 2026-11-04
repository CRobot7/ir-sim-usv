# ir-sim-usv

WAM-V 无人船仿真（基于 [IR-SIM](https://github.com/hanruihua/ir-sim)）。浪扰动是 Python 重写的 ASVLite 线性 Froude-Krylov 场，挂在 `world` 上，按船位查询。

## 运行 USV demo

在仓库根目录：

```powershell
conda activate irsim
pip install -e ".[keyboard]"
cd usage/27usv_world
python usv_world.py
```

窗口出现后：

- 不按键：喷泵为 0，船仍会被浪推动（轨迹线）
- `W`：双喷泵前进
- `A` / `D`：差速转向
- `S`：无效（喷泵不能反推）

无键盘时也可以直接给推力：

```python
env.step([150.0, 150.0])  # 左、右喷泵推力 [N]
tau = env.wave_wrench(env.robot)  # 当前船体轴浪力 [X, Y, N]
```

开环脚本：`python usv_thrust.py`

五艘船同一片浪（喷泵为 0，按船位取力；背景是同一套 `WaveField.elevation` 的 2D 高程图）：

```powershell
python 2D_multi_usv_world.py
```

`python multi_usv_world.py` 同样会画这片浪。YAML 里 `plot.show_wave: false` 可关掉。

## 激光雷达可视化

改 `usage/27usv_world/usv_world.yaml` 里 `robot.plot`：

```yaml
robot:
  plot:
    show_sensor: true     # false = 关掉雷达射线；雷达仍在算
    show_trajectory: true # 历史轨迹线
    traj_width: 0.12      # 细线；不设则按船宽 (~2.2 m) 画，会糊成一条带
    show_trail: false     # true 才会画出历史船体框
    obj_fill: false       # 船体只描边
    obj_linewidth: 2.5    # 轮廓线宽（matplotlib points）
    arrow_length: 3.0     # 艏向箭头长度 [m]，颜色跟船体
```

## 浪场

海况在同一个 YAML 的 `world.wave`。默认是 **Douglas 2 级（smooth）**：`Hs = 0.20 m`，`Tp = 5 s`（谱峰波长约 39 m，远大于 4.3 m 船长）。删掉整个 `wave` 块即无浪。

```yaml
world:
  wave:
    Hs: 0.20       # 有义波高 [m]；Douglas 2 级
    Tp: 5.0
    heading: 0.0
    seed: 0
```

先前 `Hs = 0.6 m` 是 3 级轻浪（slight），约为 WAM-V 吃水的 6 倍，已经超出线性 Airy / Froude-Krylov 的小振幅假设，艏向会被差分压力打得很猛。船型参数和 3DOF / FK 公式未改，只把海况降到这种 4 m 级双体船近岸作业更常见的 2 级浪。

船型动力学和取压几何在 `irsim/vessels/wamv.yaml`：

```yaml
robot:
  kinematics: {name: usv, vessel: wamv, payload: 0.0}
```

复制 `wamv.yaml` 改参数即可加浆板等其它型号（`hydro.recipe` 可改成直接填 `hydro.coeffs`）。
