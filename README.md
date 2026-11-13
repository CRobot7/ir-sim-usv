# ir-sim-usv

WAM-V 无人船仿真（基于 [IR-SIM](https://github.com/hanruihua/ir-sim)）。浪扰动是 Python 重写的 ASVLite 线性 Froude-Krylov 场，挂在 `world` 上，按船位查询。

场景 YAML 都在 `usage/27usv_world/`。换船、开浪、开雷达**只改 YAML**，Fossen `M/C/D` 公式不用动。

## 运行

```powershell
conda activate irsim
pip install -e ".[keyboard]"
cd usage/27usv_world
python usv_world.py          # USV14 双体
python wamv16_world.py       # WAM-V 16，可倒车
python monohull_world.py     # 单体演示
```

- `W` / `A` `D`：前进 / 差速转向
- `S`：仅当该船 `twj_min < 0` 时倒车（默认 `wamv` 不能反推）
- 脚本给推力：`env.step([150.0, 150.0])`，单位牛顿，不是 m/s

## 换船型

场景里只改 `robot.kinematics.vessel`，并让推力限幅跟这艘船的 yaml 一致。不写 `robot.shape` 时，顶视外形由 `irsim/vessels/<name>.yaml` 的 `hull.type` 生成。

| `vessel` | 文件 | 外形 | 推力盒 [N] |
|---|---|---|---|
| `wamv` | `irsim/vessels/wamv.yaml` | 双体 4.29 m | `[0, 204]` |
| `wamv16` | `irsim/vessels/wamv16.yaml` | 双体 5.0 m | `[-500, 1000]` |
| `monohull` | `irsim/vessels/monohull.yaml` | 单体 3.0 m | `[0, 40]` |

```yaml
robot:
  kinematics:
    name: usv
    vessel: wamv16          # 换成 wamv / wamv16 / monohull
    payload: 0.0            # 仅 wamv（USV14）有效；其它船 P_full=0
  # 不要手写 shape，否则外形不跟船走
  vel_min: [-500, -500]     # 与 actuators.twj_min 一致；省略则从 vessel 读
  vel_max: [1000, 1000]     # 与 actuators.twj_max 一致
gui:
  keyboard:
    key_lv_max: 1000.0      # 键盘 W/S 上限，跟 vel_max 同量纲 [N]
    key_ang_max: 1000.0
```

加一艘新船：复制 `irsim/vessels/wamv.yaml`，改 `geometry` / `load` / `hydro` / `hull.type` / `actuators.twj_min|twj_max`，场景里 `vessel:` 写成新文件名（不要后缀）。

- `hull.type: catamaran`：两根浮体 + 横梁
- `hull.type: monohull`：一块尖艏船体（`board` 当作单体）
- `hydro.recipe: klinger2017`：USV14 查表；其它船用 `hydro.coeffs` 填满 Fossen 系数

## 海浪：物理 vs 画面

**有没有浪**看有没有 `world.wave` 整块；**画不画水面**看 `world.plot.show_wave`。

```yaml
world:
  wave:                     # 删掉整个 wave: 块 = 无浪、无浪力
    Hs: 0.20                # 有义波高 [m]；Douglas 2 级。Hs=0.6 对 4 m 船太大
    Tp: 5.0                 # 谱峰周期 [s]
    heading: 0.0            # 浪向 [rad]，0 = +x
    n_freq: 20
    n_dir: 10
    seed: 0
  plot:
    show_wave: true         # false = 仍有浪力，只是不画高程图
    wave_cmap: ocean
    wave_alpha: 0.75
    wave_resolution: 1.0    # 网格 [m]
```

代码里读当前体轴浪力：

```python
tau = env.wave_wrench(env.robot)  # [X, Y, N]，单位 N / N / N·m；无浪则全 0
```

## 雷达：计算 vs 画面

**有没有雷达**看 `robot.sensors` 里有没有 `lidar2d`；**画不画射线**看 `robot.plot.show_sensor`。

```yaml
robot:
  sensors:
    - name: lidar2d
      range_min: 0.2
      range_max: 20.0
      angle_range: 6.28318530718   # 2π，360°
      number: 180
      noise: false
      offset: [0, 0, 0]            # 相对 CG [x, y, theta]
      plot:
        alpha: 0.35
  plot:
    show_sensor: true              # false = 不画射线，ranges 仍在算
```

多船场景默认没挂雷达。读扫描：

```python
scan = env.get_lidar_scan(0)  # dict: ranges, valid, angle_min, ...
pts = env.robot.get_lidar_points()  # (2, N) 世界系
```

`show_sensor: false` 只关图；要关掉雷达本身，删掉 `sensors` 里那一项。

## MAPPO / 训练接口

`irsim.make`、`step` 动作量纲、状态与雷达字段见 [PROJECT_MANIFEST.md](PROJECT_MANIFEST.md)。仿真器没有 `get_obs()`，`env.step()` 不返回 reward。
