# PROJECT_MANIFEST

IR-SIM USV ↔ light_mappo EnvCore 接口规约。仅 API 与数据格式。

IR-SIM **没有** `get_obs()`；`env.step()` **不返回** `(obs, reward, done)`。对接 MAPPO 必须在 EnvCore 中组装。

---

## 1. IR-SIM 初始化

```text
env = irsim.make(world_name, **kwargs) -> EnvBase
```

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `world_name` | `str` | 调用脚本同名 `.yaml` | 场景 YAML 路径 |
| `projection` | `str \| None` | `None` | `None`=2D；`"3d"`=3D |
| `display` | `bool` | `True` | 是否开窗口 |
| `headless` | `bool` | `False` | 训练用；隐含 `display=False` |
| `save_ani` | `bool` | `False` | 是否存动画 |
| `log_level` | `str` | `"INFO"` | |
| `seed` | `int \| None` | `None` | |
| `step_mode` | `"internal" \| "external"` | YAML / `"internal"` | 训练用 `"internal"` |

USV 训练建议：

```python
env = irsim.make(
    "multi_usv_world.yaml", display=False, headless=True, log_level="WARNING"
)
```

YAML（当前 27 多船场景）硬约束：

```text
world.width, world.height     : 80 m, 50 m
world.offset                  : [0, 0] m
world.step_time               : 0.05 s
world.control_mode            : auto          # 训练禁止 keyboard
world.collision_mode          : unobstructed  # 多船 demo；单船 demo 为 stop
robot.number                  : 5
robot.kinematics.name         : usv
```

---

## 2. `env.step(actions)` — Action

IR-SIM 原生：

```text
env.step(action, action_id=None) -> None
```

| 项 | 规格 |
|---|---|
| 单船 | `[T_port, T_stbd]` |
| 多船 list | `[[T_p, T_s], ...]` 长度 = `N`，顺序 = `env.robot_list`（id `0..N-1`） |
| 多船 dict | `{ "robot_i": [T_p, T_s], ... }` |
| dtype | `float` |
| 维度 | 每船 `action_dim = 2` |
| 量纲 | **牛顿 [N]**，不是速度 |
| 范围 | 当前 `kinematics.vessel` YAML 的 `actuators.twj_min` … `twj_max`（场景 `vel_min` / `vel_max` 可覆盖） |
| 默认 `wamv`（USV14） | `T ∈ [0, 204]`：Klinger 2017 `twj_max`，喷泵不反推 |
| `wamv16` | `T ∈ [-500, 1000]`：AC-IQN 推力盒，允许倒车 |
| `monohull` | `T ∈ [0, 40]`：演示单体 |

YAML `vel_max` / `vel_min` / `key_lv_max` 对 `kinematics: usv` 是推力限幅 [N]，不是 m/s。换船：改 `kinematics.vessel`（`wamv` / `wamv16` / `monohull` 或自备 yaml）。不写 `robot.shape` 时顶视外形由 `hull.type`（`catamaran` / `monohull`）从 vessel yaml 生成。

步进后状态由调用方自行读取。

---

## 3. 状态 / 观测中的绝对物理量

IR-SIM 无 `get_obs()`。可读量如下。

### 3.1 船体状态（绝对）

```text
robot.state  : np.ndarray, shape (6, 1)
env.get_robot_state()  : 仅第一艘，shape (6, 1)
```

| 下标 | 符号 | 含义 | 单位 | 范围（场景内） |
|---|---|---|---|---|
| 0 | `x` | 世界系位置 | m | `[0, 80]` |
| 1 | `y` | 世界系位置 | m | `[0, 50]` |
| 2 | `psi` | 艏向，+x=0，逆时针 | rad | 无包裹保证，通常 `(−π, π]` |
| 3 | `u` | 体轴纵荡 | m/s | 由水动力决定 |
| 4 | `v` | 体轴横荡（+y 向左舷） | m/s | 由水动力决定 |
| 5 | `r` | 艏摇角速度 | rad/s | 由水动力决定 |

体轴原点 = CG；+x 向前，+y 向左舷。

### 3.2 激光雷达（当前单船 YAML）

```text
env.get_lidar_scan(id) -> dict
env.robot_list[id].get_lidar_points() -> np.ndarray (2, N)
```

| 字段 | 形状 | 单位 | 说明 |
|---|---|---|---|
| `ranges` | `(180,)` | m | 束测距；未命中 = `range_max` |
| `valid` | `(180,)` bool | — | 有效回波 |
| `angle_min` | scalar | rad | `−π` |
| `angle_max` | scalar | rad | `+π` |
| `angle_increment` | scalar | rad | `2π / 179` |
| `range_min` | scalar | m | `0.2` |
| `range_max` | scalar | m | `20.0` |
| points | `(2, 180)` | m | 世界系 `[x; y]` |

`number: 180`，`angle_range: 2π`。多船 YAML **未挂雷达**。

### 3.3 其它可读量（非观测默认）

```text
env.wave_wrench(robot) -> np.ndarray (3,)   # 体轴 [X, Y, N]，单位 N, N, N·m
env.done(mode="all"|"any") -> bool          # 全部/任一 robot.done()
robot.done()                               # stop_flag or arrive_flag
env.reset(random=False) -> None
```

---

## 4. light_mappo `EnvCore` 标准接口

实现文件：`envs/custom_env/env_core.py`（[tinyzqh/light_mappo](https://github.com/tinyzqh/light_mappo)）。

```python
class EnvCore:
    def __init__(self):
        self.agent_num: int  # N
        self.obs_dim: int  # 每智能体观测维
        self.action_dim: int  # 每智能体动作维

    def reset(self) -> list[np.ndarray]:
        # len = agent_num
        # 每个元素 shape = (obs_dim,), dtype = float32/float64
        ...

    def step(self, actions) -> list:
        # actions: list, len = agent_num
        # 每个元素 shape = (action_dim,)
        # 返回 [obs_list, reward_list, done_list, info_list]
        ...
```

### 4.1 返回值形状（硬约束）

| 项 | 类型 | 形状 |
|---|---|---|
| `obs_list[i]` | `np.ndarray` | `(obs_dim,)` |
| `reward_list[i]` | `list` | `[float]`，必须是单元素列表 |
| `done_list[i]` | `bool` | scalar |
| `info_list[i]` | `dict` | `{}` 即可 |

连续动作由 `env_continuous.py` 包成 `gym.spaces.Box`。策略输出通常在 **`[−1, 1]`**；EnvCore 内必须映射到喷泵牛顿。

### 4.2 USV 对接建议（尚未实现，仅格式）

```text
agent_num  = N = 5
action_dim = 2
obs_dim    = 6 + 180 = 186     # 若使用船体状态 + 雷达；无雷达则为 6

# 策略 a ∈ [−1, 1]^2  ->  仿真 T ∈ [0, 204]^2
T = 0.5 * (a + 1.0) * 204.0

# 每船 obs（绝对量拼接，顺序固定）
obs = concat(
    state[0:6, 0],          # (6,)  x,y,psi,u,v,r
    lidar.ranges            # (180,) m ; 无雷达则省略
)

# 转 IR-SIM
env.step([T_0, T_1, ..., T_{N-1}])   # 各 T_i shape (2,)
```

`reset` / `step` 必须返回上述 list 结构；禁止把 IR-SIM 的 `None` 直接交给 MAPPO。
