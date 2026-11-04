# ir-sim-usv

WAM-V 无人船仿真（基于 [IR-SIM](https://github.com/hanruihua/ir-sim)）。

## 运行 USV demo

在仓库根目录安装并启动键盘 demo：

```powershell
conda activate irsim
pip install -e ".[keyboard]"
cd usage/27usv_world
python usv_world.py
```

窗口出现后：

- `W`：双喷泵前进
- `A` / `D`：差速转向
- `S`：无效（喷泵不能反推）

无键盘时也可以直接给推力：

```python
env.step([150.0, 150.0])  # 左、右喷泵推力 [N]
```

开环脚本：`python usv_thrust.py`

## 激光雷达可视化

改 `usage/27usv_world/usv_world.yaml` 里机器人的 `plot.show_sensor`：

```yaml
robot:
  plot:
    show_sensor: true    # 显示雷达射线
    # show_sensor: false # 关掉可视化；雷达仍在算数
```

保存后再跑 `python usv_world.py`。
