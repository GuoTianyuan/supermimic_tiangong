# 包内职责重构

开发分支：`bc_100hz_dev`。对照起点：`ce8955e`，即此前只完成 `src/` 搬迁及本地环境配置的版本。

## 参考与实际拆分

参考本地官方 SuperMimic `supermimic_ws/supermimic` 中的 `learners/`、`envs/`、`runners/`，
以及 BeyondMimic `whole_body_tracking` 中的 `robots/` 和 `tasks/tracking/mdp/`。
官方的 `models.py` 进一步按 WM、Policy、Critic 拆成文件。
项目继续使用现有 Isaac Lab / Newton / MuJoCo-Warp 环境和想象训练算法。

| 要修改的内容 | 当前入口 |
| --- | --- |
| Tiangong PD、armature、action scale、资产 | `src/supermimic/robots/tiangong3.py` |
| G1 配置、机器人选择、FK | `robots/g1.py`、`robots/__init__.py`、`robots/kinematics.py` |
| 仿真场景、物理时间步和 solver | `envs/sim_context.py` |
| 动作执行、reset、DR 应用 | `envs/tracking_env.py` |
| DR 参数及 ξ 布局 | `envs/domain_randomization.py` |
| motion 读取、插值、参考查询 | `motions/library.py` |
| 活跃池、replay 引用保留及换池 | `motions/pool.py` |
| 起点采样统计 | `motions/sampling.py`；环境采样调用在 `envs/tracking_env.py` |
| replay 窗口、历史补齐 | `storage/replay_buffer.py` |
| 奖励权重、阈值 | `tasks/tracking/config.py` |
| 状态打包、ξ 归一化 | `tasks/tracking/mdp/state.py` |
| observation、噪声、Critic 特征 | `tasks/tracking/mdp/observations.py` |
| reward 公式 | `tasks/tracking/mdp/rewards.py` |
| 硬终止、解析 sigmoid 终止 | `tasks/tracking/mdp/terminations.py` |
| action bounds、soft clip | `tasks/tracking/mdp/actions.py` |
| 网络结构和归一化层 | `models/world_model.py`、`policy.py`、`critic.py`、`common.py` |
| WM rollout、监督 loss 和更新 | `learners/wm_learner.py` |
| Actor 想象、目标和更新 | `learners/policy_learner.py` |
| Critic Huber 回归、λ-return、EMA | `learners/critic_learner.py` |
| 训练参数默认值 | `configs/train.py` |
| 数据收集、更新调度、保存和续训 | `runners/train.py` |
| CLI / 终端、CSV、TensorBoard | `cli.py` / `runners/progress.py` |

表中未写完整路径的文件均相对 `src/supermimic/`。这个目录顶层只留 `__init__.py` 和 `cli.py`。
旧模块没有保留第二份实现或兼容转发文件；仓库脚本、测试及实验脚本已同步改 import。
外部自写脚本需要按 [package_map.json](package_map.json) 修改 import。
训练命令和保存的参数名仍使用原接口。

## 分层约定

- `models` 定义可微计算和网络参数；`learners` 持有这些网络及 optimizer 的引用并执行更新。
  它们不重新创建网络。builder 保留原函数的闭包和 `torch.compile` 边界，避免因重构改变编译输入。
- `runners/train.py` 负责生命周期和调度；WM、Actor、Critic 的更新实现已移出。
- `tasks/tracking/mdp/context.py` 将观测、奖励和终止函数绑定到同一份 motion、FK 和配置，
  保留模拟器、想象轨迹使用的 `MDP` 调用接口。
- `motions` 管理参考数据，`storage` 管理实际 transition；池切换和 replay 索引契约不变。
- 机器人由入口解析参数后选择，之后才导入依赖机器人的任务和网络。
- AMP 开关属于 `models.common.AMP_DTYPE`；入口修改此处，避免修改聚合模块中的副本。
- 资产和运行产物不搬进 Python 包；资产根路径由 `utils/paths.py` 解析。

## 验证方式

从重构前保存函数与 runner 的源码，对拆分后逐个 AST 比较；Actor 中的 Critic 更新还原到原位置后再比较。
测试保留训练 CLI 默认值、机器人参数和未拆分数值模块的历史快照。
同时运行真实 GPU 仿真、短训、续训、旧 checkpoint、motion pool 和 ONNX 校验，覆盖 import 和运行时连接。
具体执行结果见本目录 `package_validation.json`，完整日志保留在本地 `logs/package_layout_20261112/`。

验证范围为本地小数据、单 GPU；多 rank/full WBC 长训未在此次目录重构中重新执行。
不以小规模功能验证代替收敛或真机效果验证。

## 本次执行结果

- 36 项单元与源码回归测试通过。
- 固定 replay 连续 3 次 WM、Actor、Critic 更新：loss、梯度、权重、RunningNorm、Adam 状态和 EMA 逐位一致。
- 16 环境真实 GPU 短训到 30 轮，保存后续到 35；重构前的 checkpoint 也可续训到 40。
- 509 项 MuJoCo 模型数组及选项与重构前逐位相同。
- 小型活跃池换池 12 次，含 10 次编译查询，参考、采样、replay 重映射、梯度和恢复检查通过。
- 现有 20,000 轮 full-WBC checkpoint 完成单条 motion 原生播放及录像评测。
- ONNX 权重与重构前相同；8 组输入对 Torch 的最大绝对误差为 2.03e-6。

两次独立的编译仿真短训并非逐位一致，具体差异也保存在验证 JSON 中；
固定 replay 的等价对照排除了该比较中的重新采样、仿真轨迹和编译 RNG 影响。
不宣称独立长训的权重或收敛轨迹逐位复现。
