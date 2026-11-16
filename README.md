# TG SuperMimic

稳定基线为 `bc_100hz`，整理后的开发分支为 `bc_100hz_dev`。
包内参考官方 SuperMimic 的 `models / learners / envs / runners` 分工，以及 BeyondMimic 的
`robots / tasks/tracking/mdp` 分层。运行入口按 Isaac Lab 示例放在 `scripts/`。
仿真仍是固定版本的 **Isaac Lab 3 kit-less / Newton / MuJoCo-Warp**；目录整理没有切换到 PhysX，
也没有将算法改为 PPO 或 ManagerBasedRLEnv。

## 目录

```text
src/supermimic/
  configs/              命令行兼容适配与配置序列化
  models/               WM、Policy、Critic 网络及 RunningNorm
  learners/             WM 监督更新、Actor 想象更新、Critic 回归
  envs/                 Isaac Lab 场景、仿真步进与 domain randomization
  robots/               G1、Tiangong3 参数及 FK
  motions/              motion 加载、插值、活跃池与起点采样
  storage/              replay buffer
  tasks/tracking/
    tracking_env_cfg.py 环境、command、obs、action、DR 范围、奖励项、终止阈值
    agents/             WM、Actor、Critic、replay、优化器与训练调度配置
    configs/tiangong3.py 天工 full WBC 的具体配置入口
    config.py           旧 RewardCfg/checkpoint 的兼容适配
    mdp/                state、actions、observations、rewards、terminations
  runners/              薄入口、训练依赖组装、采样、更新调度与 checkpoint
  logging/              每项奖励、回合统计、TensorBoard 和终端输出
  utils/                旋转数学、资产路径、录像和指标汇总
scripts/
  training/             训练入口
  evaluation/           播放、评测、录像、分析
  export/               ONNX 导出
  data/                 数据准备、评测子集
  assets/               URDF/MJCF/USD 转换与动作校验
  setup/                固定版本环境安装与检查
tests/                 unit、integration、benchmarks
experiments/            实验参数、记录及该实验专用服务器脚本
docs/                   使用说明、部署说明与迁移记录
data/ / vendor/         原有资产和固定第三方源码，位置不变
logs/                   运行结果，不纳入 Git
```

旧 `tools/` 的每个文件都已归类，完整映射见 [path_map.json](docs/refactor/path_map.json)。
包内旧模块的迁移、职责边界和验证记录见 [包结构说明](docs/refactor/PACKAGE_LAYOUT.md)。
特定服务器/GPU/日期的容量测试和排队脚本归入 `experiments/full_wbc_pool_20261111/scripts/`。
个人 `.vscode` 配置与日志不提交。

## 训练配置从哪里改

优先读 `src/supermimic/tasks/tracking/configs/tiangong3.py` 的 `full_wbc()`：
环境侧的 motion 池、DR、奖励和算法侧的 WM/Actor 窗口、batch、学习率放在一起组装。
共用字段分别定义在 `tracking_env_cfg.py` 和 `agents/supermimic_cfg.py`。
例如 `env.rewards.body_pos.weight/std` 控制身体位置奖励，
`runner.world_model.wm_window` 控制 WM 训练窗口，`runner.actor.horizon` 控制想象长度。

```bash
# 只显示分组配置，不加载仿真器、不启动训练。
python scripts/training/train.py --config tiangong3_full_wbc --print-config
# 命令行覆盖预设中的单项；目录按实际机器填写。
python scripts/training/train.py --config tiangong3_full_wbc \
  --motion_dir /media/supermimic/data/full_wbc_validated \
  --num_envs 2048 --wm_batch 2048 --pi_batch 2048 --print-config
```

`--config default`（省略时默认）完整保留旧命令行默认值。
`tiangong3_full_wbc` 预设显式使用 DexEVT、full DR + contact、30%/20% 池、
每卡环境/WM batch/Actor batch 各 4096、WM12/Actor48、gamma=.995。
预设本身不分配 GPU；两卡/四卡仍由启动器决定，池容量需要按卡数另行核对。
默认 `None` 表示延续已有的 checkpoint 继承规则；新训的参考窗仍为 1,6,...,46。
`--print-config` 是解析请求的预览，`resolved=false` 表示还未读取 checkpoint。
训练初始化后会保存实际分组 `config.json`，同时保留旧 `args.json` 和 checkpoint 的 flat args。

DR 的具体范围现在集中在 `tracking_env_cfg.py::RandomizationRangesCfg`，由
`env.events.ranges` 传到原仿真采样代码；不必再去 `envs/` 查硬编码数值。
`dr=""` 只关闭额外 DR，基础摩擦、质心、关节偏置和重置扰动仍保留。
full WBC 使用 `all,contact`：额外参数从第 3000 轮开始，push/delay 从第 6000 轮开始，
分别用 2000 轮逐渐放大范围。100 Hz 控制 / 400 Hz 物理下，4 子步延迟是 0～10 ms。
在 `full_wbc()` 的范围初始化之后，可以按下面方式覆盖单项：

```python
from dataclasses import replace

env.events.ranges = replace(env.events.ranges, kp_scale=(0.8, 1.2))
```

范围会保存到 `config.json` 和 checkpoint 的 `args.dr_ranges`；续训恢复这些值，
旧 checkpoint 则沿用机器人和 dynamics 对应的历史范围。显式预设与已保存范围不一致时会报错。
CLI 覆盖 robot/dynamics 时，同步选择对应的默认范围；预设中另行自定义的范围保留。
随机化评测也读取保存的范围；同一次评测不混用不同范围的 checkpoint。

奖励权重写在配置中，仿真与可微想象共用 `mdp/rewards.py`；每项先加权，最后乘 dt。
正奖励权重变化时，最大回报、采样价值上限和日志归一化同步计算。
续训/评测按 checkpoint 恢复保存的奖励和终止阈值；旧 checkpoint 缺少的新增权重使用原值。
若显式预设与 checkpoint 的奖励不同，报错而非静默换目标。
当前主开发基线的 `tiangong3_full_wbc` 预设已采用相对高度和单脚底点：

```python
env.rewards.root_pos_z_weight = 0.1
env.rewards.body_pos_z_mode = "relative_root"
env.rewards.body_point_offsets = {
    "ankle_roll_l_link": (0.090, 0.0, -0.058),
    "ankle_roll_r_link": (0.090, 0.0, -0.058),
}
```

骨盆位置奖励使用 `ex² + ey² + 0.1*ez²`，整项权重仍为 0.5；link 高度相对骨盆计算。
每脚一个足底前部点替代原 14 点中对应的 ankle-roll 原点；相对踝原点向前90 mm、向下58 mm。
前移点位提高对脚部俯仰误差的敏感度；没有新增 foot 奖励项或改变其权重。
位置使用 `p + R*r`，同一点的线速度使用 `v + omega × (R*r)`，实际与参考使用相同换算。
身体姿态/角速度奖励保留，绝对骨盆高度的 0.25 m 终止阈值仍保留。
这是奖励侧的点位替换：原始 FK/motion 缓存、Actor/WM/Critic 输入、网络、PD、URDF、碰撞和 DR 不变。
仿真与想象共用该奖励，日志同时保留绝对 Z 和相对 Z 误差。

新基线应显式选 `--config tiangong3_full_wbc`；`--config default` 为旧命令兼容入口，
缺少点位字段的旧 checkpoint 仍按 link 原点评分，旧绝对/相对 Z 设置也按 checkpoint 恢复。
普通 resume 不会把旧策略静默改成新目标；这次只合代码，没有开始训练或迁移现有权重。
完整分支清单及清理建议见 `docs/refactor/BRANCHES_20261116.txt`。

## 训练调度与日志整理（`bc_100hz_dev`）

`runners/train.py` 只负责解析参数、选择机器人、启动仿真并调用 Runner。
`runners/setup.py` 创建 motion、网络和优化器，并恢复 checkpoint；
`supermimic_runner.py` 调度 collect → WM → Actor/Critic；`collector.py` 负责真实采样；
`checkpoints.py` 管理保存，`metrics.py` 做跨卡统计归并。已有 learners 的数学计算保持原样。

日志分为两个目录：

- `tensorboard/`：`Reward` 每项有符号的每秒贡献、`Episode_Reward` 回合累计奖励除以回合时长上限、
  `Metrics` 跟踪误差和加速度、`Loss` 实际 WM/Actor/Critic Huber 损失、`Perf` 速度和内存。
- `tensorboard_diagnostics/`：原始指标口径、WM 逐分量误差/归一化尺度等详细诊断。

终端逐行展示奖励贡献，奖励分项按物理秒显示；CSV 原有字段和每步单位保留，新增回合字段追加在末尾。
回合统计汇总本次日志窗口结束的所有回合，并按跨卡回合数加权。
旧 run 首次用新 Logger 续训时，旧事件移至 `tensorboard_legacy/` 保留，再从 CSV 重建主面板；
旧 CSV 没有回合分项时不生成假的 `Episode_Reward`。

只想查看旧训练的新面板，无需续训：

```bash
python scripts/evaluation/log_dashboard.py --run_dir logs/<run> --output logs/<view>/<run>
tensorboard --logdir logs/<view> --host 127.0.0.1 --port 6015
```

修改记录与验证范围见 [训练日志重构记录](docs/refactor/TRAINING_LOGGING.txt)。

## 环境与常用入口

已配置好固定版本训练环境时，只需安装本项目源码；不重新解析或升级仿真依赖：

```bash
python -m pip install --no-deps --no-build-isolation -e .
python scripts/training/train.py --help
```

新机器仍用固定版本安装脚本：

```bash
bash scripts/setup/setup.sh
conda activate tg_supermimic
```

详细依赖、镜像、Conda 和 CUDA 约束见 [部署说明](docs/DEPLOY_CONDA.md)。
普通 Isaac Lab 2.3 环境不能替代这里的固定 kit-less/Newton 环境。

启动器使用方式保持一致：

```bash
./sm.sh train <run_name> --gpu 0 --robot tiangong3 --motion_dir <validated_motion_dir> [训练参数]
./sm.sh attach <run_name>
./sm.sh stop <run_name>
```

| 用途 | 新入口 |
| --- | --- |
| 直接训练 | `python scripts/training/train.py ...` 或安装后的 `tg-train ...` |
| 批量原生评测 | `python scripts/evaluation/evaluate.py ...` 或 `tg-evaluate ...` |
| 原生播放 | `python scripts/evaluation/play_native.py --ckpt ... --motion ... --control-hz 100` |
| MuJoCo sim2sim | `python scripts/evaluation/sim2sim_mujoco.py ...` |
| SMWBC 导出：obs + goal 输入 | `python scripts/export/export_onnx_policy.py --ckpt ...` |
| 固定 motion 导出：SMDANCE | `bash scripts/export/export_onnx.sh <ckpt> <motion.npz>` |
| NPZ 准备 | `python scripts/data/prepare_motion_dir.py ...` |
| 完整 WBC 四卡预设 | `python experiments/full_wbc_pool_20261111/scripts/run_pool_training.py --gpus 0,1,2,3` |

最后一项包含实验记录中的 `/media/...` 路径和环境配置，使用前按机器核对。
两种 ONNX 接口用途不同，文件分别保留。

## 未来参考时间窗

`bc_100hz_dev` / `bc_100hz_muon` 新训练默认 `--goal_frames 10 --goal_stride 5 --goal_start 1`：
100 Hz 下读取 `t+1, t+6, …, t+46`，覆盖 10～460 ms，输入仍为 10×73 维。
历史仍是连续 10 帧；控制、replay、WM 仍每步 10 ms。参考越过 clip 尾部时重复该 clip 最后一帧。
XYZ 特征仍为未来参考位置减当前参考位置、转到当前参考航向系，不增加真实机器人 XYZ 输入。

实际 start、stride 和 offsets 写入 args/checkpoint/ONNX，评估与导出按 checkpoint 读取。
旧 checkpoint 无 start 时保持从一个 stride 开始，无 stride 时保持连续 t+1…t+10。
普通续训保持旧时间窗；明确改变已有权重的输入语义时，需要指定 `--goal_start 1 --goal_stride 5`
及 `--allow_reference_timing_change`。这会保留网络、优化器和归一化统计，并在 args/checkpoint 的
`reference_timing_changes` 记录旧/新时刻及切换迭代；属于续训适应，不是从零训练对照。
历史实验预设保留原时间窗，新预设须明确填 `goal_start` 和 `goal_stride`。
池训练启动器的对应开关为 `--allow-reference-timing-change`。

## 验证

```bash
python -m pytest -q
./sm.sh run --robot tiangong3 python tests/integration/test_env.py --num_envs 16 --motion_dir <validated_motion.npz>
```

默认 pytest 仅运行 CPU 单元与回归测试；需要真实模拟器的测试在 `tests/integration/`，不会被自动混入。
迁移对照、检查结果及未验证部分见 [重构记录](docs/refactor/README.txt)。
新增录像写入 `control_dt`；旧录像无频率元数据时，渲染/分析需显式传 `--control-hz 50` 或 `100`。

- [完整操作手册](docs/HOW_TO_USE.md)
- [随库资产与许可](docs/DATA_ASSETS.md)
- [之前的仿真验收记录](docs/VALIDATION.md)
- [原始项目说明](docs/history/README_UPSTREAM.md)

项目代码沿用 MIT License；第三方源码与数据遵循各自许可。

### Motion 活跃池

Use `--motion_active_ratio 0.3 --motion_refresh_iters 1500 --motion_replace_ratio 0.2`
for a WBT-style rotating subset on each rank. The ratio is a fraction of clips, not frames;
there is no byte/frame capacity cap. Replacement counts refer to the full catalog.
The default ratio `1` retains the original dense loader.

Only active clips receive new episode starts. Clips still referenced by running environments
or **any valid replay row** stay resident until a later refresh can release them. Refreshes
run between learner iterations and remap environment frames, episode-start bins and circular
replay frame indices together. Replay and episode state are not cleared. A CPU cache reuses
resident trajectories; only new clips are read and passed through the original interpolation
and FK pipeline. Root/joint/goal/index fields stay on GPU. Float32 body reference tables are cached per clip on CPU;
queried batches transfer to GPU before reward/critic kernels. No reference quantization is used.

Return-based one-second-bin sampling remains in use, conditioned on the current active set.
This changes the temporal availability of data and is an additional experiment, not a pure
CPU/GPU storage equivalence test. Full-catalog action normalization statistics are streamed
once and cached under `.cache/motion_pool`; they do not change at pool refreshes. Checkpoints
save pool selection, sampler statistics, coverage counters and RNG state. Replay is refilled
on resume as before; keep world size, active ratio and refresh schedule when resuming.

Progress, CSV and TensorBoard include weighted per-step reward components, their fraction
of each term's maximum, penalties, and real-simulator tracking RMSE with units. Body pose
errors use the original XY/yaw re-anchored reference; root errors are global. Orientation
RMSE inherits the original reward's small acos clamp floor (about1.4mrad). Joint RMSE is
per joint, body position/velocity RMSE is a norm per tracked body. Action delta is measured
in raw policy-action units. Recent success means no physical failure before clip/time end,
not successful traversal of every frame of a motion. Finished-episode means now weight
each episode equally, rather than each vector step that happened to contain an ending.
Pool logs distinguish active/resident/legacy counts, tensor memory, refresh latency and
motion-start coverage; none of these claims full-motion coverage.

### Full WBC 实验配置

Current preset: 100 Hz control / 400 Hz physics, action-rate -0.4, joint-acc -3e-6, gamma .995,
WM12 / Actor48; four GPUs, requested4096 environments and4096 WM/Actor batch per GPU.
Active30%, replacement20% of the full catalog, every1500 learner iterations.
Exact parameters, validation scope and comparison caveats: [experiment record](experiments/full_wbc_pool_20261111/README.txt).
Historical SONIC pool preset is at commit `a8266ac`; do not mix its launch arguments with the new reward defaults.
