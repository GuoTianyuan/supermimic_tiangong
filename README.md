# TG SuperMimic

开发基线为 `bc_100hz`，目录整理在 `refactor/isaaclab-layout`。
源码采用官方仓库的 `src/` 包结构，运行入口按 Isaac Lab 示例放在 `scripts/`。
仿真仍是固定版本的 **Isaac Lab 3 kit-less / Newton / MuJoCo-Warp**；目录整理没有切换到 PhysX，
也没有将算法改为 PPO 或 ManagerBasedRLEnv。

## 目录

```text
src/supermimic/          可安装的 Python 包，公共 import 名称保持 supermimic.*
  configs/train.py       参数定义与默认值，单独导入不启动仿真
  runners/train.py       收集数据、WM、Actor/Critic 更新及 checkpoint
  runners/evaluate.py    原生环境评测
  runners/progress.py    终端、CSV、TensorBoard 格式
  env.py / mdp.py        仿真环境、观测、奖励和终止
  robot.py / tiangong3.py 机器人参数与选择
  motion.py / motion_pool.py 参考动作和活跃池
  networks.py / buffer.py / sampling.py  网络、replay、采样
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
特定服务器/GPU/日期的容量测试和排队脚本归入 `experiments/full_wbc_pool_20261111/scripts/`。
个人 `.vscode` 配置与日志不提交。

## 环境与常用入口

已配置好固定版本训练环境时，只需安装本项目源码；不重新解析或升级仿真依赖：

```bash
python -m pip install --no-deps --no-build-isolation -e .
python scripts/training/train.py --help
```

新机器仍用固定版本安装脚本：

```bash
bash scripts/setup/setup.sh
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
