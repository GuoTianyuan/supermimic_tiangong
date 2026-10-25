# How to use（h800 分支）

所有命令都在仓库根目录执行。

- **H800 服务器**（`ssh -p 2210 root@10.108.3.194`，项目在 `/media/roy/supermimic_vibed`）：没有 Docker，用 `./sm_native.sh`。
  **多卡训练只能用它**。
- **本机**（Docker）：用 `./sm.sh`，子命令相同，但只支持单卡。

下面的命令都按服务器写；在本机跑单卡时把 `./sm_native.sh` 换成 `./sm.sh`。

## 0. 新加的随机化：`--dynamics dexevt`（与 BeyondMimic 对齐）

### 怎么开

训练命令里加 **`--dynamics dexevt --dr all`** 就行。在 H800 上多卡、全量动作，直接复制：

```bash
cd /media/roy/supermimic_vibed

# 8 卡全量训练，新的 BeyondMimic 随机化
./sm_native.sh train wbc_full_dexevt --gpu all --robot tiangong3 --motion_dir data/wbc_full \
  --dynamics dexevt --dr all --dr_curriculum 3000,6000 \
  --num_envs 8192 --buffer_steps 1000 --iterations 320000 --ckpt_every 2000 -d

# 或者：7 卡训练 + 第 8 张卡持续评估（两条命令都要跑）
./sm_native.sh train wbc_full_dexevt --gpu 0,1,2,3,4,5,6 --robot tiangong3 --motion_dir data/wbc_full \
  --dynamics dexevt --dr all --dr_curriculum 3000,6000 \
  --num_envs 8192 --buffer_steps 1000 --iterations 320000 --ckpt_every 2000 -d
./sm_native.sh run --gpu 7 --robot tiangong3 python supermimic/evaluate.py --run_dir logs/wbc_full_dexevt \
  --motion_dir data/wbc_full_eval --watch --every 5

# 查看 / 停止
./sm_native.sh status
./sm_native.sh logs wbc_full_dexevt
./sm_native.sh stop wbc_full_dexevt

# 停了之后接着训练（同一条命令末尾加 --resume）
./sm_native.sh train wbc_full_dexevt --gpu all --robot tiangong3 --motion_dir data/wbc_full \
  --dynamics dexevt --dr all --dr_curriculum 3000,6000 \
  --num_envs 8192 --buffer_steps 1000 --iterations 320000 --ckpt_every 2000 -d \
  --resume logs/wbc_full_dexevt/last.pt
```

第一次跑之前要先准备 `data/wbc_full` 和 `data/wbc_full_eval`（见 1.1）。各参数含义见 1.2。

- 只写 `--dr all`、不写 `--dynamics`：是 `default`，和加这个功能之前完全一样。
- 只写 `--dynamics dexevt`、不写 `--dr`：只改脚踝力矩上限和标称 armature，不做随机化。一般不要这样用。

### 和之前（`default`）相比改了什么

| | 之前 `default` | 新的 `dexevt` | BeyondMimic DexEVT |
|---|---|---|---|
| 脚踝力矩上限 | 100 N·m | **55 N·m** | 55 N·m |
| armature 随机范围 | 其他关节 0.002–0.06，脚踝 0.01–0.06 | **所有关节 0.002–0.06** | 所有关节 0.002–0.06 |
| 脚踝标称 armature | 0.035 | **0.031**（与其他关节相同） | — |
| 关节摩擦随机 | 0.001–0.6，只有库仑摩擦 | **0.001–0.6，同一个值同时作为库仑摩擦和粘滞摩擦** | 静摩擦、动摩擦、粘滞摩擦都 0.001–0.6 |

### 全部随机化项（`--dr all` 打开的）

| `--dr` 项 | 内容 | 抽样时机 | 本仓库（两种 `--dynamics` 相同的项不重复写） | BeyondMimic |
|---|---|---|---|---|
| `pd` | 每个关节 Kp、Kd 分别缩放 | 每回合 | ×0.75–1.25 | ×0.75–1.25 |
| `mass` | 各刚体质量和惯量 | 启动时 | ×0.85–1.15 | ×0.85–1.15 |
| `armature` | 每个关节 | 启动时 | 见上表 | 0.002–0.06 |
| `joint_friction` | 每个关节 | 启动时 | 见上表 | 0.001–0.6 |
| `delay` | 控制延迟 | 每回合 | 0–4 个 5 ms 子步，所有关节同一个值 | 0–4 个 5 ms 子步，按执行器组分别抽 |
| `push` | 根速度随机增量 | 每 1–5 s | x/y ±0.5、z ±0.2 m/s；roll/pitch ±0.52、yaw ±0.78 rad/s | 相同 |
| （始终开） | 地面摩擦 | 启动时 | 0.3–1.2 | 静 0.6–1.6、动 0.6–1.2、恢复系数 0–0.5 |
| （始终开） | 躯干质心偏移 | 启动时 | x ±0.025、y/z ±0.05 m | x −0.025–0.035、y/z ±0.05 m |

`--dr_curriculum A,B` 让这些项分阶段打开：第 A 次迭代起放宽除 `push`、`delay` 外的各项，第 B 次起放宽 `push`、`delay`；
每个阶段用 `--dr_ramp`（默认 2000）次迭代从标称值放宽到完整范围。

### 代码在哪

| 文件 | 内容 |
|---|---|
| `supermimic/tiangong3.py` | `DYNAMICS`、`motor_params()`（dexevt 时脚踝 55 N·m、armature 0.031）、`set_dynamics()` |
| `supermimic/dr.py` | `DRCfg.for_dynamics()`：dexevt 时 `ankle_armature=(0.002, 0.06)`、`viscous_friction=True` |
| `supermimic/env.py` | `_apply_startup_dr()`：`viscous_friction` 为真时把摩擦也写成粘滞摩擦 |
| `supermimic/train.py` | `--dynamics` 参数，存进 checkpoint 和 `logs/<run>/args.json` |
| `supermimic/evaluate.py`、`tools/sim2sim_mujoco.py` | 不写 `--dynamics` 时自动读 run 或 checkpoint 里的值 |

### 怎么确认生效

训练开始时打印：

```
 Joint dynamics: dexevt
 Domain randomization: friction, joint offsets, torso COM + pd, push, mass, armature, joint_friction, delay (xi 180)
```

`logs/<run>/args.json` 里有 `"dynamics": "dexevt"`。进度块里的 `DR range scale` 是当前随机范围的比例，
课程开始前是 0%，放宽完是 100%。

### 和 BeyondMimic 仍然不同的地方（这个选项没有改）

- 控制频率 50 Hz（BeyondMimic 100 Hz）；仿真里关闭了自碰撞（BeyondMimic 全身自碰撞）。
- 奖励：这里是论文的跟踪奖励 + `action_rate` −0.1 + `joint_limit` −2；没有 BeyondMimic 的力矩超限惩罚（课程到 −2）、
  `feet_slide`、`joint_torque`。
- 终止：这里是根高度误差 > 0.25 m 或倾斜 > 0.8；BeyondMimic 用持续若干步的高度和姿态误差，以及脚踝高度。
- 地面摩擦、躯干质心范围和延迟的抽法见上面的随机化表。

## 1. 多卡全量动作训练（最常用）

### 1.1 准备全量数据（只需一次；数据有更新时重跑，已校验的会跳过）

```bash
cd /media/roy/supermimic_vibed
# 校验 /media/wbc_full 的每条 NPZ，合格的软链接到 data/wbc_full（原目录不写入）
./sm_native.sh run python tools/prepare_motion_dir.py --src /media/wbc_full --out data/wbc_full
# 每个动作大类抽 32 条做评估集（训练用全量，评估用这个小集合）
./sm_native.sh run python tools/make_eval_subset.py --src data/wbc_full --out data/wbc_full_eval --per_group 32
```

### 1.2 训练

用新加的 BeyondMimic 随机化，8 卡全上。第二行的 `--dynamics dexevt --dr all` 就是第 0 节的随机化：

```bash
./sm_native.sh train wbc_full_dexevt --gpu all --robot tiangong3 --motion_dir data/wbc_full \
  --dynamics dexevt --dr all --dr_curriculum 3000,6000 \
  --num_envs 8192 --buffer_steps 1000 --iterations 320000 --ckpt_every 2000
```

要同时用一张卡持续评估时，训练只用 7 张卡，第 8 张卡跑评估（见 1.4）：

```bash
./sm_native.sh train wbc_full_dexevt --gpu 0,1,2,3,4,5,6 --robot tiangong3 --motion_dir data/wbc_full \
  --dynamics dexevt --dr all --dr_curriculum 3000,6000 \
  --num_envs 8192 --buffer_steps 1000 --iterations 320000 --ckpt_every 2000 -d
```

各参数的含义：

| 参数 | 含义 |
|---|---|
| `--gpu all` / `--gpu 0,1,2,3` | 每张卡一个进程，一起训练同一个策略。动作文件按文件名轮流分到各卡，梯度和归一化统计在各卡之间平均 |
| `--num_envs 8192` | **每张卡**的环境数（`--wm_batch`、`--pi_batch` 也是每卡的值）。8 卡 × 8192 = 65,536 个环境，每卡约 41 GB 显存（不含 DR），约 2.0 it/s |
| `--buffer_steps 1000` | replay buffer 每个环境保存 1000 步；开 `--dr` 后每 1024 个环境多占约 0.6 GB |
| `--dynamics dexevt` | 关节参数和 DR 范围与 BeyondMimic DexEVT 对齐，见第 0 节。不加时为 `default` |
| `--dr all` | 打开全部扩展随机化（`pd,push,mass,armature,joint_friction,delay`），见第 0 节 |
| `--dr_curriculum 3000,6000` | 第 3000 次迭代前只有论文原有的随机化；从 3000 起放宽除推力和延迟外的各项，从 6000 起放宽推力和延迟 |
| `--dr_ramp 2000`（默认） | 每个阶段用 2000 次迭代从标称值逐步放宽到完整范围 |
| `--iterations 320000` | 总迭代数，8 卡约 47 小时；学习率按 `--lr_decay_iters 250000` 余弦衰减 |
| `--ckpt_every 2000` | 每 2000 次迭代存一个 `logs/<run>/ckpt_xxxxxx.pt`，另外一直更新 `last.pt` |
| `-d` | 后台启动（不进入 tmux） |
| `--motion_exclude REGEX` | 不加载匹配的动作文件，例如 `'^(crawl|lie_down|get_up|fuwocheng)__'` |

启动时每卡加载动作约 4.5 分钟，再编译约 1 分钟，之后才开始打印进度。前 500 次迭代只训练世界模型。

不加 `--dynamics`（即 `default`）就是之前的随机化：

```bash
./sm_native.sh train wbc_full_dr_v4 --gpu all --robot tiangong3 --motion_dir data/wbc_full \
  --dr all --dr_curriculum 3000,6000 \
  --num_envs 8192 --buffer_steps 1000 --iterations 320000 --ckpt_every 2000
```

### 1.3 查看、停止、接着训练

```bash
./sm_native.sh status                    # 所有训练的进度、ETA、TensorBoard 端口、各卡显存
./sm_native.sh logs wbc_full_dexevt      # 跟随 rank 0 的输出（Ctrl+C 只退出查看）
./sm_native.sh attach wbc_full_dexevt    # 回到训练终端；Ctrl+B 再按 D 离开，训练继续
./sm_native.sh stop wbc_full_dexevt      # 所有卡跑完当前迭代、保存 checkpoint 后退出
./sm_native.sh stop --all

# 接着训练：原命令末尾加 --resume（--dynamics 不写时沿用 checkpoint 里的）
./sm_native.sh train wbc_full_dexevt --gpu all --robot tiangong3 --motion_dir data/wbc_full \
  --dr all --dr_curriculum 3000,6000 \
  --num_envs 8192 --buffer_steps 1000 --iterations 320000 --ckpt_every 2000 \
  --resume logs/wbc_full_dexevt/last.pt
```

- rank 0 的输出在终端和 `logs/<run>/stdout.log`，其他卡在 `logs/<run>/rank<N>.log`；checkpoint 只由 rank 0 写。
- 任何一张卡的进程异常退出，其他卡会被一起结束，去 `rank*.log` 里找报错。
- TensorBoard 只监听服务器的 127.0.0.1，在**本机**执行下面的命令后打开 http://127.0.0.1:16006
  （端口以 `status` 打印的为准）：

```bash
ssh -p 2210 -N -L 16006:127.0.0.1:16006 root@10.108.3.194
```

### 1.4 评估（单卡）

全量数据放不进一张卡，所以评估用 1.1 抽出的 `data/wbc_full_eval`。`--watch` 会持续评估新的 checkpoint，结果追加到
`logs/<run>/eval.csv`；`--every 5` 表示每 5 个 checkpoint（即每 10000 次迭代）评估一次：

```bash
./sm_native.sh run --gpu 7 --robot tiangong3 python supermimic/evaluate.py --run_dir logs/wbc_full_dexevt \
  --motion_dir data/wbc_full_eval --watch --every 5
```

评估会按动作大类汇总失败数和回报，并打印失败率最高的几类。

## 2. 单卡训练（单个动作或小数据集）

```bash
# 单个舞蹈 NPZ，BeyondMimic 随机化
./sm.sh train take001_dexevt --gpu 0 --robot tiangong3 --dynamics dexevt \
  --motion_dir data/tiangong_motion/Take_001_Skeleton0_TienKung3_RobotReady_CollisionFixed_in120_out50_ankle_feet.npz \
  --num_envs 4096 --iterations 150000 --ckpt_every 1000 \
  --dr all --dr_curriculum 6000,12000 --dr_ramp 8000 -d

# 小数据集（服务器上两条行走动作）
./sm_native.sh train wbc_two_v1 --gpu 0 --robot tiangong3 --motion_dir data/wbc_two \
  --num_envs 1024 --buffer_steps 1000 --wm_batch 1024 --pi_batch 1024 --iterations 16000 --ckpt_every 500
```

本机 `./sm.sh` 的训练在前台运行，`Ctrl+C` 保存后退出；`-d` 后台运行，`./sm.sh logs <run>` 查看，`./sm.sh stop <run>` 停止。
`--mem 40g` 改容器内存上限（默认 24g），`CPUS=0-23` 绑定 CPU。

## 3. 评估、录像、sim2sim

```bash
# 评估一个 checkpoint 并保存轨迹，再渲染成 MP4
./sm_native.sh run --gpu 0 --robot tiangong3 python supermimic/evaluate.py --run_dir logs/<run> \
  --motion_dir data/wbc_full_eval --ckpt logs/<run>/ckpt_100000.pt --record logs/<run>/traj.npz
./sm_native.sh run --robot tiangong3 python tools/render.py logs/<run>/traj.npz logs/<run>/out.mp4 \
  --clips <动作名> --seconds 20

# 上面两步的一键版本（服务器上加 LAUNCHER=./sm_native.sh）
LAUNCHER=./sm_native.sh tools/record_mp4.sh logs/<run>/ckpt_100000.pt data/tiangong_motion/motion1.npz

# 在随机参数下比较多个 checkpoint（同一组随机抽样）
./sm_native.sh run --gpu 0 --robot tiangong3 python supermimic/evaluate.py --run_dir logs/<run> \
  --motion_dir data/wbc_full_eval --ckpt logs/a/ckpt_100000.pt logs/b/ckpt_100000.pt --dr all --repeats 16 --seed 0

# CPU MuJoCo sim2sim（脚踝力矩等跟 checkpoint 的 --dynamics 走）
./sm_native.sh run --robot tiangong3 python tools/sim2sim_mujoco.py --device cpu --motion_dir data/wbc_two \
  --ckpt logs/<run>/ckpt_100000.pt --delay_ms 10 --joint_friction 0.3 --push --self_collision
```

## 4. 所有脚本

### 启动器（仓库根目录）

| 脚本 | 作用 |
|---|---|
| `sm_native.sh` | 服务器启动器（无 Docker，conda 环境 `/media/roy/conda_envs/supermimic`，可用 `SM_PYTHON` 改）。`train <run> [--gpu N\|N,M,...\|all] [--robot R] [--tb_port P\|--no_tb] [-d] [train.py 参数]`、`status`、`logs <run>`、`attach <run>`、`stop <run>\|--all`、`tb [run...]`、`run [--gpu N] [--robot R] <命令>`。每个训练跑在 tmux 会话 `sm-<run>` 里，断开 SSH 不影响 |
| `sm.sh` | 本机 Docker 启动器，单卡。`train <run> [--gpu N] [--robot R] [--mem 24g] [--tb_port P\|--no_tb] [-d] ...`、`status`、`logs`、`stop`、`tb`、`run`、`cleanup`（删除旧的常驻容器） |
| `run_train.sh` | 旧入口，等同 `./sm.sh train` |
| `run_tiangong3.sh` | 旧的 Tiangong3 balance 单动作训练快捷方式（固定参数，`./sm.sh`） |

### 训练与评估（`supermimic/`）

| 脚本 | 作用 |
|---|---|
| `train.py` | 训练主程序，由 `sm*.sh train` 调用；`python supermimic/train.py --help` 看全部参数 |
| `evaluate.py` | 评估 checkpoint：`--ckpt` 指定一个或多个，`--watch` 持续评估新 checkpoint，`--record` 保存轨迹，`--dr all --repeats N` 在随机参数下评估 |

### 工具（`tools/`）

| 脚本 | 作用 |
|---|---|
| `record_mp4.sh [ckpt] [motion] [out.mp4] [秒数]` | 评估 + 渲染一键录 MP4；服务器上加 `LAUNCHER=./sm_native.sh` |
| `render.py <traj.npz> <out.mp4>` | 把 `evaluate.py --record` 的轨迹渲染成 MP4（`--clips`、`--seconds`、`--start`、`--global_ghost`） |
| `sim2sim_mujoco.py` | CPU MuJoCo 里跑策略做 sim2sim（`--delay_ms`、`--joint_friction`、`--kp_scale`、`--kd_scale`、`--push`、`--self_collision`、`--dynamics`） |
| `analyze_failure.py <traj.npz> <动作名>` | 分析录制轨迹里某个动作因哪个终止条件失败、误差多大 |
| `plot.py <logs/run>` | 画评估回报、失败数、训练回报随时间的曲线（`--out`） |
| `csv_to_tensorboard.py --run_dir logs/<run>` | 把训练 CSV 同步到 TensorBoard（`--once` 只同步一次） |
| `prepare_motion_dir.py --src <目录> --out <目录>` | 批量校验 Tiangong3 NPZ，把合格的软链接到训练目录（`--workers` 并行数） |
| `make_eval_subset.py --src <目录> --out <目录> --per_group N` | 每个动作大类按固定种子抽 N 条做评估集 |
| `prepare_tiangong3.py` | Tiangong3 URDF 转 MJCF/USD、校验单个 NPZ（`--motion_file`，服务器上加 `--skip_usd`） |
| `setup.sh` | 本机：拉取 Isaac Lab 并构建 Docker 镜像 |
| `download_data.sh` | 下载 G1 的 LAFAN1 动作和 G1 模型 |
| `urdf_to_mjcf.py`、`mjcf_to_usd.py` | G1 模型转换（`download_data.sh` 会调用） |
| `test_env.py` | 检查正运动学、状态读写、参考回放时的奖励 |
| `test_motion_layout.py` | 检查动作关节重排和可微正运动学 |
| `test_reset.py` | 检查 reset 后状态立即生效 |
| `test_spawn.py` | 生成机器人并测仿真吞吐 |
| `bench.py` | 不开仿真，单测世界模型和策略更新的速度 |

## 5. 其他文档

| 文档 | 内容 |
|---|---|
| `DEPLOY_SERVER.md` | H800 服务器环境、实测速度和显存、环境重建 |
| `DEPLOY_TIANGONG3.md` | Tiangong3 动作核验、DR 各项的实现和验收 |
| `DEPLOY_LOCAL.md` | 本机 Docker 镜像构建与验证记录 |
| `README.md` | 论文复现说明（G1 / LAFAN1） |
