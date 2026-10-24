# H800 服务器部署

- 地址：`ssh -p 2210 root@10.108.3.194`，项目：`/media/roy/supermimic_vibed`。
- 服务器是 K8s pod：没有 Docker，podman 无 GPU 支持，所以不用容器，改用 conda 环境
  `/media/roy/conda_envs/supermimic`（Python 3.12，包版本与本地 `supermimic:latest` 镜像一致）。
- `/media` 为持久存储；pod 重启后环境、代码、日志都在。
- 服务器访问不到 GitHub 和 Omniverse S3：`newton` 从本地镜像打包安装；地面资产从 `.cache/remote_assets`
  复制到 `/tmp/https`（`sm_native.sh` 每次启动自动完成）。

## 动作数据

`/media/wbc_partial` 的 npz 与天工 3 balance 动作格式相同（29 关节、39 刚体、50 fps）。
训练用两条拷到 `data/wbc_two/`（原目录不写入）：

- `BG_Normal_Walking_00001.npz`：848 帧，17.0 s
- `DB_Basic_Gait_00001.npz`：823 帧，16.5 s

已用 `tools/prepare_tiangong3.py` 生成 `.metadata.json`：最大位置误差 3.0 mm，无关节超限。换其他动作：

```bash
cp "/media/wbc_partial/<name>.npz" data/wbc_two/
./sm_native.sh run --robot tiangong3 python tools/prepare_tiangong3.py --skip_usd --motion_file "data/wbc_two/<name>.npz"
```

全量 `/media/wbc_partial`（90,269 条，9206 万帧，约 511 h）已整理到 `data/wbc_partial/`：每条是指向原文件的软链接
加 `.metadata.json`，全部通过校验（原目录不写入）。数据有更新时重跑即可，已校验的会跳过（约 5 分钟）：

```bash
./sm_native.sh run python tools/prepare_motion_dir.py --src /media/wbc_partial --out data/wbc_partial
```

## 多卡训练（全量数据）

```bash
cd /media/roy/supermimic_vibed
./sm_native.sh train wbc_full_v1 --gpu all --robot tiangong3 --motion_dir data/wbc_partial \
  --num_envs 16384 --iterations 320000 --ckpt_every 2000
```

- `--gpu all` 或 `--gpu 0,1,2,3`：每张卡一个进程，共同训练同一个策略。动作文件按文件名轮流分到各卡
  （8 卡时每卡约 1150 万帧），梯度和归一化统计在各卡之间平均，所以策略能用上全部数据。
- `--num_envs`、`--wm_batch`、`--pi_batch` 都是**每张卡**的值；8 卡 × 16384 = 131,072 个环境。
- 8 × H800 实测（全量数据）：

  | 每卡环境数 | 每卡显存 | 策略阶段速度 | 仿真步/s |
  |---|---|---|---|
  | 8192 | 41 GB | 2.0 it/s | 33 万 |
  | 16384 | 50 GB | 1.9 it/s | 63 万 |

  32 万次迭代约 47 小时。启动时每卡加载动作约 4.5 分钟，之后编译约 1 分钟。
- 推荐每卡 `--num_envs 8192 --buffer_steps 1000`：速度和 16384 环境几乎一样，buffer 显存和 16384 × 500 相同，
  但保留的历史长一倍，策略一时退化时 buffer 不会很快被摔倒数据占满。
- 进度块和 TensorBoard 的 `4_critic/horizon_value_*` 记录想象终点的 critic 估值 V(s_H)：均值、最大值，以及超过
  最大可能回报 10（0.1/(1−γ)）的比例。比例持续上升说明策略在利用 critic 的高估，通常先于训练崩溃出现；
  `policy_grad_norm_max` 是每个日志窗口内的梯度范数最大值。
- `3_policy_imagination/action_loop_gain_*`（进度块 “Action loop gain”）是策略对观测里上一动作的雅可比
  ‖∂a_t/∂a_{t−1}‖₂（最大奇异值，256 个 buffer 状态的中位数和最大值）。长期大于 1 表示想象中的动作回路会自我放大，
  BPTT 梯度随步数增长；这时可以 `--resume` 并加 `--detach_a_prev`（想象中不对这条观测回传梯度）。
- `1_tracking/dr_range_scale_mean` 是 `--dr` 各项当前范围比例的平均值（课程渐变的进度）。
- `1_tracking/diverged_sim_states`（CSV `sim_nonfinite`，非 0 时进度块里有 “Diverged sim states” 一行）是窗口内
  物理状态发散（NaN/inf 或超过 1e3）的次数，这些环境直接结束回合并 reset，不会再让训练崩溃。偶尔个位数正常；
  如果持续增多，多半是某个随机化范围让 MuJoCo-Warp 不稳定。
- rank 0 的输出在终端和 `logs/<run>/stdout.log`，其余在 `logs/<run>/rank<N>.log`；checkpoint 只由 rank 0 写。
- `stop`、`Ctrl+C`、`--resume` 与单卡相同，所有卡在同一迭代停下。任一进程异常退出时，其余进程会被一起结束。
- 评估仍是单卡，`--motion_dir` 用小目录（如 `data/wbc_two`）；全量 9206 万帧放不进一张卡。

## 单卡训练

```bash
cd /media/roy/supermimic_vibed
./sm_native.sh train wbc_two_v1 --gpu 0 --robot tiangong3 --motion_dir data/wbc_two \
  --num_envs 1024 --buffer_steps 1000 --wm_batch 1024 --pi_batch 1024 --iterations 16000 --ckpt_every 500
```

- 训练在 tmux 会话 `sm-<run>` 中运行，并直接显示在当前终端；断开 SSH 不影响训练。
- `Ctrl+C`：跑完当前迭代、保存 checkpoint 后退出；`Ctrl+B` 再按 `D`：离开，训练继续。
- `-d`：后台启动，不进入。

```bash
./sm_native.sh status               # 训练、TensorBoard 端口、各 GPU 进程、进度
./sm_native.sh logs wbc_two_v1      # 跟随输出（Ctrl+C 只退出查看）
./sm_native.sh attach wbc_two_v1    # 回到训练终端
./sm_native.sh stop wbc_two_v1      # 优雅停止；--all 停止全部
./sm_native.sh train wbc_two_v1 --gpu 0 --robot tiangong3 --motion_dir data/wbc_two ... --resume logs/wbc_two_v1/last.pt
```

TensorBoard 只监听服务器的 127.0.0.1（端口在启动时打印，从 16006 起），本机通过 SSH 转发查看：

```bash
ssh -p 2210 -N -L 16006:127.0.0.1:16006 root@10.108.3.194   # 然后打开 http://127.0.0.1:16006
```

## 评估

```bash
./sm_native.sh run --gpu 0 --robot tiangong3 python supermimic/evaluate.py --run_dir logs/wbc_two_v1 \
  --motion_dir data/wbc_two --ckpt logs/wbc_two_v1/ckpt_016000.pt --record logs/wbc_two_v1/traj.npz
```

全量训练用 `data/wbc_eval`（从 `data/wbc_partial` 固定种子抽的 256 条，1.2 h）评估；`--watch` 会持续评估新的
checkpoint，结果写入 `logs/<run>/eval.csv`（`--every` 按每 2000 次迭代一个 checkpoint 计数）。
评估会按动作大类（文件名里 `__` 前的部分，没有 `__` 时去掉末尾编号）汇总失败数和回报，打印失败率最高的几类，
完整结果在 `ckpt_*_eval.json` 的 `per_group` 里。`wbc_full` 的文件名带大类（walk、run、crawl、lie_down 等 14 类），
可以按类分层抽一个评估集：

```bash
./sm_native.sh run python tools/make_eval_subset.py --src data/wbc_full --out data/wbc_full_eval --per_group 32
./sm_native.sh run --gpu 7 --robot tiangong3 python supermimic/evaluate.py --run_dir logs/wbc_full_dr_v2 \
  --motion_dir data/wbc_full_eval --watch --every 5
```

评估会占一张卡，和 8 卡训练同时跑时用训练没占的卡，或者训练用 `--gpu 0,1,2,3,4,5,6`。
如果某些大类（例如 crawl、kneel_down、lie_down、get_up、fuwocheng 等贴地动作）一直失败、又不需要，
训练时可以用 `--motion_exclude '^(crawl|lie_down|get_up|fuwocheng)__'` 不加载它们；
`--reset_temp`（默认 25）调低时，reset 采样不那么集中在回报最低的片段上。
## 带域随机化训练（第 3–7 步）

各项含义、实现和验收方法见 `DEPLOY_TIANGONG3.md` 的“真机部署前的域随机化”。服务器上在训练命令末尾加 `--dr`：

```bash
./sm_native.sh train wbc_two_dr_pd --gpu 0 --robot tiangong3 --motion_dir data/wbc_two \
  --num_envs 1024 --buffer_steps 1000 --wm_batch 1024 --pi_batch 1024 --iterations 16000 --ckpt_every 500 --dr pd
```

- 多卡训练同样加 `--dr`。replay buffer 每 1024 个环境多占约 0.6 GB，每卡 16384 环境时多约 9.6 GB。
  全量数据建议用 `--dr_curriculum` 分阶段打开（先学会跟踪，再加本体参数，最后加推力和延迟）；每个阶段在
  `--dr_ramp`（默认 2000）次迭代内从标称值逐步放宽到完整范围，每 5 % 更新一次并打印当前比例：

```bash
./sm_native.sh train wbc_full_dr_v2 --gpu all --robot tiangong3 --motion_dir data/wbc_full \
  --dr all --dr_curriculum 3000,6000 --num_envs 8192 --buffer_steps 1000 --iterations 320000 --ckpt_every 2000
```
- 不带 `--dr` 的训练与之前完全一样，`wbc_full_v1` 可以照常 `--resume`。
- 基线和 DR 策略在同一组随机参数下评估：

```bash
./sm_native.sh run --gpu 0 --robot tiangong3 python supermimic/evaluate.py --run_dir logs/wbc_two_dr_pd \
  --motion_dir data/wbc_two --ckpt logs/wbc_two_v1/ckpt_016000.pt logs/wbc_two_dr_pd/ckpt_016000.pt \
  --dr all --repeats 16 --seed 0
./sm_native.sh run --robot tiangong3 python tools/sim2sim_mujoco.py --device cpu --motion_dir data/wbc_two \
  --ckpt logs/wbc_two_dr_pd/ckpt_016000.pt --delay_ms 10 --joint_friction 0.3 --push
```

## 已验证

- torch 2.11.0+cu130 识别 8 张 H800；Isaac Lab/Newton 仿真自检：状态往返误差 0，FK 位置误差 1.4e-5 m。
- 256 环境冒烟训练进入策略阶段（第 600 次迭代，2.56 it/s，GPU 显存 2.8 GB），`stop` 后保存 checkpoint，
  tmux 会话、TensorBoard 和 GPU 进程全部退出。
- 首次运行会为 H800 编译 Warp/torch 内核（求解器初始化约 30 s），之后使用 `.cache/` 中的缓存。

## 环境重建（pod 环境丢失时）

依赖清单在服务器的 `.cache/req_nodeps.txt`，`newton` 包在 `.cache/newton.tgz`：

```bash
P=/media/roy/conda_envs/supermimic/bin/python
conda create -y -p /media/roy/conda_envs/supermimic --override-channels \
  -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge python=3.12
$P -m pip install torch==2.11.0 torchvision==0.26.0
$P -m pip install --no-deps -r .cache/req_nodeps.txt newton-usd-schemas==0.5.0 --extra-index-url https://pypi.nvidia.com
tar xzf .cache/newton.tgz -C "$($P -c 'import site; print(site.getsitepackages()[0])')"
(cd IsaacLab/source && $P -m pip install --no-deps --no-build-isolation $(for d in isaaclab isaaclab_assets isaaclab_contrib \
  isaaclab_experimental isaaclab_newton isaaclab_ov isaaclab_physx isaaclab_ppisp isaaclab_rl isaaclab_tasks \
  isaaclab_tasks_experimental isaaclab_visualizers; do echo "-e $d"; done))
```

镜像里的包组合本身有版本冲突（例如 `cmeel-boost` 要求 numpy<2.4），所以除 torch 外都用 `--no-deps` 按精确版本安装。
