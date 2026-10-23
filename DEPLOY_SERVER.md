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
