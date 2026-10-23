# Tiangong3 / balance NPZ

## 环境

- 容器 `supermimic-tiangong3` 映射宿主 GPU **0**，容器内为 `cuda:0`。
- 复用 `supermimic:latest`；代码及数据通过项目目录挂载，不需要重建镜像。
- 原 G1 容器 `supermimic` 保持 GPU 1；已有 Conda 环境不参与部署。
- 使用 29-DoF `data/tiangong3_urdf/urdf/tiangong3.urdf`，MJCF/USD 生成到独立目录。

创建容器（首次）：

```bash
bash docker/run_container.sh 0 24g supermimic-tiangong3 tiangong3
```

已有容器重启：`docker start supermimic-tiangong3`。
24 GiB 为内存上限，不是预留；同时运行 G1 时注意宿主机内存占用。

## 动作和数值核验

指定动作：
`data/tiangong_motion/dex_evt_balance__balance_phzs5_chr01__c2475856d1.npz`

- 2877 帧，50 Hz，约 57.52 秒的参考时间跨度。
- 原始 NPZ 无名称字段；经 FK 核验，关节顺序为 URDF 树的广度优先顺序，四元数为 wxyz，根刚体索引为 0。
- 加载时将关节重排为仿真树序，保留 NPZ 提供的关节与根部速度。
- 工具在 NPZ 旁生成 `.metadata.json`，包含验证结果、名称顺序及 SHA256；动作更改后需要重新生成。
- 41 个采样帧中，所提供 URDF 与 NPZ 的最大位置误差约 3.00 mm，平均约 0.376 mm；运动刚体最大旋转误差约 0.00127 rad。
- 固定 `radar_head_link` 的坐标系相差 90°，它不参与训练跟踪，参考身体状态由当前模型 FK 生成。
- 少量帧超出当前 URDF 硬限位：waist_roll 最大 0.0329 rad；左肘 pitch 0.0273；右肩 roll 0.0691；右肘 pitch 0.0552。保留原始动作，不修改 URDF 限位；reset 沿用原训练器的关节限位裁剪。

资产生成和一致性测试：

```bash
docker exec supermimic-tiangong3 python tools/prepare_tiangong3.py \
  --motion_file data/tiangong_motion/dex_evt_balance__balance_phzs5_chr01__c2475856d1.npz
docker exec supermimic-tiangong3 python tools/test_motion_layout.py \
  --motion_file data/tiangong_motion/dex_evt_balance__balance_phzs5_chr01__c2475856d1.npz
docker exec supermimic-tiangong3 python tools/test_env.py --num_envs 16
```

执行器：URDF effort/velocity 限位，PD/armature 基线采用本机
`/home/eai/Dev/jlab_jerry/Tiangong3-mjlab/src/assets/robots/tiangong3/tk3_constants.py`
中的值，已独立写入 `supermimic/tiangong3.py`，没有运行时外部路径依赖。
动作缩放沿用 SuperMimic 的 `0.25 * effort / Kp`。
此配置使用隐式 PD，不包含其他训练框架的延迟、扭矩-转速曲线或完整摩擦模型；自碰撞仍关闭。

## 训练

```bash
./run_tiangong3.sh tiangong3_balance
tail -f logs/tiangong3_balance/stdout.log
```

启动脚本默认 1024 环境、buffer 1000、WM batch 1024、policy batch 1024、16000 次。
可在末尾覆盖参数，例如 `./run_tiangong3.sh tiangong3_small --num_envs 256`。
每个 run 名称只启动一次，重复训练请使用新名称。

## 评估与视频

```bash
docker exec supermimic-tiangong3 python supermimic/evaluate.py \
  --run_dir logs/tiangong3_balance \
  --motion_dir data/tiangong_motion/dex_evt_balance__balance_phzs5_chr01__c2475856d1.npz \
  --ckpt logs/tiangong3_balance/ckpt_002000.pt \
  --record logs/tiangong3_balance/traj.npz

docker exec -e MUJOCO_GL=egl supermimic-tiangong3 python tools/render.py \
  logs/tiangong3_balance/traj.npz logs/tiangong3_balance/balance.mp4 --seconds 10
```

默认每 2000 次保存 checkpoint。评估 `norm_return` 越高、`n_failed` 越接近 0/1 越好；该单动作结果不能与 G1 的 40 个 LAFAN1 动作直接比较。

## 本次验证结果与运行状态

- Tiangong3 16 环境仿真完成，FK/仿真位置最大误差约 1.82e-5 m。
- Tiangong3 NPZ 重排、原始速度保留、可微 FK 与 MuJoCo 一致性测试通过；G1 CSV 回归检查通过。
- `tiangong3_smoke` 600 次训练完成，世界模型/策略/critic 均更新，日志数值全部有限。
- 短训练评估和 EGL 视频导出完成；此时 norm_return 约 0.0139、失败 1/1，仅说明链路可运行。
- `tiangong3_balance` 已在 GPU 0 上启动 1024 环境、16000 次训练。
- TensorBoard 已验证地址：http://127.0.0.1:16007 ，独立容器 `supermimic-tiangong3-tensorboard`。
- 验证视频：`logs/tiangong3_smoke/balance.mp4`。

查看进程：

```bash
docker exec supermimic-tiangong3 tmux list-sessions
docker exec -it supermimic-tiangong3 tmux attach -t tiangong3_balance
```

使用 `Ctrl+B` 然后 `D` 离开 tmux，训练继续。

交付前已确认持续训练至少完成 950 次迭代，策略/critic 已更新，日志数值均有限。训练继续后台运行。
