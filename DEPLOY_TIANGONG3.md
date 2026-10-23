# Tiangong3 / balance NPZ

## 环境

- 所有命令通过 `./sm.sh` 在一次性容器中运行（`supermimic:latest`，项目目录挂载），命令结束即删除容器，
  不再有常驻容器；GPU 用 `--gpu N` 选择，容器内为 `cuda:0`。
- 容器以宿主用户（uid 1000）运行，生成的文件不归 root。
- Omniverse 地面资产缓存在 `.cache/remote_assets`，新容器无需重新下载。
- 使用 29-DoF `data/tiangong3_urdf/urdf/tiangong3.urdf`，MJCF/USD 生成到独立目录。

```bash
./sm.sh status     # 正在运行的容器、端口、内存、各 GPU 占用（含宿主进程）和训练进度
./sm.sh cleanup    # 删除旧的常驻容器 supermimic、supermimic-tiangong3、*-tensorboard*（会先确认）
```

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
./sm.sh run --robot tiangong3 python tools/prepare_tiangong3.py \
  --motion_file data/tiangong_motion/dex_evt_balance__balance_phzs5_chr01__c2475856d1.npz
./sm.sh run --robot tiangong3 python tools/test_motion_layout.py \
  --motion_file data/tiangong_motion/dex_evt_balance__balance_phzs5_chr01__c2475856d1.npz
./sm.sh run --robot tiangong3 python tools/test_env.py --num_envs 16
```

执行器：Kp/Kd、初始姿态和动作缩放对齐本机 DexEVT 配置
`/home/eai/Dev/cheerleading_competition/cheerleading_competition/source/whole_body_tracking/whole_body_tracking/robots/dex_evt.py`，
已独立写入 `supermimic/tiangong3.py`，没有运行时外部路径依赖。
- effort/velocity 限位取 URDF（与 DexEVT 一致）；并联踝每个串联轴上限 100 N·m（2 × 50 N·m 曲柄电机）。
- 动作缩放 `0.5 * effort / Kp`，踝按 55 N·m 计算（与 DexEVT 的 `D3_ACTION_SCALE` 相同）。
- armature 固定取 DexEVT 启动随机化范围的中点：0.031（0.002–0.06），踝 0.035（0.01–0.06）。
此配置使用隐式 PD，不包含其他训练框架的延迟、扭矩-转速曲线或完整摩擦模型；自碰撞仍关闭。

## 训练

```bash
./run_tiangong3.sh tiangong3_v2                          # 前台训练，终端每 50 次迭代打印一次进度块
./run_tiangong3.sh tiangong3_v2 --motion_dir <npz或目录>  # 换动作
./run_tiangong3.sh tiangong3_v2 -d                       # 后台训练，只跟随输出
```

启动脚本默认 GPU 0、内存上限 24g、1024 环境、buffer 1000、WM batch 1024、policy batch 1024、16000 次，
末尾参数可覆盖（例如 `--gpu 1`、`--num_envs 256`）。训练的 TensorBoard 由同一容器提供，
地址为启动时打印的 `http://127.0.0.1:<port>`（从 16006 起找空闲端口），训练结束端口即释放。

- 停止：前台按 `Ctrl+C`，或另开终端 `./sm.sh stop tiangong3_v2`。训练会跑完当前迭代、保存
  `ckpt_*.pt` 和 `last.pt` 后退出并删除容器；再按一次 `Ctrl+C` 则不保存立即退出。
- 恢复：`./run_tiangong3.sh tiangong3_v2 --resume logs/tiangong3_v2/last.pt`。
- 查看后台训练：`./sm.sh logs tiangong3_v2`（`Ctrl+C` 只退出查看）。
- 已有日志的 run 名称不能直接重用，除非带 `--resume`。
- 关闭前台终端不会停止训练，此时用 `./sm.sh status` 查看、`./sm.sh stop` 停止。

## 评估与视频

```bash
./sm.sh run --robot tiangong3 python supermimic/evaluate.py \
  --run_dir logs/tiangong3_v2 \
  --motion_dir data/tiangong_motion/dex_evt_balance__balance_phzs5_chr01__c2475856d1.npz \
  --ckpt logs/tiangong3_v2/ckpt_016000.pt \
  --record logs/tiangong3_v2/traj.npz

./sm.sh run --robot tiangong3 python tools/render.py \
  logs/tiangong3_v2/traj.npz logs/tiangong3_v2/balance.mp4 --seconds 10

./sm.sh tb tiangong3_balance tiangong3_v2   # 对比已有 run 的 TensorBoard，Ctrl+C 关闭
```

默认每 2000 次保存 checkpoint。评估 `norm_return` 越高、`n_failed` 越接近 0/1 越好；该单动作结果不能与 G1 的 40 个 LAFAN1 动作直接比较。

## 本次验证结果与运行状态

- Tiangong3 16 环境仿真完成，FK/仿真位置最大误差约 1.82e-5 m。
- Tiangong3 NPZ 重排、原始速度保留、可微 FK 与 MuJoCo 一致性测试通过；G1 CSV 回归检查通过。
- `tiangong3_smoke` 600 次训练完成，世界模型/策略/critic 均更新，日志数值全部有限。
- 短训练评估和 EGL 视频导出完成；此时 norm_return 约 0.0139、失败 1/1，仅说明链路可运行。
- `tiangong3_balance`（旧 mjlab PD）在 GPU 0 上跑到第 1900 次迭代停止，未保存 checkpoint（当时每 2000 次才保存）。
- 验证视频：`logs/tiangong3_smoke/balance.mp4`。
- `./sm.sh` 已验证：前台/后台训练、TTY 下 `Ctrl+C` 与 `./sm.sh stop` 均保存 checkpoint 后删除容器并释放端口，
  `--resume` 从 `last.pt` 继续，`status` 能识别各 GPU 上的容器和宿主进程。
