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
此配置使用隐式 PD，没有扭矩-转速曲线；执行器延迟、关节摩擦等用 `--dr` 打开（见“真机部署前的域随机化”）；自碰撞仍关闭。

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

## 真机部署前的域随机化（第 3–7 步）

`train.py --dr <项>` 在论文原有的随机化（地面摩擦、关节零偏、躯干质心）之上，再打开 `supermimic/dr.py` 里的几项，
范围取 DexEVT 的 `tracking_env_cfg.py`（mjlab TK3 相同）。不加 `--dr` 时行为与之前一致（已用同种子对比验证），
xi 仍是 33 维，旧 checkpoint 照常恢复。只有 `--robot tiangong3` 能用，G1 传 `--dr` 会直接报错。

| 项 | 随机化内容 | 抽样时机 | 范围 |
|---|---|---|---|
| `pd` | 每个关节的 Kp、Kd 分别缩放 | 每回合 reset | ×0.75–1.25 |
| `push` | 根速度叠加随机增量（世界系） | 每个环境每 1–5 s | x/y ±0.5、z ±0.2 m/s；roll/pitch ±0.52、yaw ±0.78 rad/s |
| `mass` | 30 个刚体段的质量和惯量同比例缩放，固定挂件（IMU、相机、头、TCP）随所挂连杆 | 启动时 | ×0.85–1.15 |
| `armature` | 每个关节 | 启动时 | 0.002–0.06，踝 0.01–0.06 kg·m² |
| `joint_friction` | 每个关节的 MuJoCo frictionloss | 启动时 | 0.001–0.6 N·m |
| `delay` | 每控制步的前 k 个 5 ms 物理子步仍跟踪上一步的目标 | 每回合 reset | k = 0–4（0–20 ms） |

用法：`--dr pd`、`--dr pd,push`、`--dr all`。评估默认仍是标称参数。
`--dr_curriculum A,B` 分阶段打开：第 A 次迭代前只有论文原有的随机化，从 A 起除 push、delay 外的各项开始放宽，
从 B 起 push、delay 开始放宽。每个阶段在 `--dr_ramp`（默认 2000）次迭代内把范围从标称值线性放宽到完整范围，
每 5 % 一档：每项取值为 标称 + s·(抽样 − 标称)，s 为当前比例。启动时项每个环境只抽一次完整范围的样本，
随 s 从标称值移向它，每档在切换时立即重写；pd、delay 从各环境的下一回合生效，推力幅度立即按比例缩放。
扩展 xi 在世界模型和 critic 的输入归一化里固定为均值 0、标准差 1/√3（[−1, 1] 上均匀分布的值），
不用滑动统计：课程开始前这些维是常数，滑动统计会冻结在常数附近，切换后输入会被放大上百倍。

实现要点：

- 随机参数全部写进 xi，世界模型和 critic 能看到，策略看不到。这样想象 rollout 仍是每个环境自己的动力学，
  策略则必须对整个范围鲁棒；不写进 xi 的随机化只会让世界模型学成平均动力学。
- 只要开了任一项，xi 就扩展成固定的 180 维，没开的项填标称值，所以不同组合之间可以 `--resume`
  （新加的项会让世界模型误差先跳高，几十次迭代后回落）；带 `--dr` 和不带的 checkpoint 不能互相恢复。
- 推力发生后，该条转移在 buffer 里记为 `cut`，世界模型的 8 步训练窗口在 `done | cut` 处截断。
- 开 `delay` 时，每个控制步改为调用 4 次 5 ms 的 `sim.step`（总物理步长不变），reset 后第一步的“上一目标”
  取 reset 姿态。只要开了任一项，世界模型就额外输入 `a_prev`。
- 每回合重写增益时只通知 MuJoCo-Warp 更新执行器参数，避免 Isaac Lab 默认触发的全量常数重算
  （1024 环境下每步约多 9 ms）。启动时的质量、armature、摩擦仍走全量重算，只发生一次。
- 开销：1024 环境 `env.step` 从 10.7 ms 增至 12.2 ms，主要来自关节摩擦约束；replay buffer 每条转移多 147 个 float，
  16384 环境 × 1000 步时每卡多约 9.6 GB 显存。

### 每一步的验收

1. 600 次迭代的冒烟测试，与无 DR 的冒烟基线比较 `wm_err8`（看第 500 次，即世界模型预热结束、策略尚未更新时）：

```bash
SMOKE=(--no_tb --num_envs 256 --buffer_steps 128 --wm_batch 256 --pi_batch 256 --iterations 600 --ckpt_every 200 --log_every 20)
./run_tiangong3.sh drsmoke_base "${SMOKE[@]}"
./run_tiangong3.sh drsmoke_pd "${SMOKE[@]}" --dr pd                                       # 第 3 步
./run_tiangong3.sh drsmoke_pd_push "${SMOKE[@]}" --dr pd,push                             # 第 4 步
./run_tiangong3.sh drsmoke_body "${SMOKE[@]}" --dr pd,push,mass,armature,joint_friction   # 第 5 步
./run_tiangong3.sh drsmoke_all "${SMOKE[@]}" --dr all                                     # 第 6 步
```

2. 完整训练一次（例如 `./run_tiangong3.sh tiangong3_dr_pd --dr pd`，服务器命令见 `DEPLOY_SERVER.md`），
   再把第 2 步的基线和 DR 策略放在同一组随机参数下评估（第 7 步）。`--dr` 取 `base`（训练时的摩擦、零偏、
   质心随机化加观测噪声）、某几项或 `all`；每个动作跑 `--repeats` 次，各次参数不同；同一 `--seed` 下所有
   checkpoint 遇到的参数完全相同：

```bash
./sm.sh run --robot tiangong3 python supermimic/evaluate.py --run_dir logs/tiangong3_dr_pd \
  --motion_dir data/tiangong_motion/dex_evt_balance__balance_phzs5_chr01__c2475856d1.npz \
  --ckpt logs/<第2步基线>/ckpt_016000.pt logs/tiangong3_dr_pd/ckpt_016000.pt --dr all --repeats 16 --seed 0
```

   `--watch` 模式下带 `--dr` 的结果写到 `eval_dr-<项>.csv` 和 `ckpt_*_eval_dr-<项>.json`，不覆盖标称评估。

3. sim2sim：`tools/sim2sim_mujoco.py` 在 CPU MuJoCo 里跑同一个策略。生成的 MJCF 加上地面，用 1 kHz 显式 PD
   （按力矩上限裁剪）代替训练时的隐式 PD，接触参数用 MuJoCo 默认值；观测、目标、奖励和终止与训练相同，
   分数可以和 `evaluate.py` 直接比较。可选执行器延迟、关节摩擦、增益误差、推力和自碰撞；不给 `--ckpt`
   时做参考动作的开环回放，用来单独检查物理设置。`--record` 的轨迹可以直接用 `tools/render.py` 渲染。

```bash
./sm.sh run --robot tiangong3 python tools/sim2sim_mujoco.py --device cpu \
  --motion_dir data/tiangong_motion/dex_evt_balance__balance_phzs5_chr01__c2475856d1.npz \
  --ckpt logs/tiangong3_dr_pd/ckpt_016000.pt --delay_ms 10 --joint_friction 0.3 --push --record logs/tiangong3_dr_pd/s2s.npz
```

DexEVT（PhysX）里的 sim2sim 还没做：需要在 DexEVT 环境里按 SuperMimic 的观测重新拼接输入。
它的 50 Hz 配置和这里的动作约定（默认站姿、0.5·τ/Kp 缩放）一致，可以作为下一步。
动作缩放已确认用 0.5·τ/Kp（髋约 0.13、腰 yaw 0.068、膝 0.196 rad/单位），部署端按同一缩放把动作换成关节目标；
本机 `/media/roy/cheerleading_competition` 的 `robots/dex_evt.py` 里固定的 `D3_ACTION_SCALE[n] = 0.25` 不适用于这里的策略。

策略输出：新训练的 checkpoint（`args.act_soft_clip`）在 `Policy.forward` 里做软限幅，|a| ≤ 15 时不变，
之后平滑饱和到 20，部署时直接用网络输出（导出时包含 `forward` 即可），关节目标为 `q0 + α·a`
（仿真里的零偏模拟的是标定误差，真机上没有这一项）。

观测里的上一动作 `a_prev`：新训练的 checkpoint（`args.reset_a_prev == "pose"`）在每回合开始时取保持当前关节目标的动作
`(q_target − q0 − 零偏)/α`，部署时策略启动的第一步也应这样初始化（从默认站姿启动时就是 0）；
更早的 checkpoint 训练时取 0。`evaluate.py` 和 `tools/sim2sim_mujoco.py` 会按 checkpoint 自动选择。

### 冒烟测试结果（本机 RTX 5090，balance 动作，256 环境）

| run | `--dr` | wm_err8 @300 | @500 | @600 |
|---|---|---|---|---|
| `drsmoke_base` | 无 | 0.091 | 0.070 | 0.193 |
| `drsmoke_pd` | `pd` | 0.107 | 0.086 | 0.184 |
| `drsmoke_pd_push` | `pd,push` | 0.111 | 0.084 | 0.226 |
| `drsmoke_body` | 再加 `mass,armature,joint_friction` | 0.129 | 0.094 | 0.179 |
| `drsmoke_all` | `all` | 0.133 | 0.106 | 0.269 |

- 无 DR 时第 500 次为 0.0696，与旧的 `tiangong3_smoke`（0.0698）一致。
- 加推力后误差不升（0.084 对 0.086），说明 `cut` 截断有效。
- 各组误差都随迭代持续下降；DR 越多，同一迭代下误差越高，`all` 在第 500 次比基线高约 50%。
  冒烟只有 256 个环境，启动时抽样的参数也只有 256 组；完整训练的环境更多，需要看完整训练的曲线再判断。
  第 600 次时策略已开始更新，各组波动都较大，不适合直接比较。
- 已逐项核对 512 个 world 的 MuJoCo-Warp 模型参数：kp/kd、armature、frictionloss、质量与写入值一致，
  惯量与质量同比例；延迟 0–4 个子步时，关节在一步内朝新目标移动 0.034/0.023/0.012/0.004/−0.001 rad。
- sim2sim 与训练仿真一致：同一开环回放，两者都在第 59 步倒地，前 40 步根高度相同到 1 mm，
  第 50 步关节均方根差 2.2 mrad。

## 本次验证结果与运行状态

- Tiangong3 16 环境仿真完成，FK/仿真位置最大误差约 1.82e-5 m。
- Tiangong3 NPZ 重排、原始速度保留、可微 FK 与 MuJoCo 一致性测试通过；G1 CSV 回归检查通过。
- `tiangong3_smoke` 600 次训练完成，世界模型/策略/critic 均更新，日志数值全部有限。
- 短训练评估和 EGL 视频导出完成；此时 norm_return 约 0.0139、失败 1/1，仅说明链路可运行。
- `tiangong3_balance`（旧 mjlab PD）在 GPU 0 上跑到第 1900 次迭代停止，未保存 checkpoint（当时每 2000 次才保存）。
- 验证视频：`logs/tiangong3_smoke/balance.mp4`。
- `./sm.sh` 已验证：前台/后台训练、TTY 下 `Ctrl+C` 与 `./sm.sh stop` 均保存 checkpoint 后删除容器并释放端口，
  `--resume` 从 `last.pt` 继续，`status` 能识别各 GPU 上的容器和宿主进程。
