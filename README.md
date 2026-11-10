# Tg SuperMimic

基于 `2f2f05fc7a33992b24757a2edb3bf572cd38c0df` 的可部署版本。
默认使用 **Conda**，包含机器人 USD/URDF/MJCF/mesh、示例训练动作，以及固定版本 Isaac Lab/Newton 源码。

`control-100hz` 分支只调整控制频率及相关时间换算：控制、replay 和 WM 每步均为
10 ms；原 motion loader 按源 FPS 插值到 100 Hz，采样 bin 仍为 1 s。
保留每控制步 4 个物理子步，因此子步为 2.5 ms；delay DR 的 0–4 个子步对应 0–10 ms。
原 `control-100hz` 分支中 PD、动作缩放、URDF、奖励、归一化、网络和训练参数保持原样。
当前 `control-100hz-pool` 分支已另行加入下面记录的 full WBC 平滑训练配置。
原 `control-100hz` 的历史和未来参考各 10 帧、想象 32 步；当前 pool 版本改为 WM12／Actor48。
回合上限仍为 500 步（现在是 5 s）。现有 50 Hz 权重不会自动变成 100 Hz 权重。

```bash
git clone -b mimic git@git.x-humanoid-cloud.com:silence143/tg_supermimic.git
cd tg_supermimic
bash tools/setup.sh
./sm.sh run --robot tiangong3 python tools/test_env.py --num_envs 16 --motion_dir data/tiangong_motion/motion1.npz
./run_tiangong3.sh tiangong3_balance -d
```

需要 Linux x86_64、Conda 和兼容 CUDA 13 的 NVIDIA 驱动。
国内镜像配置、完整安装说明和小规模训练验收见 [DEPLOY_CONDA.md](DEPLOY_CONDA.md)。

- [操作手册](HOW_TO_USE.md)
- [随库资源及许可](DATA_ASSETS.md)
- [Conda 仿真与训练验收记录](VALIDATION.md)
- [原始项目说明与结果](README_UPSTREAM.md)

训练实现保留指定提交的行为；部署改动为启动器、依赖安装和本地地面资源路径。
项目代码采用原 [MIT License](LICENSE)，第三方源码和数据遵循各自许可。

### Rotating motion pool (control-100hz-pool)

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

### Full WBC pool experiment (2026-11-11)

Current preset: 100 Hz control / 400 Hz physics, action-rate -0.4, joint-acc -3e-6, gamma .995,
WM12 / Actor48; four GPUs, requested4096 environments and4096 WM/Actor batch per GPU.
Active30%, replacement20% of the full catalog, every1500 learner iterations.
Exact parameters, validation scope and comparison caveats: [experiment record](experiments/full_wbc_pool_20261111/README.txt).
Historical SONIC pool preset is at commit `a8266ac`; do not mix its launch arguments with the new reward defaults.
