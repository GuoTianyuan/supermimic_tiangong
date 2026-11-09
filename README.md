# Tg SuperMimic

基于 `2f2f05fc7a33992b24757a2edb3bf572cd38c0df` 的可部署版本。
默认使用 **Conda**，包含机器人 USD/URDF/MJCF/mesh、示例训练动作，以及固定版本 Isaac Lab/Newton 源码。

`control-100hz` 分支只调整控制频率及相关时间换算：控制、replay 和 WM 每步均为
10 ms；原 motion loader 按源 FPS 插值到 100 Hz，采样 bin 仍为 1 s。
保留每控制步 4 个物理子步，因此子步为 2.5 ms；delay DR 的 0–4 个子步对应 0–10 ms。
PD、动作缩放、URDF、奖励公式及权重、归一化、网络和训练参数保持原样。
历史和未来参考仍各 10 帧，想象仍默认 32 步；对应时长随控制周期减半。
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
