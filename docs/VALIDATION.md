# 部署验收记录（2026-10-28）

最新本地命名环境及依赖冲突处理见 [LOCAL_ENVIRONMENT.md](LOCAL_ENVIRONMENT.md)，原始记录在 `logs/conda_migration_20261111/`。下文保留目录重构时的历史验收范围。

基线：`2f2f05fc7a33992b24757a2edb3bf572cd38c0df`。验证 GPU：RTX 5090，Python 3.12，Torch 2.11.0+cu130。

## 验证范围

- 国内 Conda 镜像成功创建 Python 3.12、tmux、ffmpeg、C++ 编译器环境。
- Python 依赖清单成功解析为 235 个固定版本。因大文件下载较慢，后续复用本机已验证运行时的同版本依赖引导 Conda 环境；**没有完成所有 wheel 从零重新下载的端到端验收**。
- `scripts/setup/setup.sh` 在该 Conda 环境中完成依赖审计、随库源码安装及 CUDA/资源检查。
- Python、Isaac Lab、Newton 的实际加载路径均确认来自 Conda 及仓库内源码；后续仿真和训练在宿主机原生执行。
- 独立 clone 后，158 个数据文件 SHA-256 全部一致；两个固定源码包均可校验并解压。
- G1、Tiangong3 的 MJCF 和 USD（含 mujoco variant/payload）及地面 USD 加载通过。
- TensorBoard 版本检查通过（2.21.0）。

## 原生仿真

```bash
./sm.sh run --gpu 1 --robot tiangong3 python tests/integration/test_env.py --num_envs 16 --motion_dir data/tiangong_motion/motion1.npz
```

退出码 0。状态往返最大误差为 0；FK 位置最大误差约 `1.72e-5 m`；参考刚体位置与 FK 的最大差异为 0。
修正了原测试脚本对短动作越界索引的问题，训练逻辑保持基线行为。

## 原生训练

```bash
./sm.sh train conda_deploy_smoke --gpu 1 --robot tiangong3 \
  --motion_dir data/tiangong_motion/motion1.npz --num_envs 16 --buffer_steps 64 \
  --wm_batch 32 --pi_batch 32 --wm_batches 1 --iterations 30 --wm_only_iters 10 \
  --ckpt_every 10 --log_every 10 --horizon 8 --critic_epochs 1 --no_tb -d
```

30/30 次迭代完成，耗时约 98 秒（含 torch.compile），自动退出；写出第 10、20、30 次迭代 checkpoint 和 `last.pt`。
最终 checkpoint 已用 CPU 重新加载，迭代号为 30。训练日志全部数值有限，最后一行：

| 指标 | 数值 |
|---|---:|
| wm_loss | 3.10877 |
| pi_loss | -0.539191 |
| v_loss | 0.0225983 |
| pi_gradnorm | 0.362497 |
| sim_nonfinite | 0 |

这是安装及训练链路验收，不是策略收敛评测。测试生成的日志、缓存和 checkpoint 未上传 Git。
上述 `--gpu 1` 是验证机器的空闲卡；只有一张 GPU 时使用 `--gpu 0`。
