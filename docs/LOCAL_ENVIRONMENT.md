# 本地固定环境验收（2026-11-11）

开发分支：`bc_100hz_dev`。环境：`/home/mig/tg_supermimic/.conda`。
本次从 Conda/Python 包源完成安装，Isaac Lab 和 Newton 从仓库内固定源码包 editable 安装。
原 `/home/mig/anaconda3/envs/supermimic` 环境未修改；服务器训练未操作。

## 使用

```bash
cd /home/mig/tg_supermimic
conda activate /home/mig/tg_supermimic/.conda
python scripts/training/train.py --help
```

也可直接使用 `./sm.sh`，启动器默认选择仓库内 `.conda/bin/python`。
激活新环境时会清空继承的 `PYTHONPATH` 并设置 `PYTHONNOUSERSITE=1`，隔离本机全局 ROS/Python 3.10 路径。
Conda 提示覆盖 PYTHONPATH 是这个隔离配置的预期提示，退出环境后恢复原变量。
复现该隔离设置：

```bash
conda env config vars set -p /home/mig/tg_supermimic/.conda PYTHONPATH= PYTHONNOUSERSITE=1
```

Python 3.12.16；Torch 2.11.0；MuJoCo/MuJoCo-Warp 3.15.0；Warp 1.17.0。
Isaac Lab 源码 commit `46d6cc3`，Newton commit `71eeba1`，全部 235 个锁定包版本匹配。
激活后已核实项目、Isaac Lab、Newton 均从本仓库对应源码目录加载。

## 实际验证

使用本机 GPU0 RTX 4090，NVIDIA 驱动 580.126.09。初次在受限执行环境中无法访问 GPU，
切换到宿主机权限后确认驱动正常；本次没有修改驱动。

- 新环境中 33 项单元/回归测试全部通过。
- 158 项资源校验通过，G1/Tiangong MJCF、USD 和地面资源以及 CUDA 运算检查通过。
- Tiangong 16 环境实际运行 150 个控制步；状态往返、FK 和模拟器对齐检查完成。
  FK 位置最大误差约 1.68e-5 m。该检查是参考目标开环执行，不是策略收敛评测。
- 16 环境、WM/Actor batch 32，WM12/Actor48、100 Hz 控制/400 Hz 物理、dexevt、
  全 DR 和自碰撞运行 30 次学习迭代，包含 WM、Actor、Critic 更新。
  使用完整 last.pt 恢复到第 35 次；WM/Actor/Critic 优化器步数分别为 35/30/30。
- checkpoint 权重和优化器状态有限，110 列 CSV 和 459 个 TensorBoard scalar tag 数值有限，
  仿真非有限值计数为 0。
- GPU 活跃池连续换池 12 次；引用保留、参考/目标/奖励/梯度、统计和状态恢复检查通过，
  含 10 次编译查询。使用 12 条合成短 motion，未做 full WBC 容量测试。
- 现有 full WBC 20,000 步 checkpoint 导出 ONNX，通过模型检查；ONNX CPU ReferenceEvaluator
  的 8 组输入输出与 PyTorch 最大绝对差 2.03e-6。此项不是 ONNX Runtime 后端测试。
- 修复 test_env.py 仍写死 0.1 的显示分母，改用 MAX_STEP_REWARD，并复跑通过。
  只改验收打印，训练奖励和参数没有变化。

## 依赖约束与范围

`pip check` 返回 1，有 8 条锁定图中的约束冲突：ovphysx/packaging、
isaacsim-asset-isolated 对 MuJoCo/USD 的约束、isaacsim-robot-schema/USD、
mujoco-usd-converter/MuJoCo、numba/NumPy、cmeel-boost/NumPy。
已保留锁定图，没有自动升级或降级；本记录只证明上述实际运行路径通过，
不代表所有可选转换工具或第三方功能均兼容。详细清单在本地 pip_check.txt。

未验证本次安装的多 GPU/NCCL 训练，也未验证全库容量、长训收敛和真机表现。
环境约 14 GiB，源码快照约 83 MiB。测试进程均已退出，没有启动正式长训。

原始记录：`logs/local_environment_20261111/`，含 install.log、environment.json、pip_freeze.txt、
pip_check.txt、pytest.xml、test_env_corrected.log、train_smoke.log、train_resume.log、
motion_pool.json、export_check.json、validation.json 及小规模 checkpoint。
这些日志、环境和 checkpoint 不提交 Git。
