# 本地 Conda 环境验收（2026-11-11）

开发分支：`bc_100hz_dev`。标准命名环境：`tg_supermimic`，路径
`/home/mig/anaconda3/envs/tg_supermimic`。仓库内原 `.conda` 已在新环境验收后移除。
原 `/home/mig/anaconda3/envs/supermimic` 未修改；本次未操作服务器训练或启动正式长训。

## 使用

```bash
conda activate tg_supermimic
cd /home/mig/tg_supermimic
python scripts/training/train.py --help
python -m pip check
```

`./sm.sh` 自动选择此命名环境，也支持显式 `SM_PYTHON`、`SM_CONDA_PREFIX`、`SM_CONDA_ENV`。
不会因为当前激活了旧 `supermimic` 或 `base` 而切换到另一套仿真库。
环境激活时清空 `PYTHONPATH`、设置 `PYTHONNOUSERSITE=1`，避免全局 ROS/Python 3.10 路径和
用户 site-packages 混入。覆盖 PYTHONPATH 的提示是隔离设置生效；退出环境后 Conda 恢复原变量。
需要额外源码导入时，启动器允许显式设置 `SM_EXTRA_PYTHONPATH`。

## 保留的训练运行时

Python 3.12.16；Torch 2.11.0；MuJoCo/MuJoCo-Warp 3.15.0；Warp 1.17.0；NumPy 2.5.1；
USD Exchange 3.0.1。209 个锁定包逐一核对通过。Isaac Lab 源码 commit `46d6cc3`、
Newton commit `71eeba1`，版本与迁移前一致。

Python 及 wheel 装在 Conda 环境；Isaac Lab、Newton 和本项目通过 editable 安装注册到环境，
源码分别保留在 `IsaacLab/`、`.vendor/newton/`、`src/supermimic/`，便于查阅与开发。
这次不修改训练算法、奖励、PD、动作缩放、DR、URDF 或控制频率。

## 冲突处理

之前 8 条依赖冲突已解决，当前 `pip check` 输出 `No broken requirements found.`。
处理方式是从训练依赖清单排除 26 个当前训练不使用的包，保留核心版本：

| 可选功能 | 冲突或问题 | 当前处理 |
| --- | --- | --- |
| PhysX 后端 `ovphysx` | 要求 packaging <24，而当前为26 | 当前训练使用 Newton/MuJoCo-Warp，不安装 ovphysx 运行时 |
| MJCF/URDF→USD 转换工具 | 锁定 MuJoCo3.12、USD Exchange2.3，与训练3.15/3.0.1冲突 | 训练直接读取已有资产；转换依赖单列 `requirements-assets.txt` |
| Pinocchio/Pink 及 cmeel 依赖 | cmeel-boost 的 NumPy 约束不满足，旧环境 Pinocchio 导入也有二进制符号错误 | TG 自有 FK 不依赖这些 IK 工具，排除整组依赖 |
| Numba/llvmlite | Numba0.64不接受当前NumPy2.5 | 当前训练不使用 Numba，排除两者 |

没有改第三方包元数据来掩盖冲突。转换脚本保留；独立转换环境说明见
[DEPLOY_CONDA.md](DEPLOY_CONDA.md)。其 24 包依赖图已成功解析，本次没有安装转换环境或重建机器人资产。
安装器和 `check_install.py` 现在都会检查依赖冲突，失败即停止。

## 实际验证

本机 GPU0 RTX4090，驱动580.126.09。结果保存在 `logs/conda_migration_20261111/`。

- 38 项单元/回归测试通过，含新增的命名环境选择、显式路径覆盖、错误环境回退和导入路径隔离测试。
- 158 项资源校验、G1/Tiangong MJCF/USD/地面加载、CUDA 运算通过。
- 16 个 Tiangong 环境实际运行150控制步，FK位置最大误差约1.68e-5m。
- 在迁移前后的环境分别创建真实仿真模型，509项 MuJoCo 数组和选项逐项完全一致。
  这检查包含质量、惯量、关节和碰撞参数；仅对应本次 Tiangong 标称模型。
- 16环境、WM/Actor batch32、WM12/Actor48、dexevt、全DR、自碰撞短训30轮；
  保存完整 last.pt 后续训到35轮，WM/Actor/Critic优化器步数分别35/30/30，权重有实际更新。
- 网络和优化器状态、110列CSV、459个TensorBoard标量tag均为有限数值，仿真非有限值计数为0。
- GPU活跃池完成12次换池、10次编译查询；旧引用保留、重映射、目标/奖励/梯度一致性、
  全目录统计和pool状态恢复通过。使用12条合成短motion。
- full WBC 20,000轮checkpoint重新导出ONNX；8组输入与PyTorch输出最大绝对差2.03e-6。
  使用ONNX CPU ReferenceEvaluator，不是ONNX Runtime后端。
- `conda activate tg_supermimic`、Python路径和源码editable入口已核实；移除旧前缀后再次完成自检。

本次验证是单GPU、小批次的功能验收，不包含多GPU通信、全库容量、长训收敛或真机效果验证。

详细文件：`validation.json`、`physics_comparison.json`、`activation_check.json`、`pip_check.txt`、
`pip_freeze.txt`、`pytest.xml`、`train_smoke.log`、`train_resume.log`、`motion_pool.json`、
`export_check.log`。环境、缓存、日志和测试checkpoint不提交Git。
初次安装的历史记录保留于 `logs/local_environment_20261111/`；本文件以迁移后的验收为准。
