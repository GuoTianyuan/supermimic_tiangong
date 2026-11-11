# 国内服务器 Conda 部署

本次验证范围和原生训练结果见 [docs/VALIDATION.md](VALIDATION.md)，其中说明了依赖复用及全新下载验证的限制。

当前开发分支为 `bc_100hz_dev`；`bc_100hz` 保留整理前的基线。环境继续使用固定源码快照和依赖锁。
无需 Docker，也不需要另外寻找机器人 USD、mesh 或示例动作。

本机 2026-11-11 的从零安装、GPU 仿真和保存续训结果见 [本地验收](LOCAL_ENVIRONMENT.md)。

## 安装方式与版本

当前训练后端是 Isaac Lab kit-less / Newton / MuJoCo-Warp，不需要完整 Isaac Sim。
推荐普通运行时用固定版本 wheel，开发和接口层用固定提交源码的 editable 安装。
`pip install -e` 仍由 pip 注册包，但运行时读取源码目录；不是所有组件都需要从源码编译。

| 组件 | 本分支安装方式 | 固定版本 |
| --- | --- | --- |
| Python | 独立 Conda 环境 | 3.12 |
| Torch / torchvision | pip/uv wheel | 2.11.0 / 0.26.0 |
| MuJoCo / MuJoCo-Warp | pip/uv wheel | 3.15.0 / 3.15.0 |
| Warp | pip/uv wheel | 1.17.0 |
| Isaac Lab | vendor 源码快照，editable | commit 46d6cc3 |
| Newton | vendor 源码快照，editable | commit 71eeba1 |
| TG SuperMimic | 当前仓库，editable | 当前 Git 提交 |

固定源码包及校验值见 `vendor/manifest.json`。安装器解压到 `IsaacLab/` 和
`.vendor/newton/`；这些是源码快照，不是新的 Git clone。不要随意切换成最新 release。
版本号、源码提交和资源路径影响兼容性，安装方式本身并不能保证与服务器一致。

默认创建标准命名 Conda 环境 `tg_supermimic`，安装在 Conda 的 `envs/` 目录，避免覆盖旧的 `supermimic` 环境。配置好 Python 依赖后，
仍需验收 GPU、真实环境步进及一次包含 Actor/Critic 更新的短训。

## 安装

要求 Linux x86_64、支持 CUDA 13 的 NVIDIA 驱动（原环境为 580 系列）、建议至少 24 GB 显存和 30 GB 磁盘。
系统需要 Git、编译工具及 OpenGL/EGL 动态库。Ubuntu 管理员可安装：

```bash
sudo apt-get update
sudo apt-get install -y git build-essential libegl1 libgl1 libopengl0 libglib2.0-0 libgomp1 libxrender1 libvulkan1
```

没有 Conda 时，从国内 Miniforge 镜像下载并安装，随后打开新终端：

```bash
curl -fL --retry 5 https://mirrors.tuna.tsinghua.edu.cn/github-release/conda-forge/miniforge/LatestRelease/Miniforge3-Linux-x86_64.sh -o /tmp/Miniforge3.sh
bash /tmp/Miniforge3.sh
```

```bash
git clone -b bc_100hz_dev git@github.com:wenhan10086/supermimic.git tg_supermimic
cd tg_supermimic
bash scripts/setup/setup.sh
conda activate tg_supermimic
```

默认创建命名环境 `tg_supermimic`。本机路径为 `/home/mig/anaconda3/envs/tg_supermimic`。
`./sm.sh` 与 `./sm_native.sh` 自动选择它，无需手动激活；不会回退到当前激活的其他环境。
tmux、ffmpeg、C++ 编译器通过 Conda 安装。Isaac Lab 和 Newton 从仓库内的固定源码压缩包安装，
不访问 GitHub；地面和机器人资源都使用相对路径，从本地加载。

Conda 和 PyPI 默认清华镜像，使用固定版 uv 并行安装 Python 包（环境仍为 Conda）。NVIDIA 专用包仍可能需要 `pypi.nvidia.com`，清华缺包时也会访问官方 PyPI；
这不是完全离线安装。可以配置可访问的内网镜像或代理，例如：

```bash
PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple bash scripts/setup/setup.sh
# 自定义环境位置，后续启动时也要设置 SM_PYTHON
SM_CONDA_PREFIX=/your/env/path bash scripts/setup/setup.sh
export SM_PYTHON=/your/env/path/bin/python
# 或自定义命名环境（安装和启动均设置）
export SM_CONDA_ENV=my_tg
bash scripts/setup/setup.sh
```

依赖版本来自已验证的训练运行时，使用 `requirements-native.lock` 和 `--no-deps` 固定安装。
安装后必须通过 `pip check`；安装器和自检脚本遇到冲突即失败。训练锁已排除当前后端不使用的
PhysX 运行时、Pinocchio/Pink IK、Numba 和离线资产转换依赖。不要为安装可选工具而升级或降级训练的 MuJoCo、Torch、NumPy 或 USD。

环境激活时清空 `PYTHONPATH` 并设置 `PYTHONNOUSERSITE=1`，避免全局 ROS/Python 3.10 或用户 site-packages 混入。
启动器也执行隔离；确需额外源码路径时显式设置 `SM_EXTRA_PYTHONPATH`。
可使用 `SM_WHEELHOUSE=/path/to/wheels` 安装预下载的 Python wheels（仍需预装 Conda 环境或可访问 Conda 镜像）。
在同平台 Python 3.12 环境中准备 wheelhouse：

```bash
python -m pip download --no-deps -d wheelhouse -r requirements-native.lock torch==2.11.0 torchvision==0.26.0 'pip>=25' wheel uv==0.11.16
```

下载时须配置能获取 NVIDIA 包的 index；wheelhouse 应保存完整依赖清单，且只能用于相同平台。

## 自检和训练

安装脚本自动验证关键依赖、MJCF、USD 和 CUDA。登录节点无 GPU 时可设置 `SM_SKIP_GPU_CHECK=1`，
然后在 GPU 节点上重新运行完整检查。

```bash
./sm.sh run python scripts/setup/check_install.py
./sm.sh run --robot tiangong3 python tests/integration/test_env.py --num_envs 16 --motion_dir data/tiangong_motion/motion1.npz
# 600 次小规模训练：覆盖世界模型预热及策略更新，作为安装后的验收
./sm.sh train conda_smoke --gpu 0 --robot tiangong3 --motion_dir data/tiangong_motion/motion1.npz \
  --num_envs 64 --buffer_steps 128 --wm_batch 64 --pi_batch 64 \
  --iterations 600 --ckpt_every 200 --log_every 20 -d
./sm.sh logs conda_smoke
./sm.sh status
```

正式舞蹈训练：

```bash
./sm.sh train dance_motion1 --gpu 0 --robot tiangong3 --tracking dance --dynamics dexevt \
  --motion_dir data/tiangong_motion/motion1.npz --num_envs 4096 \
  --iterations 150000 --ckpt_every 1000 --dr all,contact --dr_curriculum 6000,12000 --dr_ramp 8000 -d
./sm.sh stop dance_motion1
```

`--gpu all` 支持多卡；每卡的环境数和 batch 参数独立计数。首次运行编译 CUDA 内核较慢。
训练日志、checkpoint 和编译缓存保留在本地，不提交 Git。示例动作足够从头训练，不需要预训练权重。
更多命令见 [docs/HOW_TO_USE.md](HOW_TO_USE.md)；其中 `sm.sh` 在本分支已统一为 Conda 启动器。
旧 Docker 入口另存为 `sm_docker.sh`、`scripts/setup/setup_docker.sh`，默认流程不调用它们。

## 自带数据

仓库包含 G1 的完整 LAFAN1 示例、G1 与 Tiangong3 的 URDF/mesh/MJCF/USD（含 payload）、
Tiangong3 的 motion1、balance 和 50 Hz Take_001 动作及 metadata。大规模外部动作库和训练权重不包含在内。
换用自己的动作时运行 `scripts/data/prepare_motion_dir.py` 或 `scripts/assets/prepare_tiangong3.py` 校验，不要直接复制
指向其他机器目录的软链接。各数据的来源和许可见 [docs/DATA_ASSETS.md](DATA_ASSETS.md)。

## 可选资产转换工具

训练和播放直接加载仓库中已生成的 MJCF/USD，不需要安装 `isaacsim-asset-isolated`。
该转换工具锁定 MuJoCo 3.12、USD Exchange 2.3，与训练的 3.15/3.0.1 不兼容；
脚本仍保留在 `scripts/assets/`，需要重新生成资产时使用单独工具环境：

```bash
conda create -n tg_supermimic_assets python=3.12 pip -y
conda activate tg_supermimic_assets
unset PYTHONPATH
export PYTHONNOUSERSITE=1
python -m pip install --extra-index-url https://pypi.nvidia.com -r requirements-assets.txt
python -m pip check
# 在此环境运行 scripts/assets/ 下的 USD 转换脚本；完成后切回训练环境。
conda activate tg_supermimic
```

`requirements-assets.txt` 是独立的直接依赖清单，不能叠加安装到训练环境。
本次仅核对并解析该工具清单，不自动创建工具环境，不重新生成已有机器人资源。
Pinocchio/Pink、Numba 和 PhysX 后端没有加入本项目的训练依赖；未来启用对应功能时另做兼容验证。
