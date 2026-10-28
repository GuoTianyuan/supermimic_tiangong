# 国内服务器 Conda 部署

本次验证范围和原生训练结果见 [VALIDATION.md](VALIDATION.md)，其中说明了依赖复用及全新下载验证的限制。

代码基线为 `2f2f05fc7a33992b24757a2edb3bf572cd38c0df`，只追加部署相关修改。
无需 Docker，也不需要另外寻找机器人 USD、mesh 或示例动作。

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
git clone -b mimic git@git.x-humanoid-cloud.com:silence143/tg_supermimic.git
cd tg_supermimic
bash tools/setup.sh
```

默认创建仓库内 `.conda` 环境，`./sm.sh` 与 `./sm_native.sh` 都自动使用它，无需手动激活。
tmux、ffmpeg、C++ 编译器通过 Conda 安装。Isaac Lab 和 Newton 从仓库内的固定源码压缩包安装，
不访问 GitHub；地面和机器人资源都使用相对路径，从本地加载。

Conda 和 PyPI 默认清华镜像，使用固定版 uv 并行安装 Python 包（环境仍为 Conda）。NVIDIA 专用包仍可能需要 `pypi.nvidia.com`，清华缺包时也会访问官方 PyPI；
这不是完全离线安装。可以配置可访问的内网镜像或代理，例如：

```bash
PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple bash tools/setup.sh
# 自定义环境位置，后续启动时也要设置 SM_PYTHON
SM_CONDA_PREFIX=/your/env/path bash tools/setup.sh
export SM_PYTHON=/your/env/path/bin/python
```

依赖版本来自已验证的训练运行时，使用 `requirements-native.lock` 和 `--no-deps` 固定安装。
上游 importer 元数据与 USD/Newton 版本存在冲突，因此 `pip check` 可能报告已知元数据冲突，
不要用 `pip install -U` 自动“修复”导致 MuJoCo、Torch 或 USD 被降级。
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
./sm.sh run python tools/check_install.py
./sm.sh run --robot tiangong3 python tools/test_env.py --num_envs 16 --motion_dir data/tiangong_motion/motion1.npz
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
更多命令见 [HOW_TO_USE.md](HOW_TO_USE.md)；其中 `sm.sh` 在本分支已统一为 Conda 启动器。
旧 Docker 入口另存为 `sm_docker.sh`、`tools/setup_docker.sh`，默认流程不调用它们。

## 自带数据

仓库包含 G1 的完整 LAFAN1 示例、G1 与 Tiangong3 的 URDF/mesh/MJCF/USD（含 payload）、
Tiangong3 的 motion1、balance 和 50 Hz Take_001 动作及 metadata。大规模外部动作库和训练权重不包含在内。
换用自己的动作时运行 `tools/prepare_motion_dir.py` 或 `tools/prepare_tiangong3.py` 校验，不要直接复制
指向其他机器目录的软链接。各数据的来源和许可见 [DATA_ASSETS.md](DATA_ASSETS.md)。
