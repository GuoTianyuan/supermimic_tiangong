# 本机部署记录

项目目录：`/home/eai/Dev/supermimic/supermimic_vibed`

本机：Ubuntu 22.04、两张 RTX 5090、约 60 GiB RAM。运行容器使用 GPU 1。
已有 Conda 环境不参与部署，不修改其依赖。

## 网络与构建

Docker 服务使用 HTTP 代理 `http://127.0.0.1:17891`，请保持代理软件运行。
IsaacLab 固定提交：`46d6cc3d7b115906fad12efd372fbf69f4e3d3ed`。
其本地 `docker/Dockerfile.kitless` 已调整为 HTTP 超时 600 秒、4 个并发下载、Git HTTP/1.1 和 BuildKit uv 缓存。

在项目目录重建：

```bash
(cd IsaacLab && docker build --network=host \
  --build-arg HTTP_PROXY=http://127.0.0.1:17891 \
  --build-arg HTTPS_PROXY=http://127.0.0.1:17891 \
  -f docker/Dockerfile.kitless -t isaac-lab-kitless:main-46d6cc3 .)
docker build --network=host \
  --build-arg HTTP_PROXY=http://127.0.0.1:17891 \
  --build-arg HTTPS_PROXY=http://127.0.0.1:17891 \
  -t supermimic:latest docker/
```

这里必须配合 `--network=host`，构建步骤才能访问宿主机的回环代理。

## 运行

```bash
bash docker/run_container.sh 1 40g
```

容器已存在时，使用 `docker start supermimic`，不要重复创建。
运行中需保留足够宿主内存；40 GiB 是上限，不是预留。

## 资产转换与检查

```bash
docker exec supermimic python tools/urdf_to_mjcf.py
docker exec supermimic python tools/mjcf_to_usd.py
docker exec supermimic python tools/test_env.py --num_envs 64
```

## 训练

先小规模验证：

```bash
bash run_train.sh smoke_5090 --num_envs 256 --buffer_steps 128 \
  --wm_batch 256 --pi_batch 256 --iterations 600 --ckpt_every 200 --log_every 20
```

正式训练候选配置（需根据实际资源调整）：

```bash
bash run_train.sh lafan1_5090 --num_envs 1024 --buffer_steps 1000 \
  --wm_batch 1024 --pi_batch 1024 --iterations 16000
tail -f logs/lafan1_5090/stdout.log
```

## 已完成的验证

- 两个镜像已构建：`isaac-lab-kitless:main-46d6cc3`、`supermimic:latest`。
- `supermimic` 容器已运行，映射宿主 GPU 1，40 GiB RAM 上限。
- PyTorch 2.11.0+cu130 已完成 RTX 5090 CUDA 张量运算。
- 40 个动作、38 个网格和 URDF 已下载，MJCF/USD 已生成。
- 64 环境仿真检查正常结束，FK 位置最大误差约 6.45e-7 m。
- `logs/smoke_5090` 的 600 次训练完成，世界模型、策略、critic 更新及 checkpoint 保存正常，日志数值均有限。
- checkpoint 评估正常完成；该短训练策略仍有 40/40 失败，尚不具备稳定跟踪能力。
- 无窗口 EGL 视频导出正常：`logs/smoke_5090/dance.mp4`。
- 正式 16000 次训练尚未启动，1024 环境配置尚未做资源峰值验证。

测试日志：`logs/deploy_test_env.log`、`logs/smoke_5090/stdout.log`、`logs/deploy_eval.log`、`logs/deploy_render.log`。

## 评估与渲染

```bash
docker exec supermimic python supermimic/evaluate.py \
  --run_dir logs/lafan1_5090 --ckpt logs/lafan1_5090/ckpt_016000.pt \
  --record logs/lafan1_5090/traj.npz
docker exec -e MUJOCO_GL=egl supermimic python tools/render.py \
  logs/lafan1_5090/traj.npz logs/lafan1_5090/dance.mp4 \
  --clips dance2_subject1 --seconds 10
```

容器以 root 运行，新生成文件可能归 root 所有。需要从宿主编辑时可运行：

```bash
docker exec supermimic chown -R 1000:1000 /workspace/supermimic/logs
```

