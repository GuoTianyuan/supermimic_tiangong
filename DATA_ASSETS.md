# 随库资源与来源

本分支为用户指定的训练项目部署打包。顶层 MIT License 仅覆盖本项目代码，第三方文件保留原有权利。

| 目录 | 内容 / 来源 | 许可或使用范围 |
|---|---|---|
| `vendor/isaaclab-46d6cc3.tar.gz` | Isaac Lab `46d6cc3d7b115906fad12efd372fbf69f4e3d3ed` 的 source、apps、依赖锁和许可证；https://github.com/isaac-sim/IsaacLab | BSD-3-Clause，许可证及第三方通知包含在包内 |
| `vendor/newton-71eeba1.tar.gz` | 已验证运行时的 Newton 纯源码，commit `71eeba16a717d4dcf0b9dd7056a7b6f863054dc3`；https://github.com/newton-physics/newton | Apache-2.0，见 `vendor/NEWTON_LICENSE.md` 及包内 `newton/licenses` |
| `data/lafan1_g1` | 40 条原始 G1 CSV；https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset 的 `g1/` | LAFAN1 / Ubisoft La Forge，CC BY-NC-ND 4.0：https://creativecommons.org/licenses/by-nc-nd/4.0/ ，不得用于商业用途 |
| `data/g1_description`、`data/g1_usd` | 同一数据源的 Unitree G1 描述与 mesh；本项目转换生成 MJCF/USD | Unitree BSD-3-Clause；转换脚本在 `tools/` |
| `data/tiangong3_urdf`、`data/tiangong3_generated`、`data/tiangong3_usd` | 用户本地提供的 Tiangong3 描述与原有转换输出 | 保留原提供方权利，本次按用户要求用于目标训练仓库部署；未声明为 MIT 数据 |
| `data/tiangong_motion` | 用户本地提供的 motion1、balance、Take_001 50 Hz NPZ 及原校验 metadata | 同上；未包含原始 100 Hz 副本和外部大动作库 |
| `data/ground` | 已缓存的 Isaac Lab 地面 USD 及两张纹理，USD 文件保留 BSD-3-Clause 版权头 | Isaac Lab Project Developers |

`assets-manifest.json` 记录所有随库数据的 SHA-256；`vendor/manifest.json` 记录源码包版本和 SHA-256。
所有 NPZ/CSV 保持原文件内容，不上传指向本机其他目录的软链接。USD 包含所有原有 payload 和相对引用。
未包含训练日志、checkpoint、个人 Conda 环境、缓存和机器凭据。
