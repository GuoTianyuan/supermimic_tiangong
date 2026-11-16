足部跟踪点消融，2026-11-16

A：旧第3组续训
  run=tg_full_wbc_100hz_relz01_adam_s0_20261116
  a800_1 GPU4,5；代码 ed3ba12，目录 /root/tg_supermimic_runs/relative-height。
  从 iteration=33717 的 last.pt 恢复网络、Adam、目标critic、归一化及motion池状态。
  replay与物理环境按既有resume流程重建，不是完整仿真状态的逐位续接。
  原14个link原点；pelvis Z平方误差系数0.1；link相对pelvis高度。
  初始last.pt备份：/media/supermimic/artifacts/sole_ablation_20261116/before_resume/relz_033717_last.pt
  SHA256=74973fdae90168521a9c505eee9ec2b1fafbc3ba495da4c744c1eaa074faac24

B：当前开发主线从零训练
  run=tg_full_wbc_100hz_relz_sole090_adam_s0_20261116
  a800_1 GPU0,1；分支bc_100hz_dev；使用本目录config.json及tiangong3_full_wbc preset。
  仍14点，只把左右ankle原点替换为局部(.09,0,-.058)m的足底前部点。
  位置与线速度同时换算：p'=p+Rr，v'=v+omega×Rr；实际与参考均处理。
  保留脚姿态和角速度跟踪；无额外脚部reward。该点不是最前端脚尖。
  观测、参考输入、网络维度、FK缓存及critic特征不改。

共同配置
  Full WBC 306948条；2卡，各4096环境/WM batch/Actor batch，全局各8192。
  100Hz控制/400Hz物理；WM12/Actor48；goal_offsets=[1,6,11,16,21,26,31,36,41,46]。
  Adam、lr=2e-4、gamma=.995、action-rate=-.4、joint-acc=-3e-6。
  全量DR+contact，自碰撞开；原TG URDF、PD和action scale。
  活跃15%／每1500轮替换目录10%，显式覆盖4卡preset的30%/20%。
  seed0，320000迭代计划；每2500保存、每50日志；新组前500轮WM warm-up。

对比
  保留旧33717模型作为锚点，新组用32500/35000附近checkpoint比较。
  旧组延长后改善支持“继续训练有效”；新组同龄改善支持“跟踪点有效”。
  单种子实验、不同代码组织及旧组resume重建replay不保证逐位一致。
  这两项试验不能单独证明motion Z错误；应保持motion8、初态、物理与播放速度一致，
  结合脚姿态、足底高度/接触、跟踪误差与回报分项判断，不只看总回报。

验证与启动
  preflight.py：两rank学习/换池/保存续训，以及4096环境/batch的编译更新。
  正式启动：python experiments/full_wbc_pool_20261111/scripts/run_pool_training.py --gpus 0,1 --config experiments/sole_point_ablation_20261116/config.json
  预检与实测状态保存在/media/supermimic/artifacts/sole_point_ablation_20261116。
  续训备份保存在/media/supermimic/artifacts/sole_ablation_20261116。
