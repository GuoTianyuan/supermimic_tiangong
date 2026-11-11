Full WBC development-branch comparison, 2026-11-12 (Asia/Shanghai)

Requested host: a800_2, GPUs 0,1,2,3. Existing repository /root/tg_supermimic;
branch bc_100hz_dev. Reuse /media assets, environment, dataset and run storage.
Run: tg_full_wbc_100hz_dev_s0_20261112. Start from scratch; no --resume.

Matched to a800_1: full WBC 306948 clips; 4 ranks, each 4096 environments,
WM batch4096 and Actor batch4096. 100Hz control, 400Hz physics; WM12/Actor48;
action-rate -.4, joint acceleration -3e-6, gamma .995, lr 2e-4, seed0.
TG DexEVT PD/URDF/action scale, normalization, state_delta WM, analytic sigmoid,
full DR including contact; curriculum 3000/6000, ramp2000. Pool30%, replace20%
of each rank catalog every1500 iterations. 320000 iterations, save2500, log50.

Self-collision enabled from the first step. Normal adjacency/URDF filters remain.
a800_1 comparison resumed at32037 from a model previously trained without
self-collisions. It also predates the corrected duplicate-bin update and padded
history noise. Thus this is a development-baseline comparison, not a pure
source-layout ablation; compare equal total iterations/samples and fixed-motion
evaluations, with initialization history explicitly noted.

One shared launcher is reused:
python experiments/full_wbc_pool_20261111/scripts/run_pool_training.py \
  --config experiments/full_wbc_dev_20261112/config.json --gpus 0,1,2,3

Only --config support and host-specific cache paths were added to the launcher.
No training/math defaults changed for this launch. Four-rank update/refresh/save/
resume preflight must pass before the production run. Actual runtime status and
manifests are stored under /media/supermimic/artifacts/full_wbc_dev_20261112 and
/media/supermimic/logs/colleague/tg_full_wbc_100hz_dev_s0_20261112.
