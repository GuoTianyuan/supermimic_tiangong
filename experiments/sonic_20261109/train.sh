#!/usr/bin/env bash
# Run in an existing terminal/tmux window; uses the original native launcher.
set -euo pipefail
cd "$(dirname "$0")/../.."
RUN=${1:-tg_sonic_100hz_s0_20261109}
GPUS=${2:-1,2}
mkdir -p "logs/$RUN"
exec bash sm_native.sh _train "$RUN" --gpu "$GPUS" --robot tiangong3 --no_tb \
 --motion_dir /media/supermimic/data/sonic_validated \
 --num_envs 2048 --iterations 320000 --buffer_steps 1000 --utd 0.4 --horizon 32 \
 --goal_frames 10 --hist 10 --wm_window 8 --wm_batches 10 --wm_batch 2048 --pi_batch 2048 \
 --lr 0.0002 --gamma 0.99 --lam 0.95 --action_noise 0.1 \
 --critic_epochs 5 --critic_minibatches 4 --critic_tau 0.05 \
 --critic_layer_norm --act_soft_clip --act_norm ref --reset_a_prev pose \
 --reset_temp 25 --warmup_steps 64 --wm_only_iters 500 --lr_decay_iters 250000 \
 --ckpt_every 2500 --log_every 50 --seed 0 --amp bf16 \
 --dr all,contact --dr_curriculum 3000,6000 --dr_ramp 2000 --dynamics dexevt
