2026-11-11 full WBC / TG 100 Hz pool training

Authoritative settings: config.json. This experiment evolves control-100hz-pool from a8266ac.
Old pool settings are recoverable at a8266ac; do not use the historical SONIC launch script
on the updated branch as though it still had the old reward defaults.

User-approved final configuration:
- Full WBC from /media_li/wbc_full; 306948 original files, read-only.
  TG original validation writes links/metadata to /media/supermimic/data/full_wbc_validated.
- 4 GPUs on a800_1, 4096 environments per GPU = 16384 global.
- Candidate WM batch and Actor batch: 4096 per GPU = 16384 global, capacity-tested before launch.
- Pool active 30% of clips, replace 20% of the entire rank catalog every 1500 learner iterations.
  Four disjoint rank catalogs together cover the full dataset. Ratios refer to clip counts,
  not frames, bytes, or a fraction of the already active 30%.
- Control/replay/WM transition 10 ms; 4 physical substeps of 2.5 ms (400 Hz).
- Action-rate weight -0.4; DexEVT joint-acceleration weight -3e-6; gamma .995.
- WM training window 12 (120 ms), Actor imagination 48 (480 ms), history/future 10 each.
- Initial learning rate .0002, gradients averaged across ranks before clipping; no extra /4.
- UTD .4, 10 WM minibatches/update, 5 critic epochs x 4 minibatches, lambda .95, tau .05.
- Full DR all,contact, original DexEVT ranges and curriculum 3000/6000 with 2000-step ramps.
- 320000 learner iterations; checkpoint every 2500, metrics every 50, seed0, bf16.
- PD, action scale, robot asset, network structure, normalization algorithm and analytic sigmoid
  imagination termination remain TG's. No training starts from old checkpoint weights.

Storage required to preserve 30/20 on full WBC:
Root/joint/goal/index tables for resident clips remain GPU tensors. Precomputed body position,
orientation, linear/angular velocity remain float32 CPU caches, per clip. Only queried batches
are pinned and copied to GPU. Reward queries transfer all H x B imagined frames together,
not a transfer for every imagined step. Critic transfers positions only. No float16 references,
no altered interpolation, no FK reconstruction during learning, no sampling exclusions.
Per-clip body caching avoids a duplicate contiguous full CPU body table. Old motions remain
resident while referenced by an environment or any valid replay row. Rotation remaps frame and
bin indices together without clearing replay or cutting episodes. 30% is not a memory cap.

Validation and launch:
- tools/validate_full_wbc.py: original TG FK/layout validator on every original file.
- tools/prepare_pool_catalog.py: four CPU full-catalog action-statistic caches and capacity estimates.
- tools/test_motion_pool.py --compile: circular replay, repeated refresh, exact targets,
  goals, rewards, critic features and gradients; active-only resets; checkpoint restoration.
- tools/verify_smooth_reward.py and tools/verify_prediction_windows.py: reward math and WM12 masks.
- tools/run_pool_training.py --smoke: actual simulator/WM/Actor/Critic updates and refresh.
- tools/stress_pool_rank.py: full rank shard after a 30/20 refresh with EVERY old active clip
  pinned, then real full-batch learning. This deliberately overestimates first-refresh residency.
- tools/wait_for_pool_gpus.py: require four GPUs with no compute processes, memory<1GiB,
  utilization<5% for three checks; use the same GPU locks as physics-frequency experiments.
- tools/pool_distributed_preflight.py: once four cards are available, test small four-rank
  training, pool rotation, checkpoint and resume before fresh full production training.
  Does not stop existing jobs and does not silently reduce batch sizes or active ratios.

Manual launch after validations and with four free GPUs:
  python3 tools/run_pool_training.py --gpus 0,1,2,3
Use actual allocated indices; this example does not reserve or free cards.

Interpretation:
This is not a one-variable loader ablation versus the SONIC dense WM12/Actor48 group:
full WBC, pool availability, and total environment/batch size differ. The main training
hyperparameters match. Normalize comparisons by samples and physical time, not only iterations.
The pure control/physics frequency experiments remain on separate branches and original rewards.

Logs:
New pool logs normalize per-step reward by MAX_STEP_REWARD=.05 at100Hz. Earlier dense100Hz logs
used a fixed .1, so multiply those old sim_rew/imag_rew values by2 to compare the same scale.
The CSV legacy column wm_err8 now means the configured twelfth step; terminal/TensorBoard label12.
Pool logs additionally include tracking component scores, root/body RMSE, action delta RMS,
joint acceleration RMS, clean-end/failure counts, active/resident/legacy sizes and coverage.
Motion-start coverage does not prove full-frame traversal or full-motion success.
