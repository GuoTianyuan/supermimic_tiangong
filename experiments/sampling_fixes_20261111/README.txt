2026-11-11: local sampling correctness and critic diagnostic fixes on bc_100hz

Scope: this follows the history-noise fix and optional WM parameterization work.
No server synchronization/restart, new training run, deployment export or future
reference/history default changes are performed by this change.

Reset sampling:
- Multiple finished environments with the same start bin are aggregated first.
  Each touched bin gets one update: G_bin = .8*G_bin + .2*mean(batch returns).
  Untouched bins do not change. This keeps TG's episode-start-bin EMA design;
  it does not replace it with the official 20-step visited-bin ReturnSampler.
- A tail bin containing only the clip's last frame has no successor transition.
  Exclude it from eligible reset bins, keeping global/resident bin IDs stable.
  Both dense and rotating-pool loaders use this filter.

Imagination replay starts:
- Same uniform replay rejection sampling and condition f+horizon <= clip_end.
- Bound the fast rejection sampler to 8 attempts. If still short, enumerate valid
  replay cells in chunks of 65536 and fill uniformly with replacement.
- Convert selected physical ring-buffer rows to logical age before reconstructing
  history. Keep the episode/buffer boundary padding and noise_source mapping.
- If the entire replay has no valid start, raise an explanatory RuntimeError;
  never loop indefinitely or silently shorten the horizon/clamp the reference.
- The fast path retains seeded results. Very short motions/clip ends remain
  excluded as Actor imagination starts; that algorithmic choice is unchanged.

Critic diagnostics only (optimizer/target equations unchanged):
- Keep the positional CSV column v_loss for compatibility. Its value remains the
  legacy last-minibatch survival-weighted MSE averaged at log time.
- Its new TensorBoard tag is 4_critic/legacy_last_minibatch_mse; historical event
  files and their old value_loss series are not rewritten.
- Append fit_weighted_huber, fit_weighted_mse, fit_survival_mean to CSV and TB.
  Pool raw weighted error numerators and survival mass over all minibatches,
  epochs, updates in the log window, and ranks. These are pre-update fitting
  errors, not a post-fit evaluation or an unweighted average of batch losses.
- Terminal displays 'Critic fit Huber / MSE' using the new aggregate metrics.
- Statistics do not retain computation graphs or affect gradients/optimization.

Tests:
- tests/unit/test_sampling_and_critic_metrics.py exercises actual env bookkeeping and
  reset methods via AST, duplicate/unique/empty updates, dense/pool tail bins,
  pre-wrap and wrapped replay, exact fallback distribution, all-invalid failure,
  weighted metrics, and actual critic fit loops before/after (20 minibatches).
- CPU regression confirmed bitwise-equal critic parameter updates, plus matching
  diagnostic totals. Cross-rank numerator/mass addition is tested numerically;
  no real distributed job or full simulator/CUDA training was launched.
- Both WM modes additionally ran train.py's actual 12-step rollout, masked loss,
  backward and Adam step on CPU: 70-output state_delta and 35-output integrated,
  both using the full 70-channel state error. Output/statistics are adapted, not
  merely the inference equation.

Unchanged experiment settings:
100Hz, history10, future t+1...t+10, WM12/Actor48, gamma=.995, lambda=.95,
full DR, existing rewards/PD/URDF, 30% active pool / 20% catalog replacement.
A future t+5...t+50 (50-500ms) window remains a proposed independent ablation;
training/reference construction and export/runtime must be changed together.
