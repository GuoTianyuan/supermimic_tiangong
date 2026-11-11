2026-11-11: read-only WM channel diagnostics on local bc_100hz
Base HEAD: 8fc67db2a5fb7dd325f0814a23387b64f3989554, atop local WM/sampling fixes.
No loss weights, rollout, optimizer, reset sampling, network or prediction mode changed here.
Published with the bc_100hz fixes; not deployed to a running server by this change.

Supervision layout: root position 3 (target heading frame), root orientation 3
(SO(3) log rotation vector), root linear velocity 3 and angular velocity 3
(target body frame), joint position 29, joint velocity 29 = 70 scalar errors.
Both state_delta and velocity_integrated modes retain this same supervision.
Network output dimension is respectively 70 or 35; that is distinct from loss size.

Scale per channel = max(RunningNorm std of real one-step state increments, floor).
Floors: p .003 m, orientation .005 rad, v .02 m/s, w .05 rad/s,
        q .003 rad, qd .05 rad/s.
No separate learned contact-force/reward/termination-head loss is part of this WM loss.

For every training minibatch, collect detached sufficient statistics:
  normalized SSE per channel, raw SSE per channel, scale * valid count,
  floor-active * valid count, valid count and B*H slot count.
No second rollout/FK or per-minibatch device-to-host copy. Accumulate float64 on
GPU and all-reduce once per logging interval, before transferring summaries.
Statistics measure pre-optimizer-update fitting error across ALL WM minibatches
in the interval and all ranks. They are not held-out error or gradient attribution.

New group CSV columns and TensorBoard tags:
  2_world_model/fit_all_batches/loss_per_step
  2_world_model/fit_all_batches/valid_fraction
  2_world_model/groups/<group>/loss_contribution
  2_world_model/groups/<group>/loss_share
  2_world_model/groups/<group>/normalized_mse
  2_world_model/groups/<group>/raw_rmse
  2_world_model/groups/<group>/scale_mean
  2_world_model/groups/<group>/floor_fraction

Group contribution = group normalized SSE / (total B*H slots * 70).
Group shares = group normalized SSE / all normalized SSE; shares sum to 1.
Group normalized MSE = group normalized SSE / (valid count * group width).
Raw RMSE = sqrt(group raw SSE / (valid count * group width)), in physical units.
This is a component RMS, not vector-norm RMS. Rotation-vector component RMS
is distinct from an Euler-angle error or full orientation angle RMS.
Scale mean is averaged over valid samples and group channels, for diagnostics
only: training still uses each channel's own scale. Floor fraction is also
valid-sample/channel weighted. Zero-valid windows have zero error statistics;
check valid_fraction before interpreting them.

The six contributions sum to fit_all_batches/loss_per_step. This retains the
training mask and B*H denominator, not a conditional valid-only loss. Existing
loss_per_step still averages the last WM minibatch from each iteration; it need
not equal the new all-batch metric. Existing CSV columns stay in place and new
columns are appended. On resume, old rows have NaN for unavailable new columns.
Existing historical TensorBoard observations are not reconstructed or rewritten.

TensorBoard additionally records all 70 named channels under:
  2_world_model/channels/<group>/<axis_or_actual_joint_name>/
with normalized_mse, raw_rmse, loss_share, scale, floor_fraction.
Root axes x/y/z mean local translation or rotation-vector components, not Euler.
No per-channel terminal flood: the terminal prints six group rows with loss,
share, component RMS, mean scale and floor fraction.

Interpretation:
- At equal normalized per-component error, each 29-D joint group contributes
  29/70 = 41.43%, each 3-D root group 3/70 = 4.29%.
- Large share alone does not establish a useless term or excessive weighting.
  Compare per-component normalized MSE, raw physical error and scales.
- Loss shares do not directly measure shares of gradients on shared parameters.
- New diagnostics appear only when training executes this updated code. Existing
  server processes and old checkpoints cannot supply historical channel errors.

Official comparison (local HybridRobotics/supermimic at 5efbf114):
GenCoord for 29 joints also totals 70, but uses root pos 3 + rotation matrix 9
(scaled by sqrt(1/2)) + joint pos 29 + joint vel 29. No separately scored root
linear/angular velocity blocks. Matrix entries do not represent nine rotation
DOFs. MaxCoord is a different representation and need not have 70 components.
Official default fixed tolerances: .01 m root pos, .02 rad root orientation,
.1 rad joint pos, 3.14 rad/s joint velocity. The weighted loss is averaged over
scalar channels, so larger blocks still count for more. This is source evidence,
not a claim that the unavailable mjlab_research paper run was reproduced.

Verification:
  tests/unit/test_wm_loss_diagnostics.py
  tests/unit/test_wm_prediction_and_history.py
  tests/unit/test_sampling_and_critic_metrics.py
See validation.json for CPU checks, exact loss/gradient/Adam comparisons in both
WM modes, masks, named channels, reductions and TensorBoard event round trips.
Actual CUDA performance and multi-rank simulator execution were not tested here.
