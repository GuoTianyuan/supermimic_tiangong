2026-11-11: history padding repair and opt-in WM parameterization ablation
Base: bc_100hz at 8fc67db2a5fb7dd325f0814a23387b64f3989554.

Validated locally before publication on bc_100hz; not deployed to servers and not used to restart existing runs.

1. Replay returns noise_source for observation history. Imagination and action-loop
   diagnostics now share one noise draw across copies of the oldest retained real
   frame, matching real collect/reset padding. Genuine frames keep independent
   noise. With no padding, seeded noise results are unchanged. No change to the
   real rollout noise amplitudes, replay selection or reset sampling weights.

2. Training flag:
     --wm_prediction state_delta          (default, original 70 outputs)
     --wm_prediction velocity_integrated  (new, 35 outputs)
   Old checkpoints without the flag resolve to state_delta. Resume inherits the
   saved choice; changing mode on resume is rejected. Integrated-mode resume also
   checks control/physics timing. Use a new run for this architecture ablation.

   Integrated outputs: delta_v_body[3], delta_w_body[3], delta_qd[29].
   With n physics substeps, c=(n+1)/(2*n). Current dt=.01, n=4, c=.625.
     p1  = p + dt*(v + c*R*delta_v_body)
     R1  = R*Exp(dt*(R^T*w + c*delta_w_body))
     v1  = v + R*delta_v_body
     w1  = w + R*delta_w_body
     q1  = q + dt*(qd + c*delta_qd)
     qd1 = qd + delta_qd
   This constrains velocity/pose integration, not contact dynamics or rigid-body
   equations. Constant acceleration during a control step is an approximation.

3. The integrated ablation keeps TG input normalization, hidden layers, full 70-D
   state loss, time-sum reduction and ERR_FLOOR. Its 35-D output RunningNorm is
   separate from a 70-D state-increment error normalizer. Both are updated from
   the same real transition data, not imagined predictions. For identical real
   data the 70-D loss scales are exactly equal to the legacy mode. This is not
   a wholesale reproduction of the official loss/network/optimizer setup.

4. CPU checks in tools/test_wm_prediction_and_history.py cover strict old weight
   compatibility, bitwise default predictions/statistics, 35/70-D statistics,
   checkpoint round trips, 1/2/4/8-substep integration, root-frame transforms,
   constant-velocity motion, 12/48-step action/network gradients, Dynamo fullgraph
   capture (CPU eager backend), reset/history/buffer-wrap padding and H=1.
   Simulator-only imports are stubbed; actual numerical methods are loaded.
   Full simulator, multi-rank collectives, GPU compilation/capacity and training
   convergence have NOT been tested for the new mode.

Unchanged / remaining discussion:
- hist=10, future reference t+1...t+10, reset sampler and reward unchanged.
- env.py duplicate-bin index_put_ still needs an aggregated update design/fix;
  it was audited separately and is not silently changed with this ablation.
- A longer reference window requires matched export/deployment reference timing;
  in a live stream without future prediction, longer lookahead adds latency.
- Existing actor exports and running server jobs are untouched.
