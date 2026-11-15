Relative-height reward ablation, 2026-11-16
Branch bc_100hz_relz from Adam baseline bc_100hz_dev 1c1f8e6.
Full WBC 306948 clips; a800_1 GPU4,5; start from scratch, seed0.
Each GPU: env4096, WMbatch4096, Actorbatch4096; global8192 each.
100Hz control/400Hz physics, WM12/Actor48, gamma.995, action noise.1.
Pool active15%, replace10% of each full rank catalog every1500 iterations.
Root position error: ex^2+ey^2+0.1*ez^2; reward weight0.5 and std0.3 unchanged.
This is a squared-error coefficient, NOT multiplying the entire root reward by0.1.
Body target XYZ is anchored to robot root plus yaw-aligned relative reference positions.
Body positions still track relative height and articulation; global reference Z offset no longer enters this term.
World-frame body linear/angular velocity tracking unchanged, including vertical velocity.
All observation/critic inputs, motion data, WM state loss and optimizer stay unchanged.
Root height termination0.25m, soft width0.02m, tilt threshold and width unchanged.
Absolute root Z remains weakly rewarded AND still enters failure termination.
Action-rate-.4, joint-acc-3e-6, joint-limit-2; full DR/contact, DexEVT PD/scale, self collision unchanged.
Future slots [1,6,11,16,21,26,31,36,41,46], hist10, 320000 learning iterations.
The existing Adam control changed reference timing at63242; this new run has t+1 from initialization.
Thus comparison to that run is exploratory, not a perfectly matched fresh-seed paired experiment.
Logs retain unweighted root-Z/XYZ errors and add baseline-compatible body-position and relative-body-Z errors.
Training and imagination share the reward implementation; evaluation/play restore saved reward settings.
Legacy checkpoints retain reference-Z mode and unit Z coefficient. Ordinary resume rejects reward-mode changes.
No claim of better foot contact until evaluation; data height offsets may not be the only cause.
