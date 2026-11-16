"""Full WBC baseline: edit the task and algorithm settings together here.

Defaults describe four ranks (launch separately with torchrun --nproc_per_node=4).
No GPU allocation, training start or server changes happen by loading this file.
"""
from supermimic.tasks.tracking.tracking_env_cfg import TrackingEnvCfg, randomization_ranges
from supermimic.tasks.tracking.agents.supermimic_cfg import SuperMimicRunnerCfg


def full_wbc():
    env = TrackingEnvCfg()
    env.scene.robot = "tiangong3"
    env.scene.num_envs = 4096
    env.scene.dynamics = "dexevt"

    env.commands.motion_dir = "/media/supermimic/data/full_wbc_validated"
    env.commands.motion_active_ratio = 0.30
    env.commands.motion_replace_ratio = 0.20
    env.commands.motion_refresh_iters = 1500
    # None preserves automatic inheritance on resume; fresh runs use 1, 6, ..., 46.
    env.commands.goal_start = None
    env.commands.goal_stride = None
    env.commands.goal_frames = 10
    env.observations.hist = 10
    env.actions.action_noise = 0.1

    # Concrete values (including base friction / reset noise) are declared in tracking_env_cfg.py.
    env.events.ranges = randomization_ranges("tiangong3", "dexevt")
    env.events.dr = "all,contact"
    env.events.dr_curriculum = "3000,6000"
    env.events.dr_ramp = 2000
    env.rewards.action_rate.weight = -0.4
    env.rewards.joint_acc.weight = -3e-6
    env.rewards.joint_limit.weight = -2.0
    env.rewards.root_pos_z_weight = 0.1
    env.rewards.body_pos_z_mode = "relative_root"
    # One nominal sole-center point per foot, in ankle_roll-link coordinates [m].
    # Replace the two ankle points within the existing 14-point reward, without an extra reward term.
    env.rewards.body_point_offsets = {
        "ankle_roll_l_link": (0.045, 0.0, -0.058),
        "ankle_roll_r_link": (0.045, 0.0, -0.058),
    }

    runner = SuperMimicRunnerCfg()
    runner.world_model.wm_window = 12
    runner.world_model.wm_batch = 4096
    runner.actor.horizon = 48
    runner.actor.pi_batch = 4096
    runner.actor.gamma = 0.995
    runner.optimizer.lr = 2e-4
    runner.ckpt_every = 2500
    return env, runner
