"""Display names and units. Pure Python, usable for archived CSVs without Isaac Lab."""
import math

REWARD_NAMES = {
    "reward_body_pos": "body_position", "reward_body_ori": "body_orientation",
    "reward_body_linvel": "body_linear_velocity", "reward_body_angvel": "body_angular_velocity",
    "reward_root_pos": "root_position", "reward_root_ori": "root_orientation",
    "pen_action_rate": "action_rate", "pen_joint_acc": "joint_acceleration", "pen_joint_limit": "joint_limit",
}
EPISODE_FIELDS = [*("episode_" + key for key in REWARD_NAMES),
                  "episode_return", "episode_length_seconds", "episode_count"]


def dashboard_scalars(row, control_dt, wm_window, max_reward_rate=5.0):
    """Reward/* = signed weighted contribution per second (reward has dt already).

    Episode_Reward/* = mean completed-episode sum / maximum episode seconds,
    matching Isaac Lab RewardManager. Never reconstruct it from step means.
    """
    if not math.isfinite(control_dt) or control_dt <= 0:
        raise ValueError("control_dt must be positive and finite")
    result = {}
    def add(tag, key, scale=1.):
        value = float(row.get(key, float("nan")))
        if math.isfinite(value):
            result[tag] = value * scale
    for key, name in REWARD_NAMES.items():
        add("Reward/" + name, key, 1 / control_dt)
        add("Episode_Reward/" + name, "episode_" + key)
    # Old runs have a maximum positive tracking rate of 5 per second.
    add("Reward/total", "sim_reward_normalized", max_reward_rate)
    add("Reward/tracking", "track_fraction", max_reward_rate)
    for tag, key in {
        "Train/reward_normalized": "sim_reward_normalized",
        "Train/tracking_fraction": "track_fraction",
        "Train/completed_return_normalized": "ep_ret_norm",
        "Train/episode_length_steps": "ep_len",
        "Train/failure_fraction": "term_frac",
        "Train/episodes_ended": "episodes_ended",
        "Train/ended_clip": "ended_clip", "Train/ended_time_limit": "ended_time",
        "Train/episode_return": "episode_return",
        "Train/episode_length_seconds": "episode_length_seconds",
        "Loss/world_model": "wm_fit_loss_per_step",
        "Loss/actor": "pi_loss", "Loss/critic_huber": "fit_weighted_huber",
        "World_Model/online_error_1_step": "wm_nse1",
        f"World_Model/replay_error_{wm_window}_steps": "wm_err8",
        "World_Model/valid_fraction": "wm_fit_valid_fraction",
        "Imagination/reward_normalized": "imag_reward_normalized",
        "Imagination/survival": "imag_survival",
        "Perf/iterations_per_second": "it_per_s", "Perf/sim_steps": "sim_steps",
        "Perf/wall_hours": "wall_h", "Optimizer/learning_rate": "lr",
        "Optimizer/actor_grad_norm": "pi_gradnorm",
        "Domain_Randomization/range_fraction": "dr_scale",
        "Metrics/sim_nonfinite": "sim_nonfinite",
        "Metrics/action_delta_rms": "action_delta_rms",
        "Metrics/joint_acceleration_rms_rad_s2": "joint_acc_rms",
        "Metrics/smoothness_penalty_over_tracking": "smooth_pen_frac",
    }.items():
        add(tag, key)
    for key in row:
        if key.startswith("error_"):
            add("Metrics/tracking/" + key.removeprefix("error_"), key)
        elif key.startswith("pool_"):
            add("Motion_Pool/" + key.removeprefix("pool_"), key)
    return result


def custom_layout():
    return {"Reward contributions (per second)": {
        "Tracking": ["Multiline", ["Reward/" + name for key, name in REWARD_NAMES.items() if key.startswith("reward_")]],
        "Penalties": ["Multiline", ["Reward/" + name for key, name in REWARD_NAMES.items() if key.startswith("pen_")]],
        "Net vs tracking": ["Multiline", ["Reward/total", "Reward/tracking"]],
    }}
