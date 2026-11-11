"""CSV/TensorBoard schema and terminal display, separate from learning updates."""
import torch
from supermimic.diagnostics import (CRITIC_FIELDS, EXTRA_FIELDS, REWARD_KEYS, WM_FIELDS,
                                   WM_GROUPS, WM_TB_TAGS, WM_UNITS)

# sm_native.sh reads some columns by position: new ones go at the end
CSV_FIELDS = ["iteration", "wall_h", "sim_steps", "wm_loss", "wm_err8", "pi_loss", "v_loss", "imag_rew", "pi_gradnorm",
              "imag_survival", "sim_rew", "ep_len", "ep_ret_norm", "term_frac", "it_per_s", "lr",
              "v_end_mean", "v_end_max", "v_end_over", "pi_gradnorm_max", "a_gain_med", "a_gain_max", "dr_scale", "sim_nonfinite",
              "smooth_pen_frac", "joint_acc", *EXTRA_FIELDS, *CRITIC_FIELDS, *WM_FIELDS]  # fmt: skip
SMOOTH_KEYS = ("pen_smooth", "track", "joint_acc")
# TensorBoard sorts groups by name, so the numeric prefixes put the tracking metrics first.
def build_tb_tags(wm_window):
    TB_TAGS = {
        "sim_rew": "1_tracking/sim_reward_per_step",
        "term_frac": "1_tracking/fall_rate_of_finished_episodes",
        "ep_len": "1_tracking/episode_length_steps",
        "ep_ret_norm": "1_tracking/finished_episode_return_normalized",
        "wm_loss": "2_world_model/loss_per_step",
        "wm_err8": f"2_world_model/error_after_{wm_window}_steps",
        "imag_rew": "3_policy_imagination/reward_per_step",
        "imag_survival": "3_policy_imagination/survival_rate_over_horizon",
        "pi_loss": "3_policy_imagination/policy_loss",
        "pi_gradnorm": "3_policy_imagination/policy_grad_norm",
        "pi_gradnorm_max": "3_policy_imagination/policy_grad_norm_max",
        "a_gain_med": "3_policy_imagination/action_loop_gain_median",
        "a_gain_max": "3_policy_imagination/action_loop_gain_max",
        "dr_scale": "1_tracking/dr_range_scale_mean",
        "sim_nonfinite": "1_tracking/diverged_sim_states",
        "smooth_pen_frac": "1_tracking/smoothness_penalty_over_tracking_reward",
        "joint_acc": "1_tracking/joint_acc_sq_sum_rad2_per_s4",
        "v_loss": "4_critic/legacy_last_minibatch_mse",
        "v_end_mean": "4_critic/horizon_value_mean",
        "v_end_max": "4_critic/horizon_value_max",
        "v_end_over": "4_critic/horizon_value_above_max_return_frac",
        "it_per_s": "5_speed/iterations_per_second",
        "sim_steps": "5_speed/total_sim_steps",
        "wall_h": "5_speed/wall_clock_hours",
        "lr": "6_optimizer/learning_rate",
    }

    for key in EXTRA_FIELDS:
        group = ("7_motion_pool" if key.startswith("pool_") else
                 "8_reward_terms" if key in REWARD_KEYS or key == "track_fraction" else
                 "9_tracking_errors" if key.startswith("error_") else "1_tracking")
        TB_TAGS[key] = f"{group}/{key}"
    TB_TAGS["wm_nse1"] = "2_world_model/error_after_1_step"
    for key in CRITIC_FIELDS:
        TB_TAGS[key] = f"4_critic/{key}"
    TB_TAGS.update(WM_TB_TAGS)

    return TB_TAGS


def _hms(seconds: float) -> str:
    s = int(max(seconds, 0))
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def _rss_gb() -> float:
    with open("/proc/self/status") as f:
        return next(int(line.split()[1]) for line in f if line.startswith("VmRSS")) / 2**20


def print_progress(r: dict, n_done: float, steps_per_s: float, elapsed_s: float, eta_s: float, last_ckpt: str, *, args, g_max: float, control_dt: float,
                   dr_terms, world: int, pooled: bool):
    width, pad = 80, 34
    policy = r["iteration"] > args.wm_only_iters
    title = f"Learning iteration {r['iteration']}/{args.iterations}  ({'world model + policy' if policy else 'world-model warm-up'})"
    free, total = torch.cuda.mem_get_info()
    lines = [(f"WM loss / NSE1 / NSE{args.wm_window}", f"{r['wm_loss']:.4f} / {r['wm_nse1']:.4f} / {r['wm_err8']:.4f}")]
    lines += [("WM all-batch loss / valid", f"{r['wm_fit_loss_per_step']:.4f} / {100*r['wm_fit_valid_fraction']:.1f}%"),
              ("WM groups (all batches/ranks)", "loss | share | component RMS | mean scale | at floor")]
    for group, unit in zip(WM_GROUPS, WM_UNITS):
        prefix = f"wm_{group}"
        lines.append((f"  WM {group}",
                      f"{r[prefix+'_loss_contribution']:.4g} | {100*r[prefix+'_loss_share']:.1f}% | "
                      f"{r[prefix+'_raw_rmse']:.4g} {unit} | {r[prefix+'_scale_mean']:.4g} | "
                      f"{100*r[prefix+'_floor_fraction']:.0f}%"))
    if policy:
        lines += [("Policy loss / grad norm (max)", f"{r['pi_loss']:.3f} / {r['pi_gradnorm']:.3f} ({r['pi_gradnorm_max']:.3f})"),
                  ("Critic fit Huber / MSE", f"{r['fit_weighted_huber']:.4f} / {r['fit_weighted_mse']:.4f}"),
                  ("Critic V(s_H) mean / max", f"{r['v_end_mean']:.2f} / {r['v_end_max']:.2f} "
                   f"({100 * r['v_end_over']:.1f}% above the max return {g_max:.0f})"),
                  ("Imagined net reward / survival", f"{r['imag_reward_normalized']:.3f} / {100*r['imag_survival']:.2f}%"),
                  ("Action loop gain median / max", f"{r['a_gain_med']:.2f} / {r['a_gain_max']:.2f}")]  # fmt: skip
    if dr_terms:
        lines.append(("DR range scale (mean of terms)", f"{100 * r['dr_scale']:.0f}%"))
    if r["sim_nonfinite"]:
        lines.append(("Diverged sim states (reset)", f"{r['sim_nonfinite']} in this window"))
    if not policy:
        lines.append(("Policy updates", f"start after iteration {args.wm_only_iters}"))
    lines.append(("Sim net reward / maximum", f"{r['sim_reward_normalized']:.3f}"))
    lines.append(("Smoothness penalty / tracking", f"{100 * r['smooth_pen_frac']:.1f}% (sum qdd^2 {r['joint_acc']:.3g})"))
    if n_done:
        lines += [("Mean episode length", f"{r['ep_len']:.1f} steps ({r['ep_len'] * control_dt:.1f} s)"),
                  ("Mean episode return (normalized)", f"{r['ep_ret_norm']:.3f}"),
                  ("Terminated (fell) episodes", f"{100 * r['term_frac']:.2f}% ({r['episodes_failed']:.0f}/{n_done:.0f})"),
                  ("Recent success (no failure)", f"{100 * r['success_frac']:.2f}% | clip end {r['ended_clip']:.0f}, time limit {r['ended_time']:.0f}")]  # fmt: skip
    else:
        lines.append(("Episodes", "none finished in this window"))
    lines += [
        ("Tracking reward / maximum", f"{100*r['track_fraction']:.1f}% | net {r['sim_reward_normalized']:.3f}"),
        ("Reward body p/R/v/w (% max)", " / ".join(f"{100*r[k]/control_dt:.1f}" for k in REWARD_KEYS[:4])),
        ("Reward root p/R (% max)", " / ".join(f"{100*r[k]/(.5*control_dt):.1f}" for k in REWARD_KEYS[4:6])),
        ("Penalty rate/acc/limit (step)", " / ".join(f"{r[k]:.4g}" for k in REWARD_KEYS[6:])),
        ("Root RMSE XY / Z / ori", f"{r['error_root_xy_m']:.3f} m / {r['error_root_z_m']:.3f} m / {r['error_root_ori_rad']:.3f} rad"),
        ("Body RMSE pos / ori", f"{r['error_body_pos_m']:.3f} m / {r['error_body_ori_rad']:.3f} rad"),
        ("Body RMSE lin / ang vel", f"{r['error_body_linvel_mps']:.3f} m/s / {r['error_body_angvel_radps']:.3f} rad/s"),
        ("Joint RMSE pos / vel", f"{r['error_joint_pos_rad']:.3f} rad / {r['error_joint_vel_radps']:.3f} rad/s"),
        ("Action delta / joint accel RMS", f"{r['action_delta_rms']:.3f} / {r['joint_acc_rms']:.2f} rad/s2"),
    ]
    if pooled:
        lines += [
            ("Motion active / legacy / resident", f"{r['pool_active']:.0f} / {r['pool_legacy']:.0f} / {r['pool_resident']:.0f}"),
            ("Resident frames / reference memory", f"{r['pool_frames']:,.0f} | GPU {r['pool_gpu_gib']:.2f}, CPU {r['pool_cpu_gib']:.2f} GiB"),
            ("Last refresh / new files", f"{r['pool_refresh_s']:.2f}s / {r['pool_loaded']:.0f}"),
            ("Motion starts seen / catalog", f"{r['pool_starts_seen']:.0f}/{r['pool_catalog']:.0f} | {r['pool_starts_drawn']:,.0f} draws; not full-motion coverage"),
        ]
    footer = [("Sim steps", f"{r['sim_steps']:,}"),
              ("Speed", f"{r['it_per_s']:.2f} it/s, {steps_per_s:,.0f} sim steps/s"),
              ("Learning rate", f"{r['lr']:.2e}"),
              ("GPU memory" + (" (rank 0)" if world > 1 else ""), f"torch {torch.cuda.memory_reserved() / 2**30:.1f} GB, device {(total - free) / 2**30:.1f}/{total / 2**30:.1f} GB used"),
              ("Host RAM (RSS)", f"{_rss_gb():.1f} GB"),
              ("Last checkpoint", last_ckpt),
              ("Time elapsed", _hms(elapsed_s)),
              ("ETA", _hms(eta_s))]  # fmt: skip
    out = ["#" * width, " " * max((width - len(title)) // 2, 0) + f"\033[1m{title}\033[0m", ""]
    out += [f"{k + ':':>{pad}} {v}" for k, v in lines] + ["-" * width] + [f"{k + ':':>{pad}} {v}" for k, v in footer]
    print("\n".join(out), flush=True)
