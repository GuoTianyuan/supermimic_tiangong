"""Training CLI schema; importing it does not initialize Isaac Lab or CUDA."""
import argparse
import sys
from .experiment import CONFIG_NAMES, cli_defaults


class TrainingArgumentParser(argparse.ArgumentParser):
    def parse_known_args(self, args=None, namespace=None):
        argv = sys.argv[1:] if args is None else list(args)
        selector = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
        selector.add_argument("--config", choices=CONFIG_NAMES, default="default")
        selected, _ = selector.parse_known_args(argv)
        self.set_defaults(**cli_defaults(selected.config))
        return super().parse_known_args(argv, namespace)


def build_parser(add_launcher_args=None):
    defaults = cli_defaults()
    parser = TrainingArgumentParser(allow_abbrev=False)
    parser.add_argument("--config", choices=CONFIG_NAMES, default="default",
                        help="named task/agent preset; explicit CLI options override it")
    parser.add_argument("--print-config", action="store_true",
                        help="print grouped requested configuration without loading Isaac Lab; null values inherit on resume")
    parser.add_argument("--num_envs", type=int, default=defaults["num_envs"], help="environments per GPU")
    parser.add_argument("--iterations", type=int, default=defaults["iterations"])
    parser.add_argument("--buffer_steps", type=int, default=defaults["buffer_steps"])
    parser.add_argument("--utd", type=float, default=defaults["utd"], help="updates per vectorized simulator step")
    parser.add_argument("--horizon", type=int, default=defaults["horizon"])
    parser.add_argument("--goal_frames", type=int, default=defaults["goal_frames"])
    parser.add_argument("--goal_stride", type=int, default=defaults["goal_stride"],
                        help="future reference spacing in control frames (new run: 5; "
                        "resume: checkpoint spacing, legacy checkpoints: 1)")
    parser.add_argument("--goal_start", type=int, default=defaults["goal_start"],
                        help="first future slot in control frames (new run: 1; resume: checkpoint start, "
                        "or one stride for legacy checkpoints); 1/5 gives 10..460 ms at 100 Hz")
    parser.add_argument("--allow_reference_timing_change", action="store_true",
                        help="explicitly fine-tune a checkpoint with changed reference offsets, retaining all training states")
    parser.add_argument("--goal_drop", type=str, default=defaults["goal_drop"],
                        help="comma list of reference parts left out of the policy's goal input: root_pos (future pelvis "
                        "displacement), root_rot, root_lin_vel, root_ang_vel, q, qd. Rewards, termination and resets still "
                        "use the full reference. Default: none; a resumed run keeps its checkpoint's")  # fmt: skip
    parser.add_argument("--hist", type=int, default=defaults["hist"])
    parser.add_argument("--wm_window", type=int, default=defaults["wm_window"])
    parser.add_argument("--wm_prediction", choices=["state_delta", "velocity_integrated"], default=defaults["wm_prediction"],
                        help="WM output: original state increments, or velocity increments with explicit pose "
                        "integration (default: state_delta; resume keeps the checkpoint mode)")
    parser.add_argument("--wm_batches", type=int, default=defaults["wm_batches"])
    parser.add_argument("--wm_batch", type=int, default=defaults["wm_batch"])
    parser.add_argument("--pi_batch", type=int, default=defaults["pi_batch"])
    parser.add_argument("--lr", type=float, default=defaults["lr"])
    parser.add_argument("--gamma", type=float, default=defaults["gamma"])
    parser.add_argument("--lam", type=float, default=defaults["lam"])
    parser.add_argument("--action_noise", type=float, default=defaults["action_noise"])
    parser.add_argument("--root_pos_z_weight", type=float, default=None,
                        help="coefficient of squared root Z error; omitted: task preset or checkpoint")
    parser.add_argument("--body_pos_z_mode", choices=["reference", "relative_root"], default=None,
                        help="body-point height reward: absolute reference Z or relative to pelvis")
    parser.add_argument("--critic_epochs", type=int, default=defaults["critic_epochs"])
    parser.add_argument("--critic_minibatches", type=int, default=defaults["critic_minibatches"])
    parser.add_argument("--critic_tau", type=float, default=defaults["critic_tau"], help="per-iteration EMA rate of the target critic")
    parser.add_argument("--no_critic", action="store_true")
    # The next four default to the new behaviour; a resumed run keeps its checkpoint's choice.
    parser.add_argument("--critic_layer_norm", action=argparse.BooleanOptionalAction, default=defaults["critic_layer_norm"],
                        help="LayerNorm in the critic MLP (default: on)")
    parser.add_argument("--act_soft_clip", action=argparse.BooleanOptionalAction, default=defaults["act_soft_clip"],
                        help="policy output saturates smoothly from |a| = 15 towards the simulator clip of 20 (default: on)")
    parser.add_argument("--act_norm", choices=["ref", "running"], default=defaults["act_norm"],
                        help="normalize the action inputs with fixed per-joint statistics of the actions that hold the "
                        "reference poses, or with running statistics like the other inputs (default: ref)")  # fmt: skip
    parser.add_argument("--reset_a_prev", choices=["pose", "zero"], default=defaults["reset_a_prev"],
                        help="previous action at reset: the one that holds the reset pose, or 0 (default: pose)")
    parser.add_argument("--detach_a_prev", action="store_true",
                        help="in imagination, no policy gradient through the previous action that the policy observes")
    parser.add_argument("--reset_temp", type=float, default=defaults["reset_temp"],
                        help="inverse temperature of the return-based reset sampling (paper: 25)")
    parser.add_argument("--motion_active_ratio", type=float, default=defaults["motion_active_ratio"],
                        help="resident active fraction; <1 enables the rotating NPZ pool (no byte cap)")
    parser.add_argument("--motion_refresh_iters", type=int, default=defaults["motion_refresh_iters"])
    parser.add_argument("--motion_replace_ratio", type=float, default=defaults["motion_replace_ratio"],
                        help="fraction of the full catalog replaced at each refresh, as in WBT")
    parser.add_argument("--motion_stats_cache", default=defaults["motion_stats_cache"])
    parser.add_argument("--motion_exclude", type=str, default=defaults["motion_exclude"], help="regex; motion files whose name matches are not loaded")
    parser.add_argument("--warmup_steps", type=int, default=defaults["warmup_steps"], help="sim steps before any update")
    parser.add_argument("--wm_only_iters", type=int, default=defaults["wm_only_iters"], help="world-model-only iterations at the start")
    parser.add_argument("--lr_decay_iters", type=int, default=defaults["lr_decay_iters"])
    parser.add_argument("--ckpt_every", type=int, default=defaults["ckpt_every"])
    parser.add_argument("--log_every", type=int, default=defaults["log_every"])
    parser.add_argument("--motion_dir", type=str, default=defaults["motion_dir"], help="Motion directory or a single CSV/NPZ file")
    parser.add_argument("--log_dir", type=str, default=defaults["log_dir"])
    parser.add_argument("--seed", type=int, default=defaults["seed"])
    parser.add_argument("--resume", type=str, default=defaults["resume"])
    parser.add_argument("--amp", type=str, default=defaults["amp"], choices=["none", "bf16"], help="autocast dtype for MLP matmuls")
    parser.add_argument("--robot", choices=["g1", "tiangong3"], default=defaults["robot"])
    parser.add_argument("--dr", type=str, default=defaults["dr"], help="extra domain randomization for deployment (tiangong3): comma list of "
                        "pd, push, mass, armature, joint_friction, delay, contact, or all (all but contact; all,contact "
                        "adds it); see src/supermimic/envs/domain_randomization.py")  # fmt: skip
    parser.add_argument("--dr_curriculum", type=str, default=defaults["dr_curriculum"], help="A,B: no --dr terms before iteration A, all but "
                        "push and delay from A, all from B")  # fmt: skip
    parser.add_argument("--dr_ramp", type=int, default=defaults["dr_ramp"], help="iterations over which each --dr_curriculum stage widens "
                        "its ranges from nominal to full (0: at once)")  # fmt: skip
    parser.add_argument("--dynamics", choices=["default", "dexevt"], default=defaults["dynamics"],
                        help="tiangong3 joint dynamics and --dr ranges: default (100 N*m ankles, ankle armature 0.01-0.06, "
                        "Coulomb friction) | dexevt (current cheerleading_competition DexEVT: 55 N*m ankles, armature "
                        "0.002-0.06 on every joint, Coulomb and viscous friction 0.001-0.6). Default: default; a resumed "
                        "run keeps its checkpoint's")  # fmt: skip
    parser.set_defaults(**defaults)
    if add_launcher_args is not None:
        add_launcher_args(parser)
    return parser
