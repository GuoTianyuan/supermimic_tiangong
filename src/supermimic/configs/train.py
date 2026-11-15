"""Training CLI schema; importing it does not initialize Isaac Lab or CUDA."""
import argparse
import os
from supermimic.utils.paths import REPO_ROOT

REPO = str(REPO_ROOT)


def build_parser(add_launcher_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=4096, help="environments per GPU")
    parser.add_argument("--iterations", type=int, default=320_000)
    parser.add_argument("--buffer_steps", type=int, default=1000)
    parser.add_argument("--utd", type=float, default=0.4, help="updates per vectorized simulator step")
    parser.add_argument("--horizon", type=int, default=48)
    parser.add_argument("--goal_frames", type=int, default=10)
    parser.add_argument("--goal_stride", type=int, default=None,
                        help="future reference spacing in control frames (new run: 5; "
                        "resume: checkpoint spacing, legacy checkpoints: 1)")
    parser.add_argument("--goal_start", type=int, default=None,
                        help="first future slot in control frames (new run: 1; resume: checkpoint start, "
                        "or one stride for legacy checkpoints); 1/5 gives 10..460 ms at 100 Hz")
    parser.add_argument("--allow_reference_timing_change", action="store_true",
                        help="explicitly fine-tune a checkpoint with changed reference offsets, retaining all training states")
    parser.add_argument("--goal_drop", type=str, default=None,
                        help="comma list of reference parts left out of the policy's goal input: root_pos (future pelvis "
                        "displacement), root_rot, root_lin_vel, root_ang_vel, q, qd. Rewards, termination and resets still "
                        "use the full reference. Default: none; a resumed run keeps its checkpoint's")  # fmt: skip
    parser.add_argument("--hist", type=int, default=10)
    parser.add_argument("--wm_window", type=int, default=12)
    parser.add_argument("--wm_prediction", choices=["state_delta", "velocity_integrated"], default=None,
                        help="WM output: original state increments, or velocity increments with explicit pose "
                        "integration (default: state_delta; resume keeps the checkpoint mode)")
    parser.add_argument("--wm_batches", type=int, default=10)
    parser.add_argument("--wm_batch", type=int, default=2048)
    parser.add_argument("--pi_batch", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--action_noise", type=float, default=0.1)
    parser.add_argument("--critic_epochs", type=int, default=5)
    parser.add_argument("--critic_minibatches", type=int, default=4)
    parser.add_argument("--critic_tau", type=float, default=0.05, help="per-iteration EMA rate of the target critic")
    parser.add_argument("--no_critic", action="store_true")
    # The next four default to the new behaviour; a resumed run keeps its checkpoint's choice.
    parser.add_argument("--critic_layer_norm", action=argparse.BooleanOptionalAction, default=None,
                        help="LayerNorm in the critic MLP (default: on)")
    parser.add_argument("--act_soft_clip", action=argparse.BooleanOptionalAction, default=None,
                        help="policy output saturates smoothly from |a| = 15 towards the simulator clip of 20 (default: on)")
    parser.add_argument("--act_norm", choices=["ref", "running"], default=None,
                        help="normalize the action inputs with fixed per-joint statistics of the actions that hold the "
                        "reference poses, or with running statistics like the other inputs (default: ref)")  # fmt: skip
    parser.add_argument("--reset_a_prev", choices=["pose", "zero"], default=None,
                        help="previous action at reset: the one that holds the reset pose, or 0 (default: pose)")
    parser.add_argument("--detach_a_prev", action="store_true",
                        help="in imagination, no policy gradient through the previous action that the policy observes")
    parser.add_argument("--reset_temp", type=float, default=25.0,
                        help="inverse temperature of the return-based reset sampling (paper: 25)")
    parser.add_argument("--motion_active_ratio", type=float, default=1.0,
                        help="resident active fraction; <1 enables the rotating NPZ pool (no byte cap)")
    parser.add_argument("--motion_refresh_iters", type=int, default=1500)
    parser.add_argument("--motion_replace_ratio", type=float, default=.25,
                        help="fraction of the full catalog replaced at each refresh, as in WBT")
    parser.add_argument("--motion_stats_cache", default=os.path.join(REPO, ".cache", "motion_pool"))
    parser.add_argument("--motion_exclude", type=str, default="", help="regex; motion files whose name matches are not loaded")
    parser.add_argument("--warmup_steps", type=int, default=64, help="sim steps before any update")
    parser.add_argument("--wm_only_iters", type=int, default=500, help="world-model-only iterations at the start")
    parser.add_argument("--lr_decay_iters", type=int, default=250_000)
    parser.add_argument("--ckpt_every", type=int, default=2000)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--motion_dir", type=str, default=None, help="Motion directory or a single CSV/NPZ file")
    parser.add_argument("--log_dir", type=str, default=os.path.join(REPO, "logs", "run"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--amp", type=str, default="bf16", choices=["none", "bf16"], help="autocast dtype for MLP matmuls")
    parser.add_argument("--robot", choices=["g1", "tiangong3"], default=os.environ.get("SUPERMIMIC_ROBOT", "g1"))
    parser.add_argument("--dr", type=str, default="", help="extra domain randomization for deployment (tiangong3): comma list of "
                        "pd, push, mass, armature, joint_friction, delay, contact, or all (all but contact; all,contact "
                        "adds it); see src/supermimic/envs/domain_randomization.py")  # fmt: skip
    parser.add_argument("--dr_curriculum", type=str, default="", help="A,B: no --dr terms before iteration A, all but "
                        "push and delay from A, all from B")  # fmt: skip
    parser.add_argument("--dr_ramp", type=int, default=2000, help="iterations over which each --dr_curriculum stage widens "
                        "its ranges from nominal to full (0: at once)")  # fmt: skip
    parser.add_argument("--dynamics", choices=["default", "dexevt"], default=None,
                        help="tiangong3 joint dynamics and --dr ranges: default (100 N*m ankles, ankle armature 0.01-0.06, "
                        "Coulomb friction) | dexevt (current cheerleading_competition DexEVT: 55 N*m ankles, armature "
                        "0.002-0.06 on every joint, Coulomb and viscous friction 0.001-0.6). Default: default; a resumed "
                        "run keeps its checkpoint's")  # fmt: skip
    if add_launcher_args is not None:
        add_launcher_args(parser)
    return parser
