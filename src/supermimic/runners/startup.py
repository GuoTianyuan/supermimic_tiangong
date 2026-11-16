"""One-off run description; progress output is handled by TrainingLogger."""
import os
import torch
from supermimic.envs.domain_randomization import LATE_TERMS, describe, describe_scale
from supermimic.motions.library import CONTROL_DT
from supermimic.tasks.tracking.mdp import ACTION_CLIP


def print_startup(components, dr_scale):
    args = components.args
    num_clips = components.num_clips
    num_frames = components.num_frames
    start_it = components.start_it
    dr = components.dr
    dr_curriculum = components.dr_curriculum
    G_MAX = components.max_return
    mdp = components.mdp
    WORLD = components.world
    MAIN = components.is_main
    HZ = args.horizon
    tb_url = os.environ.get("SUPERMIMIC_TB_URL")
    if MAIN:
        envs = f"{args.num_envs} envs" + (f" per GPU x {WORLD} GPUs = {args.num_envs * WORLD}" if WORLD > 1 else "")
        print("=" * 80)
        print(f" SuperMimic training | run {os.path.basename(os.path.normpath(args.log_dir))} | robot {args.robot} | {torch.cuda.get_device_name(0)}")
        print(f" Motion {args.motion_dir}: {num_clips} clips, {num_frames} frames" + (f" (split over {WORLD} GPUs)" if WORLD > 1 else ""))
        print(f" {envs} | iterations {start_it} -> {args.iterations} | world-model warm-up until {args.wm_only_iters}"
              f" | checkpoint every {args.ckpt_every}")
        print(f" Joint dynamics: {args.dynamics}")
        print(f" Robot self-collisions: {args.self_collisions}")
        print(f" Reward action-rate {mdp.cfg.w_action_rate:g}, joint-acc {mdp.cfg.w_joint_acc:g}; gamma {args.gamma:g}; "
              f"WM {args.wm_window} steps, Actor {HZ} steps; control {CONTROL_DT:g}s, physics {CONTROL_DT/4:g}s")
        print(" Domain randomization: friction, joint offsets, torso COM" + (f" + {describe(dr.terms)} (xi {dr.xi_dim})" if dr.terms else ""))
        if dr_curriculum:
            print(f"   curriculum: + {describe(dr.terms - LATE_TERMS) or '-'} from iteration {dr_curriculum[0]}, "
                  f"+ {describe(dr.terms & LATE_TERMS) or '-'} from {dr_curriculum[1]}, each widening over "
                  f"{args.dr_ramp} iterations; now {describe_scale(dr_scale)}")
        print(f" Stabilization: action inputs normalized by {args.act_norm}, actions "
              f"{'soft-clipped' if args.act_soft_clip else 'hard-clipped'} at {ACTION_CLIP:.0f}, reset a_prev "
              f"{args.reset_a_prev}{', a_prev detached in imagination' if args.detach_a_prev else ''}, critic "
              f"{'LayerNorm, ' if args.critic_layer_norm else ''}target EMA {args.critic_tau}, values clamped to +-{G_MAX:.0f}")
        if args.motion_exclude or args.reset_temp != 25.0:
            print(f" Motions excluded by {args.motion_exclude!r}" if args.motion_exclude else " All motions",
                  f"| reset sampling temperature {args.reset_temp:g}")
        print(f" Logs {args.log_dir}" + (f" | TensorBoard {tb_url}" if tb_url else ""))
        print(" Ctrl+C: stop after the current iteration and save a checkpoint")
        print("=" * 80, flush=True)
