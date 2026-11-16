"""Import-safe training entry point: CLI -> dependencies -> simulator -> runner."""
import os
from supermimic.configs.train import build_parser


def main(argv=None):
    from isaaclab.app import add_launcher_args, launch_simulation
    args = build_parser(add_launcher_args).parse_args(argv)
    os.environ["SUPERMIMIC_ROBOT"] = args.robot
    # Robot-dependent modules must be imported after selecting the robot.
    from supermimic.runners.runtime import StopController
    with StopController() as stop:
        import torch.distributed as dist
        from supermimic.runners.setup import build_training
        from supermimic.runners.supermimic_runner import SuperMimicRunner
        from supermimic.envs.sim_context import make_sim
        from supermimic.envs.tracking_env import TrackingSim
        from supermimic.envs.domain_randomization import curriculum_scale
        rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
        try:
            c = build_training(args, rank, world)
            stop.device, stop.world = c.dev, world
            sim_cfg = make_sim(c.dev, args)
            with launch_simulation(sim_cfg, args):
                scale = curriculum_scale(c.dr.terms, c.dr_curriculum, args.dr_ramp, c.start_it)
                env = TrackingSim(sim_cfg, args.num_envs, c.motion, c.mdp, gamma=args.gamma,
                                  dr=c.dr, dr_scale=scale, reset_a_prev=args.reset_a_prev, reset_temp=args.reset_temp)
                SuperMimicRunner(c, env, stop).learn()
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()


if __name__ == "__main__":
    main()
