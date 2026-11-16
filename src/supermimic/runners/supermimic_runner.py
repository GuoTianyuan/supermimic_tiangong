"""SuperMimic collect -> WM -> imagined Actor/Critic training schedule.

The runner coordinates existing learners; it does not define reward, networks,
optimizers, CSV/TensorBoard formatting or checkpoint serialization.
"""
import os
import time
import math
import torch
from supermimic.envs.domain_randomization import curriculum_scale, describe_scale
from supermimic.learners.wm_learner import build_world_model_learner
from supermimic.learners.policy_learner import build_policy_learner
from supermimic.storage.replay_buffer import ReplayBuffer
from supermimic.utils.metrics import RealMetrics, CriticMetrics, WorldModelMetrics
from supermimic.motions.library import CONTROL_DT
from supermimic.motions.pool import MotionPool
from supermimic.robots import G1_JOINT_NAMES
from supermimic.logging.legacy import SMOOTH_KEYS, _hms
from supermimic.runners.collector import RolloutCollector
from supermimic.runners.checkpoints import save_checkpoint
from supermimic.runners.metrics import TrainingMetrics
from supermimic.runners.startup import print_startup
from supermimic.logging.episode import EpisodeMetrics
from supermimic.logging.logger import TrainingLogger


def learning_rate_at(args, it):
    if it >= args.lr_decay_iters:
        return args.lr * 0.03
    return args.lr * (0.03 + 0.97 * 0.5 * (1 + math.cos(math.pi * it / args.lr_decay_iters)))


class SuperMimicRunner:
    def __init__(self, components, env, stop):
        self.components, self.env, self.stop = components, env, stop

    def learn(self):
        components, env, stop = self.components, self.env, self.stop
        args = components.args
        dev = components.dev
        dr = components.dr
        dr_curriculum = components.dr_curriculum
        motion = components.motion
        mdp = components.mdp
        wm = components.wm
        pi = components.pi
        critic = components.critic
        critic_targ = components.critic_targ
        opt_wm = components.opt_wm
        opt_pi = components.opt_pi
        opt_v = components.opt_v
        start_it = components.start_it
        wall_offset = components.wall_offset
        GOAL_DROP = components.goal_drop
        G_MAX = components.max_return
        WORLD = components.world
        RANK = components.rank
        MAIN = components.is_main
        HZ = args.horizon
        PROFILE = bool(int(os.environ.get("SM_PROFILE", "0")))
        dr_scale = curriculum_scale(dr.terms, dr_curriculum, args.dr_ramp, start_it)
        buffer = ReplayBuffer(args.buffer_steps, args.num_envs, dev, xi_dim=dr.xi_dim)
        real_metrics = RealMetrics(dev)
        critic_metrics = CriticMetrics(dev)
        wm_metrics = WorldModelMetrics(dev, G1_JOINT_NAMES)
        episode_metrics = EpisodeMetrics(args.num_envs, env.max_episode_steps * CONTROL_DT, dev)
        collector = RolloutCollector(components, env, buffer, real_metrics, episode_metrics)
        wm_learner = build_world_model_learner(args=args, wm=wm, opt_wm=opt_wm, buffer=buffer,
            wm_metrics=wm_metrics, dev=dev, world=WORLD)
        policy_learner = build_policy_learner(args=args, wm=wm, pi=pi, critic=critic, critic_targ=critic_targ,
            opt_pi=opt_pi, opt_v=opt_v, buffer=buffer, mdp=mdp, motion=motion,
            critic_metrics=critic_metrics, dev=dev, GOAL_DROP=GOAL_DROP, G_MAX=G_MAX, world=WORLD, PROFILE=PROFILE)

        print_startup(components, dr_scale)
        stats = {"r": [], "ep_len": [], "ep_ret": [], "ep_count": [], "end_clip": [], "end_time": [], "term": [], "done": [], **{k: [] for k in SMOOTH_KEYS}}
        metrics = TrainingMetrics()
        # The replay buffer is not checkpointed, so a resumed run refills it before sampling world-model windows.
        if MAIN:
            print(f"Collecting {args.warmup_steps} warm-up steps; the first iterations compile kernels with torch.compile...", flush=True)
        collector.collect(args.warmup_steps, stats)
        logger = TrainingLogger(args, start_it, enabled=MAIN, control_dt=CONTROL_DT)
        t_start = time.time() - wall_offset
        t_session = t_log = time.time()
        steps_log = collector.sim_steps
        last_log_it = start_it
        step_acc = 0.0
        agg = {"wm": [], "wm8": [], "wm1": [], "pi": [], "v": [], "ir": [], "gn": [], "surv": [], "vm": [], "vx": [], "vo": []}
        done_it = saved_it = start_it
        last_ckpt = "none yet"
        try:
            for it in range(start_it, args.iterations):
                if stop.requested():
                    break
                if it == args.wm_only_iters and it > start_it and MAIN:
                    print(f"World-model warm-up done; policy and critic updates start at iteration {it + 1} (first ones compile kernels).", flush=True)
                scale = curriculum_scale(dr.terms, dr_curriculum, args.dr_ramp, it)
                if scale != dr_scale:
                    env.set_dr_scale(scale)
                    if MAIN:
                        print(f"Domain randomization ranges at iteration {it}: {describe_scale(scale)}", flush=True)
                    dr_scale = scale
                lr = learning_rate_at(args, it)
                for o in (opt_wm, opt_pi, opt_v):
                    for gparam in o.param_groups:
                        gparam["lr"] = lr
                step_acc += 1.0 / args.utd
                n = int(step_acc)
                step_acc -= n
                prof = PROFILE and it % 20 == 0 and it > args.wm_only_iters + 5
                if prof:
                    torch.cuda.synchronize(); tp0 = time.time()
                collector.collect(n, stats)
                if prof:
                    torch.cuda.synchronize(); tp1 = time.time()
                for _ in range(args.wm_batches):
                    l, e8, e1 = wm_learner.update()
                if prof:
                    torch.cuda.synchronize(); tp2 = time.time()
                agg["wm"].append(l), agg["wm8"].append(e8), agg["wm1"].append(e1)
                if it >= args.wm_only_iters:
                    pl, vl, ir, gn, surv, vm, vx, vo = policy_learner.update()
                    if prof:
                        torch.cuda.synchronize(); tp3 = time.time()
                        print(f"[prof] collect {1e3*(tp1-tp0):.0f} ms ({n} steps) | wm {1e3*(tp2-tp1):.0f} ms | policy+critic {1e3*(tp3-tp2):.0f} ms", flush=True)
                    agg["pi"].append(pl), agg["v"].append(vl), agg["ir"].append(ir), agg["gn"].append(gn), agg["surv"].append(surv)
                    agg["vm"].append(vm), agg["vx"].append(vx), agg["vo"].append(vo)
                done_it = it + 1
                if isinstance(motion, MotionPool) and args.motion_refresh_iters and done_it % args.motion_refresh_iters == 0:
                    motion.refresh(env, buffer)
                    if int(motion.catalog_len[motion.active_ids].max()) <= HZ:
                        raise ValueError("No active clip can supply the imagination horizon; increase active ratio")
                    print(f"[rank {RANK}] motion pool refreshed: {motion.metrics()}", flush=True)
                if it == start_it and MAIN:
                    print(f"First iteration done after {time.time() - t_session:.0f} s; progress is printed every {args.log_every} iterations.", flush=True)
                if done_it % args.log_every == 0:
                    now = time.time()
                    ips = (done_it - last_log_it) / (now - t_log)
                    last_log_it = done_it
                    sps = (collector.sim_steps - steps_log) / (now - t_log)
                    t_log, steps_log = now, collector.sim_steps
                    row, channels = metrics.reduce(components=components, env=env,
                        real_metrics=real_metrics, critic_metrics=critic_metrics, wm_metrics=wm_metrics,
                        policy_learner=policy_learner, stats=stats, agg=agg, it=it, done_it=done_it,
                        now=now, t_start=t_start, sim_steps=collector.sim_steps, ips=ips, lr=lr, dr_scale=dr_scale)
                    episode = episode_metrics.flush(CONTROL_DT)
                    eta = (now - t_session) / (done_it - start_it) * (args.iterations - done_it)
                    logger.log(row, channels, episode, steps_per_s=sps, elapsed_s=now-t_start,
                               eta_s=eta, last_ckpt=last_ckpt)
                    stats = {k: [] for k in stats}
                    agg = {k: [] for k in agg}
                if done_it % args.ckpt_every == 0 or done_it == args.iterations:
                    last_ckpt, saved_it = save_checkpoint(done_it, components=components, env=env, sim_steps=collector.sim_steps, t_start=t_start), done_it
            if done_it > saved_it:
                save_checkpoint(done_it, components=components, env=env, sim_steps=collector.sim_steps, t_start=t_start)
        finally:
            logger.close()
        if MAIN:
            finished = done_it == args.iterations
            print(f"Training {'finished' if finished else 'stopped'} at iteration {done_it}/{args.iterations} "
                  f"({_hms(time.time() - t_start)} total).", flush=True)
            if not finished and done_it > 0:
                print(f"Resume with: --resume {os.path.join(args.log_dir, 'last.pt')}", flush=True)
