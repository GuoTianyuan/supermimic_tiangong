"""Exercise pool refresh with circular replay, exact cached targets, diagnostics and gradients."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("SUPERMIMIC_ROBOT", "tiangong3")
import numpy as np
import torch

from supermimic.buffer import ReplayBuffer
from supermimic.diagnostics import RealMetrics, REWARD_KEYS
from supermimic.kinematics import G1Kinematics
from supermimic.mdp import MDP, MAX_STEP_REWARD, flatten_state, reward_cfg
from supermimic.motion import MotionLib
from supermimic.motion_pool import MotionPool
from supermimic.robot import JOINT_NAMES, default_joint_pos_list
from supermimic.env import TrackingSim


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    torch.manual_seed(9)
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = a.device
    kin = G1Kinematics(dev)
    q0 = torch.tensor(default_joint_pos_list(), device=dev)
    with tempfile.TemporaryDirectory() as tmp:
        files = []
        for i in range(12):
            n = 65+i*7
            phase = np.arange(n, dtype=np.float32)[:, None]
            pos = np.concatenate((phase*.001+i, phase*0, phase*0+1), -1)[:, None, :]
            quat = np.zeros((n, 1, 4), np.float32)
            quat[..., 0] = 1
            q = q0.cpu().numpy()[None] + .01*np.sin(phase*.03+i)
            f = Path(tmp)/f"clip_{i:02d}.npz"
            np.savez(f, fps=np.array([50.]), body_pos_w=pos, body_quat_w=quat,
                     joint_pos=q, joint_vel=np.zeros_like(q), body_lin_vel_w=np.zeros_like(pos),
                     body_ang_vel_w=np.zeros_like(pos))
            Path(str(f)+".metadata.json").write_text(json.dumps(dict(robot="tiangong3", joint_names=JOINT_NAMES,
                root_body_index=0, quaternion_format="wxyz", frames=n, fps=50.,
                sha256=hashlib.sha256(f.read_bytes()).hexdigest())))
            files.append(str(f))
        dense = MotionLib(files, kin, dev)
        full = MotionPool(files, kin, dev, active_ratio=1, stats_cache=str(Path(tmp)/"stats"))
        for k in MotionPool.FRAME_FIELDS:
            assert torch.equal(getattr(full, k), getattr(dense, k)), k
        # At ratio1 the pool must reproduce the original adaptive-bin reset sampler exactly.
        def sampler(m):
            return SimpleNamespace(motion=m, bin_G=torch.arange(m.num_bins,device=dev)*.001,
                bin_w_avg=torch.ones(m.num_bins,device=dev)/m.num_bins,G_max=5.,reset_temp=25.,device=dev)
        d_sampler, p_sampler = sampler(dense), sampler(full)
        torch.manual_seed(112)
        f0,b0=TrackingSim.sample_frames(d_sampler,1000)
        torch.manual_seed(112)
        f1,b1=TrackingSim.sample_frames(p_sampler,1000)
        assert torch.equal(f0,f1) and torch.equal(b0,b1)
        assert torch.equal(d_sampler.bin_w_avg,p_sampler.bin_w_avg)
        var, mean, count = full.reference_joint_stats()
        v0, m0 = torch.var_mean(dense.q, dim=0, correction=0)
        torch.testing.assert_close(mean, m0.double(), atol=2e-7, rtol=2e-6)
        torch.testing.assert_close(var, v0.double(), atol=1e-8, rtol=1e-5)
        assert count == dense.num_frames
        del full
        pool = MotionPool(files, kin, dev, active_ratio=.25, replace_ratio=.25, seed=10,
                          stats_cache=str(Path(tmp)/"stats"))
        vf, mf, nf = pool.reference_joint_stats()
        assert torch.equal(vf, var) and torch.equal(mf, mean) and nf == count
        f = pool.clip_start[:2]+torch.tensor([9, 121], device=dev)
        env = SimpleNamespace(f=f.clone(), start_bin=pool.frame_bin[pool.clip_start[:2]].clone(),
            bin_G=torch.arange(pool.num_bins, dtype=torch.float32, device=dev),
            bin_w_avg=torch.arange(1,pool.num_bins+1,dtype=torch.float32,device=dev))
        replay = ReplayBuffer(5, 2, dev, xi_dim=181)
        assert replay.reference_frames().numel() == 0
        for t in range(8):
            tr = {k: torch.zeros_like(v[0]) for k, v in replay.data.items()}
            tr["f"] = f+t
            tr["s"] = flatten_state(pool.ref_state(tr["f"]))
            tr["a_prev"].fill_(t*.01)
            tr["ep_step"].fill_(t)
            replay.add(tr)
            assert replay.reference_frames().shape == (min(t+1,5),2)
        assert replay.ptr == 3 and replay.size == 5
        old_ids = set(pool.resident_ids)
        mdp = MDP(kin, pool, q0, reward_cfg("dexevt"))
        xi = torch.zeros(10,181,device=dev)
        aa = torch.randn(10,29,device=dev)*.01
        aprev = torch.zeros_like(aa)

        def identity(f):
            cid = pool.clip_id[f]
            return pool.resident_global[cid].clone(), (f-pool.clip_start[cid]).clone()

        def evaluate(f):
            state = tuple(x.clone().requires_grad_(True) for x in pool.ref_state(f))
            rr, terms = mdp.reward(state, f, aa, aprev, terms=True, qd_prev=state[5])
            cc = mdp.critic_features(state,f,xi)
            grad = torch.autograd.grad(rr.sum()+.001*cc.square().sum(),state)
            assert all(torch.isfinite(x).all() for x in (*grad, rr, cc))
            torch.testing.assert_close(sum(terms[k] for k in REWARD_KEYS),rr,rtol=1e-6,atol=1e-8)
            return rr.detach(), cc.detach(), [x.detach() for x in grad]

        frames = replay.reference_frames().flatten()
        before_id = identity(frames)
        before_state = [x.clone() for x in pool.ref_state(frames)]
        before_bodies = [getattr(pool,k)[frames].clone() for k in ("body_p","body_R","body_v","body_w")]
        before_goal = pool.goal(frames,10).clone()
        before_env = identity(env.f)
        before_return = env.bin_G[env.start_bin].clone()
        before = evaluate(frames)
        assert pool.refresh(env,replay)
        frames = replay.reference_frames().flatten()
        after_id = identity(frames)
        assert all(torch.equal(x,y) for x,y in zip(before_id,after_id))
        assert all(torch.equal(x,y) for x,y in zip(before_env,identity(env.f)))
        assert torch.equal(before_return,env.bin_G[env.start_bin])
        assert all(torch.equal(x,y) for x,y in zip(before_state,pool.ref_state(frames)))
        assert all(torch.equal(x,getattr(pool,k)[frames]) for x,k in zip(before_bodies,("body_p","body_R","body_v","body_w")))
        assert torch.equal(before_goal,pool.goal(frames,10))
        after = evaluate(frames)
        assert torch.equal(before[0],after[0]) and torch.equal(before[1],after[1])
        assert all(torch.equal(x,y) for x,y in zip(before[2],after[2]))
        assert old_ids-set(pool.active_ids) and set(before_id[0].cpu().tolist()) <= set(pool.resident_ids)
        # The sampler only receives active bins, including after a full active-set replacement.
        assert set(pool.resident_global[pool.bin_clip[pool.active_bins]].cpu().tolist()) == set(pool.active_ids)
        f_draw,b_draw=TrackingSim.sample_frames(SimpleNamespace(motion=pool,bin_G=env.bin_G,
            bin_w_avg=env.bin_w_avg,G_max=5.,reset_temp=25.,device=dev),1000)
        assert set(identity(f_draw)[0].cpu().tolist()) <= set(pool.active_ids)
        # Load the original reward from this branch's parent: diagnostics must not change values/gradients.
        original = Path(tmp)/"mdp_before.py"
        original.write_bytes(subprocess.check_output(["git","-C",str(Path(__file__).resolve().parents[1]),
                                                      "show","65c6d15:supermimic/mdp.py"]))
        spec=importlib.util.spec_from_file_location("supermimic.mdp_before",original)
        module=importlib.util.module_from_spec(spec)
        sys.modules[spec.name]=module
        spec.loader.exec_module(module)
        old_mdp = module.MDP(kin,pool,q0,module.RewardCfg(**vars(mdp.cfg)))
        state = tuple(x.clone().requires_grad_(True) for x in pool.ref_state(frames))
        r0 = old_mdp.reward(state,frames,aa,aprev,qd_prev=state[5])
        r1, terms = mdp.reward(state,frames,aa,aprev,terms=True,qd_prev=state[5])
        assert torch.equal(r0,r1)
        g0=torch.autograd.grad(r0.sum(),state,retain_graph=True)
        g1=torch.autograd.grad(r1.sum(),state)
        assert all(torch.equal(x,y) for x,y in zip(g0,g1))
        meter = RealMetrics(dev)
        meter.update(dict(r=r1.detach(),terms={k:v.detach() for k,v in terms.items()}))
        metrics = meter.flush()
        assert abs(metrics["track_fraction"]-float(terms["track"].detach().mean()/MAX_STEP_REWARD)) < 1e-6
        assert all(np.isfinite(v) for v in metrics.values())
        assert all(v == 0 for v in meter.flush().values())
        # Repeat refreshes with unchanged valid replay. Every old reference must remain exact.
        compiled_calls = 0
        if a.compile:
            def query(f):
                S = pool.ref_state(f)
                return mdp.reward(S,f,aa,aprev,qd_prev=S[5]), mdp.critic_features(S,f,xi),pool.goal(f,10)
            query_c=torch.compile(query, fullgraph=True)
        for _ in range(10):
            pool.refresh(env,replay)
            frames=replay.reference_frames().flatten()
            assert all(torch.equal(x,y) for x,y in zip(before_id,identity(frames)))
            assert torch.equal(before_goal,pool.goal(frames,10))
            if a.compile:
                c=query_c(frames)
                e=query(frames)
                for x,y in zip(c,e):
                    torch.testing.assert_close(x,y,rtol=3e-4,atol=5e-5)
                compiled_calls += 1
        # A replay row/episode is the only reason for legacy residency. Release after both stop referring.
        replay.size=0
        j=pool.resident_ids.index(pool.active_ids[0])
        env.f.fill_(int(pool.clip_start[j])+1)
        env.start_bin.fill_(int(pool.frame_bin[env.f[0]]))
        pinned={pool.active_ids[0]}
        pool.refresh(env,replay)
        assert set(pool.resident_ids) == set(pool.active_ids)|pinned
        saved=Path(tmp)/"pool.pt"
        torch.save(pool.state_dict(env),saved)
        state=torch.load(saved,weights_only=True,map_location="cpu")
        restored=MotionPool(files,kin,dev,active_ratio=.25,replace_ratio=.25,
                            stats_cache=str(Path(tmp)/"stats"),state=state)
        assert restored.active_ids==pool.active_ids
        assert torch.equal(restored.bin_returns,pool.bin_returns)
        assert torch.equal(restored.starts_seen,pool.starts_seen)
        assert restored.rng.bit_generator.state==pool.rng.bit_generator.state
        report=dict(device=dev, circular_replay_rows=5, refreshes=pool.refresh_count,
            root_joint_and_cached_body_references_bitwise_equal=True, goals_rewards_critic_gradients_bitwise_equal=True,
            diagnostics_preserve_same_config_reward_and_gradients=True, full_catalog_action_stats_match=True,
            original_adaptive_sampler_preserved=True, only_active_clips_receive_starts=True,
            releases_unreferenced_legacy=True, checkpoint_pool_state_restored=True,
            compiled_refresh_calls=compiled_calls, metrics=metrics, pool=pool.metrics())
        Path(a.output).write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
