"""CPU contract checks; load actual tensor code with simulator-only imports stubbed.

This does not launch Isaac Lab, validate GPU capacity, or claim training quality.
Run with the training torch environment and PYTHONPATH cleared. --legacy-networks
can point to networks.py from before the ablation to check exact compatibility.
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
from pathlib import Path
import sys
import types

import torch


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--support-repo', type=Path, help='unchanged dependencies when testing staged sources')
    parser.add_argument('--legacy-networks', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = args.repo.resolve()
    support = (args.support_repo or root).resolve()
    torch.set_num_threads(1)
    torch.manual_seed(27)
    pkg = types.ModuleType('supermimic')
    pkg.__path__ = [str(root/'src/supermimic'), str(support/'src/supermimic')]
    sys.modules[pkg.__name__] = pkg
    # Only these simulator-facing dependencies are stubs. All numerical methods
    # under test (including observations, rotations, replay and WM) are real code.
    for name, fields in [('robots.kinematics', {'G1Kinematics': object}),
                         ('motions.library', {'CONTROL_DT': .01, 'MotionLib': object})]:
        mod = types.ModuleType(f'{pkg.__name__}.{name}')
        mod.__dict__.update(fields)
        sys.modules[mod.__name__] = mod
    rot = __import__("supermimic.utils.rotations", fromlist=["_"])
    mdp = __import__("supermimic.tasks.tracking.mdp", fromlist=["_"])
    net = __import__("supermimic.models", fromlist=["_"])
    bufmod = __import__("supermimic.storage.replay_buffer", fromlist=["_"])
    B = 4
    def states():
        return (torch.randn(B,3), rot.so3_exp(.3*torch.randn(B,3)),
                torch.randn(B,3), .2*torch.randn(B,3), torch.randn(B,29), torch.randn(B,29))
    S,S1=states(),states()
    a,ap,xi=torch.randn(B,29),torch.randn(B,29),torch.randn(B,33)
    common=dict(hidden=(32,24),xi_dim=33,a_prev=True)
    base=net.WorldModel(**common)
    integrated=net.WorldModel(**common,prediction='velocity_integrated',control_dt=.01,physics_substeps=4)
    report={}
    if args.legacy_networks:
        legacy=load(pkg.__name__+'.legacy_networks',args.legacy_networks)
        old=legacy.WorldModel(**common)
        with torch.no_grad():
            old.net[-1].weight.normal_(std=.02)
            old.net[-1].bias.normal_(std=.02)
        old.out_norm.update(old.targets(S,S1))
        base.load_state_dict(old.state_dict(),strict=True)
        assert set(base.state_dict())==set(old.state_dict())
        for x,y in zip(base.step(S,a,xi,ap),old.step(S,a,xi,ap)):
            torch.testing.assert_close(x,y,rtol=0,atol=0)
        old.out_norm.update(old.targets(S1,S))
        base.update_output_normalization(S1,S)
        torch.testing.assert_close(base.error_std(),old.out_norm.std(),rtol=0,atol=0)
        report['legacy_strict_load_prediction_and_norm_bitwise_equal']=True
    base=net.WorldModel(**common)
    for current,next_state in [(S,S1),(S1,S)]:
        base.update_output_normalization(current,next_state)
        integrated.update_output_normalization(current,next_state)
    torch.testing.assert_close(base.error_std(),integrated.error_std(),rtol=0,atol=0)
    assert integrated.out_norm.mean.shape==(35,) and integrated.error_std().shape==(70,)
    ids=torch.tensor(list(range(6,12))+list(range(41,70)))
    torch.testing.assert_close(integrated.out_norm.mean,base.out_norm.mean[ids])
    torch.testing.assert_close(integrated.out_norm.std(),base.out_norm.std()[ids])
    for model in (base,integrated):
        clone=net.WorldModel(**common,prediction=model.prediction,control_dt=.01,physics_substeps=4)
        clone.load_state_dict(model.state_dict(),strict=True)
        for x,y in zip(model.step(S,a,xi,ap),clone.step(S,a,xi,ap)):
            torch.testing.assert_close(x,y,rtol=0,atol=0)
    report['output_35_error_70_stats_and_checkpoint_roundtrip']=True
    dt=.01
    delta=.02*torch.randn(B,35)
    p,R,v,w,q,qd=S
    dv_w=(R@delta[:,:3,None]).squeeze(-1)
    dw_w=(R@delta[:,3:6,None]).squeeze(-1)
    class FixedDelta(torch.nn.Module):
        def forward(self,x): return delta
    for n in (1,2,4,8):
        wm=net.WorldModel(**common,prediction='velocity_integrated',control_dt=dt,physics_substeps=n)
        wm.net=FixedDelta()
        with torch.no_grad(): wm.out_norm.var.fill_(1-wm.out_norm.eps**2)
        result=wm.step(S,a,xi,ap)
        pn,vn,qn,qdn=p.clone(),v.clone(),q.clone(),qd.clone()
        # Independent semi-implicit reference: n constant-acceleration substeps.
        for _ in range(n):
            vn=vn+dv_w/n; pn=pn+dt/n*vn
            qdn=qdn+delta[:,6:]/n; qn=qn+dt/n*qdn
        for actual,expected in [(result[0],pn),(result[2],vn),(result[3],w+dw_w),
                                (result[4],qn),(result[5],qdn)]:
            torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-6)
        c=sum(range(1,n+1))/n**2
        theta_w=dt*(w+c*dw_w)
        K=torch.zeros(B,3,3)
        K[:,0,1]=-theta_w[:,2]; K[:,0,2]=theta_w[:,1]
        K[:,1,0]=theta_w[:,2]; K[:,1,2]=-theta_w[:,0]
        K[:,2,0]=-theta_w[:,1]; K[:,2,1]=theta_w[:,0]
        torch.testing.assert_close(result[1],torch.linalg.matrix_exp(K)@R,atol=2e-6,rtol=2e-6)
    report['integration_matches_1_2_4_8_substeps_and_world_frame_rotation']=True
    wm=net.WorldModel(**common,prediction='velocity_integrated',control_dt=dt,physics_substeps=4)
    for x,y in zip(wm.step(S,a,xi,ap),(p+dt*v,R@rot.so3_exp(dt*(R.transpose(-1,-2)@w[...,None]).squeeze(-1)),v,w,q+dt*qd,qd)):
        torch.testing.assert_close(x,y)
    report['zero_delta_is_constant_velocity_motion']=True
    for mode in ('state_delta','velocity_integrated'):
        for length in (12,48):
            wm=net.WorldModel(**common,prediction=mode,control_dt=dt,physics_substeps=4)
            with torch.no_grad(): wm.net[-1].weight.normal_(std=1e-4)
            actions=(.01*torch.randn(length,B,29)).requires_grad_(True)
            state=S
            for t in range(length): state=wm.step(state,actions[t],xi,actions[t-1] if t else ap)
            loss=wm.state_error(state,S1).square().mean()
            loss.backward()
            grads=[p.grad for p in wm.parameters() if p.grad is not None]
            assert grads and all(torch.isfinite(g).all() for g in grads)
            assert torch.isfinite(actions.grad).all() and actions.grad[0].abs().max()>0
    report['12_48_step_network_and_action_gradients_finite_nonzero']=True
    wm=net.WorldModel(**common,prediction='velocity_integrated',control_dt=dt,physics_substeps=4)
    compiled=torch.compile(wm.step,backend='eager',fullgraph=True)
    for x,y in zip(compiled(S,a,xi,ap),wm.step(S,a,xi,ap)): torch.testing.assert_close(x,y)
    report['integrated_step_dynamo_fullgraph_cpu']=True
    obs=mdp.MDP.__new__(mdp.MDP); obs.obs_noise=torch.ones(99)
    buf=bufmod.ReplayBuffer(17,2,'cpu')
    for t in range(43):
        tr={k:torch.zeros_like(v[0]) for k,v in buf.data.items()}
        tr['f'].fill_(t); tr['ep_step'][:]=torch.tensor([t%11,t%7])
        buf.add(tr)
    for H in (1,2,10):
        _,h=buf.sample_starts(2048,H)
        clean=h['f'].float().unsqueeze(-1).expand(-1,-1,99)
        noisy=obs.add_history_noise(clean,h['noise_source'])
        for i in range(H):
            for j in range(H):
                same=h['f'][:,i]==h['f'][:,j]
                assert torch.equal(noisy[same,i],noisy[same,j])
        assert (h['f']>=26).all() # retained ring boundary
        identity=torch.arange(H).expand(len(clean),H)
        torch.manual_seed(12); expected=obs.add_noise(clean)
        torch.manual_seed(12); actual=obs.add_history_noise(clean,identity)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    reset=torch.zeros(B,10,99)
    noise=obs.add_history_noise(reset,torch.full((B,10),9,dtype=torch.long))
    assert torch.equal(noise,noise[:,-1:].expand_as(noise))
    report['history_h1_h2_h10_reset_buffer_wrap_and_unpadded_rng']=True
    # Parse modified training source; no simulator launch in this CPU test.
    for path in (root/'src/supermimic').rglob('*.py'): ast.parse(path.read_text(),filename=str(path))
    source='\n'.join((root/f'src/supermimic/{path}').read_text() for path in ['runners/train.py','learners/wm_learner.py','learners/policy_learner.py'])
    assert source.count('mdp.add_history_noise(')==2
    assert 'wm.update_output_normalization(S0, S1)' in source
    assert 'torch.maximum(wm.error_std(), ERR_FLOOR)' in source
    assert 'Cannot resume a checkpoint with a different --wm_prediction' in source
    report['training_wiring_and_syntax']=True
    report['scope']='CPU tensor/gradient/graph tests, small networks; no full simulator or multi-GPU training test'
    text=json.dumps(report,indent=2)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(text)
    print(text)

if __name__=='__main__': main()
