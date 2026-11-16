"""CPU regression checks for reset aggregation, bounded replay and critic logging.

Uses actual tensor code, stubbing only the simulator-facing mdp module. Optional
--legacy-train checks that the critic optimizer updates are unchanged bitwise.
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
import torch.nn.functional as F


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec); sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[2])
    p.add_argument('--legacy-train',type=Path)
    p.add_argument('--output',type=Path)
    args=p.parse_args(); root=args.repo.resolve()
    torch.set_num_threads(1); torch.manual_seed(11)
    pkg=types.ModuleType('supermimic'); pkg.__path__=[str(root/'src/supermimic')]
    sys.modules[pkg.__name__]=pkg
    stub=types.ModuleType(pkg.__name__+'.tasks.tracking.mdp')
    stub.STATE_DIM=71;stub.XI_DIM=33;stub.MAX_STEP_REWARD=.05
    sys.modules[stub.__name__]=stub
    sampling=__import__("supermimic.motions.sampling", fromlist=["_"])
    buffer=__import__("supermimic.storage.replay_buffer", fromlist=["_"])
    diagnostics=__import__("supermimic.utils.metrics", fromlist=["_"])
    report={}
    initial=torch.tensor([1.,2.,3.,4.,5.])
    bins=torch.tensor([3,1,3,3,1]); returns=torch.tensor([1.,-2.,7.,4.,8.])
    expected=initial.clone();expected[1]=.8*2+.2*3;expected[3]=.8*4+.2*4
    for _ in range(30):
        order=torch.randperm(len(bins)); actual=initial.clone()
        sampling.update_bin_returns_(actual,bins[order],returns[order])
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    untouched=initial.clone()
    sampling.update_bin_returns_(untouched,torch.empty(0,dtype=torch.long),torch.empty(0))
    torch.testing.assert_close(untouched,initial)
    unique=torch.tensor([0,2,4]); target=torch.tensor([2.,3.,-5.]); actual=initial.clone()
    sampling.update_bin_returns_(actual,unique,target)
    torch.testing.assert_close(actual[unique],.8*initial[unique]+.2*target,rtol=0,atol=0)
    report['duplicate_returns_order_invariant_unique_and_empty_updates']=True
    # Execute the actual env.step bookkeeping block with a minimal fake env.
    tree=ast.parse((root/'src/supermimic/envs/tracking_env.py').read_text())
    env=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='TrackingSim')
    step=next(n for n in env.body if isinstance(n,ast.FunctionDef) and n.name=='step')
    block=next(n for n in step.body if isinstance(n,ast.If) and ast.unparse(n.test)=='len(ids) > 0')
    obj=types.SimpleNamespace(ep_step=torch.ones(2),gamma=.995,ep_ret=torch.tensor([1.,3.]),
                             ep_disc_ret=torch.tensor([1.,3.]),start_bin=torch.zeros(2,dtype=torch.long),
                             bin_G=torch.zeros(1),reset=lambda ids:None)
    out={}
    exec(compile(ast.Module(body=[block],type_ignores=[]),'<actual reset update>','exec'),
         dict(self=obj,torch=torch,ids=torch.arange(2),term=torch.ones(2,dtype=torch.bool),
              out=out,MAX_STEP_REWARD=.05,update_bin_returns_=sampling.update_bin_returns_))
    torch.testing.assert_close(obj.bin_G,torch.tensor([.4]))
    report['actual_env_step_aggregates_shared_start_bin']=True

    # Run the real reset sampler on a 101-frame clip (its last bin has one frame).
    method=next(n for n in env.body if isinstance(n,ast.FunctionDef) and n.name=='sample_frames')
    namespace=dict(torch=torch,BIN_FRAMES=100)
    exec(compile(ast.Module(body=[method],type_ignores=[]),'<actual sample_frames>','exec'),namespace)
    motion=types.SimpleNamespace(bin_start=torch.tensor([0,100,101]),bin_clip=torch.tensor([0,0,1]),
                                 clip_end=torch.tensor([100,103]),num_bins=3,record_starts=lambda bins:None)
    fake=types.SimpleNamespace(motion=motion,device='cpu',reset_temp=25.,G_max=10.,
                               bin_G=torch.zeros(3),bin_w_avg=torch.ones(3)/3)
    for pooled in (False,True):
        if pooled:motion.active_bins=torch.arange(3)
        frames,bins=namespace['sample_frames'](fake,5000)
        assert not (bins==1).any()
        assert (frames+1<=motion.clip_end[motion.bin_clip[bins]]).all()
        assert set(bins.tolist())=={0,2}
    motion.active_bins=torch.tensor([1])
    try:namespace['sample_frames'](fake,1)
    except RuntimeError as error:assert 'successor frame' in str(error)
    else:raise AssertionError('No eligible reset bin must error explicitly')
    report['real_reset_sampler_excludes_last_frame_only_bin_dense_and_pool']=True

    cid=torch.arange(3).repeat_interleave(200); ends=torch.tensor([199,399,599])
    def make_buf(count):
        b=buffer.ReplayBuffer(9,3,'cpu')
        for t in range(count):
            tr={k:torch.zeros_like(v[0]) for k,v in b.data.items()}
            tr['f'][:]=torch.arange(3)*200+t
            tr['s'][:,0]=t;tr['xi'][:,0]=torch.arange(3)
            tr['ep_step'][:]=(t+torch.arange(3))%7
            b.add(tr)
        return b
    def old_sampler(b,B,H,horizon):
        starts,hists,n=[],[],0
        while n<B:
            s,h=b.sample_starts(2*B,H);f=s['f']
            idx=(f+horizon<=ends[cid[f]]).nonzero().squeeze(-1)[:B-n]
            starts.append({k:v[idx] for k,v in s.items()});hists.append({k:v[idx] for k,v in h.items()});n+=len(idx)
        return tuple({k:torch.cat([x[k] for x in ds]) for k in ds[0]} for ds in (starts,hists))
    for count in (5,23):
        b=make_buf(count)
        torch.manual_seed(71); expected=old_sampler(b,32,10,48)
        torch.manual_seed(71); actual=b.sample_valid_starts(32,10,48,cid,ends)
        for x,y in zip(expected,actual):
            for k in x: torch.testing.assert_close(x[k],y[k],rtol=0,atol=0)
        # Only two eligible cells, different environment identities; force fallback.
        b.data['f'].fill_(199)
        b.data['f'][1,0]=0;b.data['f'][3,2]=400
        s,h=b.sample_valid_starts(10000,10,48,cid,ends,max_attempts=0)
        assert set(s['f'].tolist())=={0,400}
        assert abs(float((s['f']==0).float().mean())-.5)<.025
        for i in range(10):
            assert (h['s'][:,i,0]<=s['s'][:,0]).all()
            assert (h['s'][:,i,0]>=max(0,count-b.T)).all()
        assert torch.equal(h['xi'][:,:,0],s['xi'][:,None,0].expand(-1,10))
        assert torch.equal(h['s'][:,-1],s['s'])
        b.data['f'].fill_(199)
        calls=[0]; sample=b.sample_starts
        def counted(*a,**kw): calls[0]+=1;return sample(*a,**kw)
        b.sample_starts=counted
        try:b.sample_valid_starts(16,10,48,cid,ends)
        except RuntimeError as error: assert 'No replay start' in str(error)
        else:raise AssertionError('Entirely invalid replay must not spin or sample invalid states')
        assert calls[0]==8
    report['fast_path_seeded_equal_exact_fallback_uniform_wrap_and_no_valid_error']=True

    # Weighted diagnostics must pool numerators/mass, including unequal batches.
    m=diagnostics.CriticMetrics('cpu')
    v=torch.tensor([0.,4.,-3.,2.]);y=torch.tensor([.2,1.,2.,1.]);w=torch.tensor([1.,.05,0.,.8])
    e=(v-y).abs(); h=torch.where(e<=1,.5*e.square(),e-.5)
    for section in (slice(0,1),slice(1,4)):
        m.update(v[section],y[section],w[section],h[section])
    actual=m.flush()
    torch.testing.assert_close(torch.tensor(actual['fit_weighted_huber']),(w*h).sum()/w.sum())
    torch.testing.assert_close(torch.tensor(actual['fit_weighted_mse']),(w*(v-y).square()).sum()/w.sum())
    assert not m.sums.any()
    # Equivalent rank shards add raw sums, as the all_reduce in flush does.
    m0,m1=diagnostics.CriticMetrics('cpu'),diagnostics.CriticMetrics('cpu')
    m0.update(v[:1],y[:1],w[:1],h[:1]);m1.update(v[1:],y[1:],w[1:],h[1:])
    m0.sums+=m1.sums
    for k,val in m0.flush().items(): assert abs(val-actual[k])<1e-7
    report['critic_weighted_all_minibatch_and_rank_sum_aggregation']=True

    source='\n'.join(p.read_text() for p in (root/'src/supermimic/runners').glob('*.py'))+'\n'+(root/'src/supermimic/learners/wm_learner.py').read_text()+'\n'+(root/'src/supermimic/learners/policy_learner.py').read_text()+'\n'+(root/'src/supermimic/learners/critic_learner.py').read_text()+'\n'+(root/'src/supermimic/logging/legacy.py').read_text()
    def critic_loop(text):
        loop=next(n for n in ast.walk(ast.parse(text)) if isinstance(n,ast.For) and ast.unparse(n.iter)=='range(args.critic_epochs)')
        return compile(ast.Module(body=[loop],type_ignores=[]),'<actual critic fit loop>','exec')
    if args.legacy_train:
        class Critic(torch.nn.Module):
            def __init__(self):
                super().__init__();self.net=torch.nn.Linear(3,1);self.seen=[]
            def forward(self,x):
                value=self.net(x).squeeze(-1);self.seen.append((x[:,0].long().clone(),value.detach().clone()));return value
        c0,c1=Critic(),Critic();c1.load_state_dict(c0.state_dict())
        Xf=torch.randn(16,3);Xf[:,0]=torch.arange(16);Yf=torch.randn(16);Wf=torch.rand(16)
        metrics=diagnostics.CriticMetrics('cpu')
        for c,text in ((c0,args.legacy_train.read_text()),(c1,source)):
            opt=torch.optim.Adam(c.parameters(),lr=2e-4)
            torch.manual_seed(17)
            exec(critic_loop(text),dict(critic=c,opt_v=opt,Xf=Xf,Yf=Yf,Wf=Wf,n=16,mb=4,
                 args=types.SimpleNamespace(critic_epochs=5,critic_minibatches=4),dev='cpu',
                 torch=torch,F=F,critic_metrics=metrics,sync_grads=lambda _:None,
                 clip_finite_grads=lambda model,norm:torch.nn.utils.clip_grad_norm_(model.parameters(),norm)))
        for x,y in zip(c0.parameters(),c1.parameters()):torch.testing.assert_close(x,y,rtol=0,atol=0)
        assert len(c1.seen)==20
        mass=hs=ms=0.
        for idx,value in c1.seen:
            error=(value-Yf[idx]).abs(); weight=Wf[idx]
            hs+=float((weight*torch.where(error<=1,.5*error.square(),error-.5)).sum())
            ms+=float((weight*error.square()).sum());mass+=float(weight.sum())
        logged=metrics.flush()
        assert abs(logged['fit_weighted_huber']-hs/mass)<1e-7
        assert abs(logged['fit_weighted_mse']-ms/mass)<1e-6
        report['actual_20_minibatch_updates_bitwise_unchanged_and_metrics_match']=True
    assert 'legacy_last_minibatch_mse' in source and 'extra.update(critic_metrics.flush())' in source
    assert '*EXTRA_FIELDS, *CRITIC_FIELDS,' in source
    for file in (root/'src/supermimic').rglob('*.py'):ast.parse(file.read_text(),filename=str(file))
    report['scope']='CPU source-level numerical/regression checks; no real simulator, CUDA or multi-rank runtime test'
    result=json.dumps(report,indent=2)+'\n';print(result)
    if args.output:args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(result)

if __name__=='__main__':main()
