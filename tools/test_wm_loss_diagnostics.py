"""CPU checks of real WM loss/optimizer code and detached channel diagnostics.

Simulator-only imports are stubbed. This is not a CUDA or simulator benchmark.
"""
from __future__ import annotations
import argparse
import ast
import copy
import contextlib
import io
import tempfile
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
    p = argparse.ArgumentParser()
    p.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--legacy-train', type=Path)
    p.add_argument('--output', type=Path)
    args = p.parse_args(); root = args.repo.resolve()
    torch.set_num_threads(1); torch.manual_seed(41)
    pkg = types.ModuleType('_tg_wm_metrics_test'); pkg.__path__ = [str(root/'supermimic')]
    sys.modules[pkg.__name__] = pkg
    for name, fields in [('kinematics', {'G1Kinematics': object}),
                         ('motion', {'CONTROL_DT': .01, 'MotionLib': object})]:
        mod = types.ModuleType(pkg.__name__+'.'+name); mod.__dict__.update(fields); sys.modules[mod.__name__] = mod
    rot = load(pkg.__name__+'.rotations', root/'supermimic/rotations.py')
    mdp = load(pkg.__name__+'.mdp', root/'supermimic/mdp.py')
    networks = load(pkg.__name__+'.networks', root/'supermimic/networks.py')
    diag = load(pkg.__name__+'.diagnostics', root/'supermimic/diagnostics.py')
    names = [f'joint_{i}' for i in range(29)]
    report = {}
    raw = torch.randn(4, 3, 70, requires_grad=True)
    scale = torch.linspace(.001, .1, 70); floor = scale*.7; floor[::3] = scale[::3]
    mask = torch.tensor([[1.,1.,1.], [1.,1.,0.], [1.,0.,0.], [1.,1.,1.]])
    square = (raw/scale).square()
    summary = diag.wm_error_summary(raw, square, mask, scale, floor)
    assert not summary.requires_grad
    metrics = diag.WorldModelMetrics('cpu', names); metrics.update(summary)
    out, channels = metrics.flush()
    numerator = (square.detach().double()*mask[...,None]).sum((0,1))
    physical = (raw.detach().double().square()*mask[...,None]).sum((0,1))
    torch.testing.assert_close(torch.tensor(out['wm_fit_loss_per_step']), (numerator.sum()/(12*70)).float())
    assert abs(out['wm_fit_valid_fraction']-.75)<1e-10
    assert abs(sum(out[f'wm_{g}_loss_contribution'] for g in diag.WM_GROUPS)-out['wm_fit_loss_per_step'])<1e-10
    assert abs(sum(out[f'wm_{g}_loss_share'] for g in diag.WM_GROUPS)-1)<1e-10
    assert set(out)==set(diag.WM_FIELDS) and len(channels)==70*5
    for group, start, end in metrics.groups:
        torch.testing.assert_close(torch.tensor(out[f'wm_{group}_normalized_mse']), (numerator[start:end].mean()/9).float())
        torch.testing.assert_close(torch.tensor(out[f'wm_{group}_raw_rmse']), (physical[start:end].mean()/9).sqrt().float())
        assert abs(out[f'wm_{group}_floor_fraction']-float((scale[start:end]<=floor[start:end]).float().mean()))<1e-7
    for i, name in enumerate(metrics.channel_names):
        torch.testing.assert_close(torch.tensor(channels[f'2_world_model/channels/{name}/scale']),scale[i])
        torch.testing.assert_close(torch.tensor(channels[f'2_world_model/channels/{name}/raw_rmse']), (physical[i]/9).sqrt().float())
    assert metrics.sums.count_nonzero()==0
    report['masked_loss_additivity_shares_physical_units_scales_and_detachment'] = True

    # Equal per-component error gives 29/70 for each joint group, NOT 1/6.
    metrics.update(diag.wm_error_summary(torch.ones(2,12,70),torch.ones(2,12,70),torch.ones(2,12),torch.ones(70),torch.zeros(70)))
    out,_=metrics.flush()
    assert abs(out['wm_joint_pos_loss_share']-29/70)<1e-10
    assert abs(out['wm_root_pos_loss_share']-3/70)<1e-10
    report['channel_count_weighting_is_preserved'] = True

    # Unequal minibatch size/horizon, changing scales and masks, and simulated
    # all_reduce must weight sufficient statistics rather than averaging means.
    raw2=torch.randn(2,5,70); scale2=2*scale; mask2=torch.tensor([[1.,1.,1.,1.,1.],[1.,1.,0.,0.,0.]])
    summary2=diag.wm_error_summary(raw2,(raw2/scale2).square(),mask2,scale2,floor)
    metrics.update(summary); metrics.update(summary2); merged=metrics.flush()
    metrics.update(summary)
    old_init, old_reduce = diag.dist.is_initialized,diag.dist.all_reduce
    try:
        diag.dist.is_initialized=lambda:True
        diag.dist.all_reduce=lambda tensor:tensor.add_(summary2.double())
        reduced=metrics.flush()
    finally:diag.dist.is_initialized,diag.dist.all_reduce=old_init,old_reduce
    assert merged==reduced
    out,channels=merged
    expected_scale=(scale*9+scale2*7)/16
    for i,name in enumerate(metrics.channel_names):
        torch.testing.assert_close(torch.tensor(channels[f'2_world_model/channels/{name}/scale']),expected_scale[i])
    assert abs(out['wm_fit_valid_fraction']-16/22)<1e-10
    empty,empty_ch=metrics.flush(); assert all(v==0 for v in (*empty.values(),*empty_ch.values()))
    metrics.update(diag.wm_error_summary(raw2,(raw2/scale2).square(),torch.zeros_like(mask2),scale2,floor))
    empty,empty_ch=metrics.flush(); assert all(v==0 for v in (*empty.values(),*empty_ch.values()))
    report['unequal_batches_masks_scales_rank_reduction_and_zero_valid'] = True

    source=(root/'supermimic/train.py').read_text(); tree=ast.parse(source)
    def extract(name, namespace, parsed=tree):
        node=next(n for n in parsed.body if isinstance(n,ast.FunctionDef) and n.name==name)
        exec(compile(ast.Module(body=[node],type_ignores=[]),'<actual train '+name+'>','exec'),namespace)
        return namespace[name]
    floors=torch.cat([torch.full((n,),v) for n,v in [(3,.003),(3,.005),(3,.02),(3,.05),(29,.003),(29,.05)]])
    legacy_tree=ast.parse(args.legacy_train.read_text()) if args.legacy_train else None
    for mode in ('state_delta','velocity_integrated'):
        B,H=3,12
        def states():
            return (torch.randn(B,H,3),rot.so3_exp(.05*torch.randn(B,H,3)),torch.randn(B,H,3),
                    torch.randn(B,H,3)*.1,torch.randn(B,H,29)*.05,torch.randn(B,H,29)*.1)
        S,S1=states(),states(); flat,flat1=mdp.flatten_state(S),mdp.flatten_state(S1)
        model=networks.WorldModel(hidden=(32,24),xi_dim=33,a_prev=True,prediction=mode,control_dt=.01,physics_substeps=4)
        model.update_output_normalization(S,S1)
        old=copy.deepcopy(model)
        opt=torch.optim.Adam(model.parameters(),lr=2e-4); opt_old=torch.optim.Adam(old.parameters(),lr=2e-4)
        namespace=dict(torch=torch,wm=model,ERR_FLOOR=floors,wm_error_summary=diag.wm_error_summary,
                       args=types.SimpleNamespace(wm_window=H),unflatten_state=mdp.unflatten_state,
                       normalize_xi=lambda x:x)
        rollout=extract('_wm_rollout',namespace); current_loss=extract('_wm_err',namespace)
        oldspace={**namespace,'wm':old}; rollout_old=extract('_wm_rollout',oldspace)
        if legacy_tree:old_loss=extract('_wm_err',oldspace,legacy_tree)
        else:
            def old_loss(pred,gt,mask):
                e=(old.state_error(pred,mdp.unflatten_state(gt))/torch.maximum(old.error_std(),floors)).square().mean(-1)
                return (e*mask).sum()/gt.shape[0],e[:,-1],e[:,0]
        action=torch.randn(B,H,29)*.1; prev=torch.randn_like(action); xi=torch.randn(B,H,33)
        mask=torch.ones(B,H); mask[1,4:]=0;mask[2,8:]=0
        for _ in range(3):
            pred=rollout(flat,action,prev,xi); pred_old=rollout_old(flat,action,prev,xi)
            now=current_loss(pred,flat1,mask); before=old_loss(pred_old,flat1,mask)
            for x,y in zip(now[:3],before):torch.testing.assert_close(x,y,rtol=0,atol=0)
            opt.zero_grad(); opt_old.zero_grad(); now[0].backward(); before[0].backward()
            for x,y in zip(model.parameters(),old.parameters()):
                torch.testing.assert_close(x.grad,y.grad,rtol=0,atol=0)
            opt.step();opt_old.step()
            for x,y in zip(model.parameters(),old.parameters()):torch.testing.assert_close(x,y,rtol=0,atol=0)
            metrics.update(now[3]); logged,_=metrics.flush()
            torch.testing.assert_close(torch.tensor(logged['wm_fit_loss_per_step']),now[0].detach()/H)
        # Ensure the real compiled loss including detached summaries has no
        # graph break. CPU eager backend does not benchmark CUDA kernels.
        pred=rollout(flat,action,prev,xi)
        compiled=torch.compile(current_loss,backend='eager',fullgraph=True)
        for x,y in zip(compiled(pred,flat1,mask),current_loss(pred,flat1,mask)):
            torch.testing.assert_close(x,y,rtol=0,atol=0)
        report[mode+'_12_step_loss_gradients_and_adam_updates_bitwise_unchanged']=True
    report['compiled_loss_fullgraph_and_all_batch_logging_matches_training_loss']=True

    csv_assignment=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CSV_FIELDS' for t in n.targets))
    namespace={**vars(diag)};exec(compile(ast.Module(body=[csv_assignment],type_ignores=[]),'<CSV>','exec'),namespace)
    fields=namespace['CSV_FIELDS'];assert len(fields)==len(set(fields))
    if legacy_tree:
        old_assignment=next(n for n in legacy_tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CSV_FIELDS' for t in n.targets))
        oldnamespace={**vars(diag)};exec(compile(ast.Module(body=[old_assignment],type_ignores=[]),'<old CSV>','exec'),oldnamespace)
        assert fields[:len(oldnamespace['CSV_FIELDS'])]==oldnamespace['CSV_FIELDS']
    assert set(diag.WM_FIELDS)<=set(fields)
    report['existing_csv_positions_preserved_new_fields_appended']=True
    # Exercise the actual terminal formatter and persist/read the diagnostic
    # TensorBoard schema, without starting a simulator or training process.
    sample={key:0. for key in fields};sample.update(merged[0]);sample.update(iteration=1000,sim_steps=100)
    printspace=dict(torch=torch,args=types.SimpleNamespace(wm_only_iters=500,wm_window=12,iterations=320000),
                    WM_GROUPS=diag.WM_GROUPS,WM_UNITS=diag.WM_UNITS,REWARD_KEYS=diag.REWARD_KEYS,
                    G_MAX=10.,CONTROL_DT=.01,dr=types.SimpleNamespace(terms=[]),WORLD=1,
                    motion=object(),MotionPool=type('TestPool',(),{}),_rss_gb=lambda:0.)
    extract('_hms',printspace);print_progress=extract('print_progress',printspace)
    old_mem,old_reserved=torch.cuda.mem_get_info,torch.cuda.memory_reserved
    output=io.StringIO()
    try:
        torch.cuda.mem_get_info=lambda:(80*2**30,80*2**30)
        torch.cuda.memory_reserved=lambda:0
        with contextlib.redirect_stdout(output):print_progress(sample,0,1.,10.,100.,'test.pt')
    finally:torch.cuda.mem_get_info,torch.cuda.memory_reserved=old_mem,old_reserved
    assert 'WM all-batch loss / valid' in output.getvalue() and 'WM joint_vel' in output.getvalue()
    from torch.utils.tensorboard import SummaryWriter
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    with tempfile.TemporaryDirectory(prefix='tg-wm-metrics-') as temp:
        tags={**{diag.WM_TB_TAGS[k]:v for k,v in merged[0].items()},**merged[1]}
        with SummaryWriter(temp) as writer:
            for tag,value in tags.items():writer.add_scalar(tag,value,1000)
        events=EventAccumulator(temp,size_guidance={'scalars':0});events.Reload()
        assert set(events.Tags()['scalars'])==set(tags)
        for tag,value in tags.items():
            event=events.Scalars(tag)[0];assert event.step==1000
            torch.testing.assert_close(torch.tensor(event.value),torch.tensor(value))
    report['actual_terminal_output_and_388_tensorboard_scalar_roundtrips']=True
    report['scope']='CPU real tensor/optimizer checks and simulated all_reduce; no live simulator, CUDA performance or multi-rank runtime claim'
    result=json.dumps(report,indent=2)+'\n';print(result)
    if args.output:args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(result)

if __name__=='__main__':main()
