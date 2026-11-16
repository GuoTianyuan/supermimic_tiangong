"""One sole point replaces each ankle origin: physical geometry, gradients and compatibility."""
from dataclasses import asdict, replace
import json
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch

from supermimic.configs.experiment import load_config, config_snapshot, resolve_reward_config
from supermimic.configs.train import build_parser
from supermimic.robots import MJCF_PATH, JOINT_NAMES, default_joint_pos_list
from supermimic.robots.kinematics import G1Kinematics
from supermimic.tasks.tracking.config import from_task_cfg, reward_cfg, checkpoint_reward_cfg
from supermimic.tasks.tracking.mdp import MDP
from supermimic.utils.rotations import mat_to_quat, so3_exp


def baseline_cfg():
    env,_=load_config('tiangong3_full_wbc')
    return from_task_cfg(env.rewards,env.terminations)


def sample():
    torch.manual_seed(160)
    kin=G1Kinematics('cpu'); n=4
    q0=torch.tensor(default_joint_pos_list())
    p=torch.tensor([.1,-.2,.85]).repeat(n,1)
    R=so3_exp(torch.randn(n,3)*.1)
    v,w=torch.randn(n,3)*.15,torch.randn(n,3)*.15
    q=q0.repeat(n,1)+torch.randn(n,29)*.03
    qd=torch.randn(n,29)*.12
    S=(p,R,v,w,q,qd)
    cached=tuple(x.detach().clone() for x in kin.forward(*S))
    def bodies(f,positions_only=False):
        return (cached[0][f],None,None,None) if positions_only else tuple(x[f] for x in cached)
    motion=SimpleNamespace(root_p=p.clone(),root_R=R.clone(),q=q.clone(),qd=qd.clone(),body_reference=bodies)
    cfg=baseline_cfg()
    new=MDP(kin,motion,q0,cfg)
    old=MDP(kin,motion,q0,replace(cfg,body_point_offsets={}))
    return new,old,S,torch.arange(n),torch.zeros(n,29),cached


def test_exactly_one_point_per_foot_and_no_added_reward_weight():
    new,old,S,f,a,cached=sample()
    assert new.cfg.body_point_offsets=={
        'ankle_roll_l_link':(.045,0.,-.058),'ankle_roll_r_link':(.045,0.,-.058)}
    assert len(new.kin.tracked_idx)==14
    assert (new.tracking_points.offsets.abs().sum(-1)>0).sum()==2
    assert new.max_step_reward==old.max_step_reward==.05
    r,terms=new.reward(S,f,a,a,terms=True,qd_prev=S[5])
    torch.testing.assert_close(r,torch.full_like(r,.05))
    assert not any('foot' in key for key in terms if key.startswith(('reward_','pen_')))
    torch.testing.assert_close(r,sum(v for k,v in terms.items() if k.startswith(('reward_','pen_')) and k!='pen_smooth'))
    # Same pose/orientation/velocity references, no changes to observable inputs or privileged critic dimensions.
    xi=torch.zeros(4,33)
    torch.testing.assert_close(new.obs(S,a,xi,f),old.obs(S,a,xi,f),rtol=0,atol=0)
    torch.testing.assert_close(new.critic_features(S,f,xi),old.critic_features(S,f,xi),rtol=0,atol=0)
    before=[x.clone() for x in cached]
    new.reward(S,f,a,a,ref_bodies=cached)
    for x,y in zip(before,cached):torch.testing.assert_close(x,y,rtol=0,atol=0)


def test_sole_positions_and_velocities_match_mujoco_point_jacobian():
    new,_,S,f,a,cached=sample()
    points=new.tracking_points.apply(*cached)
    m=mujoco.MjModel.from_xml_path(MJCF_PATH);d=mujoco.MjData(m)
    p,R,v,w,q,qd=(x[0] for x in S)
    d.qpos[:3]=p.numpy();d.qpos[3:7]=mat_to_quat(R).numpy()
    d.qvel[:3]=v.numpy();d.qvel[3:6]=(R.T@w).numpy()
    for i,name in enumerate(JOINT_NAMES):
        j=m.joint(name).id
        d.qpos[m.jnt_qposadr[j]]=q[i].item();d.qvel[m.jnt_dofadr[j]]=qd[i].item()
    mujoco.mj_forward(m,d)
    tracked=[new.kin.body_names[i] for i in new.kin.tracked_idx]
    for name,offset in new.cfg.body_point_offsets.items():
        i=tracked.index(name);bid=m.body(name).id
        target=d.xpos[bid]+d.xmat[bid].reshape(3,3)@np.array(offset)
        jp,jr=np.zeros((3,m.nv)),np.zeros((3,m.nv))
        mujoco.mj_jac(m,d,jp,jr,target,bid)
        np.testing.assert_allclose(points[0][0,i].numpy(),target,rtol=0,atol=3e-6)
        np.testing.assert_allclose(points[2][0,i].numpy(),jp@d.qvel,rtol=0,atol=3e-6)
    keep=[i for i,n in enumerate(tracked) if n not in new.cfg.body_point_offsets]
    for a,b in zip(points,cached):
        dim=-3 if a.shape[-2:]==(3,3) else -2
        ix=torch.tensor(keep)
        torch.testing.assert_close(a.index_select(dim,ix),b.index_select(dim,ix),rtol=0,atol=0)


def test_ankle_pitch_has_sole_position_gradient_and_orientation_is_retained():
    new,old,S,f,a,cached=sample()
    state=list(S);q=S[4].clone()
    j=JOINT_NAMES.index('ankle_pitch_l_joint');q[:,j]+=.4;q.requires_grad_();state[4]=q
    _,nt=new.reward(state,f,a,a,terms=True)
    _,ot=old.reward(state,f,a,a,terms=True)
    assert (nt['e_body_pos']>ot['e_body_pos']+1e-6).all()
    torch.testing.assert_close(nt['reward_body_ori'],ot['reward_body_ori'],rtol=0,atol=0)
    grad=torch.autograd.grad(nt['reward_body_pos'].sum(),q)[0]
    assert torch.isfinite(grad).all() and (grad[:,j]<0).all()


def test_cached_multi_step_reward_and_compiled_gradients():
    new,_,S,f,a,cached=sample()
    state=[x.unsqueeze(0).repeat((2,)+(1,)*x.ndim).requires_grad_() for x in S]
    state[4]=(state[4]+.07).detach().requires_grad_()
    ff=f.repeat(2,1);aa=a.repeat(2,1,1).requires_grad_()
    refs=tuple(x.repeat((2,)+(1,)*x.ndim) for x in cached)
    before=tuple(x.clone() for x in refs)
    r=new.reward(state,ff,aa,aa,ref_bodies=refs)
    torch.testing.assert_close(r,new.reward(state,ff,aa,aa),rtol=0,atol=0)
    compiled=torch.compile(new.reward,backend='aot_eager',fullgraph=True)
    cr=compiled(state,ff,aa,aa,ref_bodies=refs)
    torch.testing.assert_close(r,cr,rtol=0,atol=0)
    rg=torch.autograd.grad(r.sum(),state,retain_graph=True,allow_unused=True)
    cg=torch.autograd.grad(cr.sum(),state,allow_unused=True)
    for x,y in zip(rg,cg):
        if x is None:assert y is None
        else:
            assert torch.isfinite(x).all()
            torch.testing.assert_close(x,y,rtol=0,atol=0)
    for x,y in zip(before,refs):torch.testing.assert_close(x,y,rtol=0,atol=0)


def test_relative_height_plus_sole_offsets_is_translation_invariant_for_body_reward():
    new,_,S,f,a,cached=sample()
    r,initial=new.reward(S,f,a,a,terms=True)
    new.motion.root_p[:,2]+=.12;cached[0][...,2]+=.12
    _,shifted=new.reward(S,f,a,a,terms=True)
    torch.testing.assert_close(shifted['reward_body_pos'],initial['reward_body_pos'],atol=1e-9,rtol=0)
    torch.testing.assert_close(shifted['reward_body_linvel'],initial['reward_body_linvel'],rtol=0,atol=0)
    torch.testing.assert_close(shifted['reward_root_pos'],torch.full((4,),.005*np.exp(-.1*.12**2/.3**2)))
    assert (shifted['e_body_pos_ref_z']>.01).all()
    assert (shifted['e_body_z_relative']<1e-12).all()


def test_new_and_old_checkpoint_objectives_are_unambiguous():
    args=build_parser().parse_args(['--config','tiangong3_full_wbc'])
    cfg=resolve_reward_config(args)
    saved=json.loads(json.dumps(vars(args)))
    assert checkpoint_reward_cfg(saved)==cfg
    resume=build_parser().parse_args(['--robot','tiangong3','--dynamics','dexevt'])
    assert resolve_reward_config(resume,{'args':saved})==cfg
    assert config_snapshot(resume)['env']['rewards']['body_point_offsets']==cfg.body_point_offsets
    old={'dynamics':'dexevt','reward_cfg':{'w_joint_acc':-3e-6}}
    restored=checkpoint_reward_cfg(old)
    assert restored.body_point_offsets=={} and restored.root_pos_z_weight==1
    assert restored.body_pos_z_mode=='reference'
    relz={**old,'root_pos_z_weight':.1,'body_pos_z_mode':'relative_root'}
    assert checkpoint_reward_cfg(relz).root_pos_z_weight==.1
    assert checkpoint_reward_cfg(relz).body_point_offsets=={}
    with pytest.raises(ValueError,match='differ from checkpoint'):
        resolve_reward_config(args,{'args':old})


def test_unknown_point_link_rejected():
    new,_,_,_,_,_=sample()
    with pytest.raises(ValueError,match='require tracked links'):
        MDP(new.kin,new.motion,new.default_q,replace(new.cfg,body_point_offsets={'typo':(.1,0,0)}))
