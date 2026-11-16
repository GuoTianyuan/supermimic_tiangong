"""Central DR declarations must reproduce existing draws and reach the real sampler."""
import ast
import json
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from supermimic.configs.experiment import (
    config_snapshot, resolve_randomization_config, saved_randomization_ranges, requested_randomization_ranges,
)
from supermimic.configs.train import build_parser
from supermimic.tasks.tracking.tracking_env_cfg import RandomizationRangesCfg, randomization_ranges
from supermimic.envs.domain_randomization import DRCfg
from supermimic.utils.rotations import so3_exp

ROOT = Path(__file__).resolve().parents[2]
BEFORE = json.loads((ROOT/'tests/fixtures/package_layout_before.json').read_text())['functions']
CURRENT = ast.parse((ROOT/'src/supermimic/envs/tracking_env.py').read_text())


def function(name, old=False):
    if old:
        return ast.parse(next(e['code'] for e in BEFORE[name] if e['source']=='env.py')).body[0]
    cls = next(n for n in CURRENT.body if isinstance(n,ast.ClassDef) and n.name=='TrackingSim')
    return next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name)


def run_block(node, scope):
    exec(compile(ast.Module(body=[node],type_ignores=[]), '<real sampler block>', 'exec'), scope)


def test_dynamics_and_robot_ranges_and_runtime_layout_unchanged():
    expected = dict(kp_scale=(.75,1.25),kd_scale=(.75,1.25),push_interval_s=(1.,5.),
                    push_lin_vel=(.5,.5,.2),push_ang_vel=(.52,.52,.78),mass_scale=(.85,1.15),
                    armature=(.002,.06),ankle_armature=(.01,.06),joint_friction=(.001,.6),
                    viscous_friction=False,max_delay=4,contact_soft=(.02,.9,.95),contact_hard=(.01,.99,.99))
    for robot,friction in [('g1',(.3,1.2)),('tiangong3',(.6,1.6))]:
        for dynamics in ('default','dexevt'):
            ranges = randomization_ranges(robot,dynamics)
            old = {**expected,'ankle_armature':(.002,.06),'viscous_friction':True} if dynamics=='dexevt' else expected
            assert {k:getattr(ranges,k) for k in old} == old
            assert ranges.friction == friction
            c=DRCfg.for_dynamics(frozenset({'pd','contact'}), dynamics, ranges=ranges)
            assert all(getattr(c,k)==v for k,v in asdict(ranges).items())
    assert DRCfg().xi_dim == 33
    assert DRCfg(frozenset({'contact'})).xi_dim == DRCfg(frozenset({'pd'})).xi_dim+1


def test_config_preview_and_checkpoint_roundtrip_and_override_conflict():
    args=build_parser().parse_args(['--config','tiangong3_full_wbc'])
    snap=config_snapshot(args)
    assert snap['env']['events']['ranges']['friction']==(.6,1.6)
    assert snap['env']['events']['ranges']['viscous_friction'] is True
    ranges=resolve_randomization_config(args)
    saved=json.loads(json.dumps(vars(args)))
    assert saved_randomization_ranges(saved)==ranges
    saved['dr_ranges']['kp_scale']=[.8,1.2]
    resumed=build_parser().parse_args(['--robot','tiangong3','--dynamics','dexevt'])
    result=resolve_randomization_config(resumed,{'args':saved})
    assert result.kp_scale==(.8,1.2)
    assert config_snapshot(resumed)['env']['events']['ranges']['kp_scale']==(.8,1.2)
    with pytest.raises(ValueError,match='differ from checkpoint'):
        resolve_randomization_config(args,{'args':saved})
    assert saved_randomization_ranges({'robot':'tiangong3','dynamics':'dexevt'})==ranges
    with pytest.raises(ValueError,match='Unknown DR'):
        saved_randomization_ranges({'dr_ranges':{'typo':1}})


@pytest.mark.parametrize('field,value',[('max_delay',0),('max_delay',5),('kp_scale',(1.,1.)),
                                       ('torso_com',(-.1,.1,.1)),('contact_hard',(.001,.99,.99)),
                                       ('joint_offset',(0.,float('nan')))])
def test_invalid_ranges_fail_before_simulator(field,value):
    with pytest.raises(ValueError): replace(RandomizationRangesCfg(),**{field:value}).validate()


def base_draws(ranges, old=False, randomize=True):
    n=32; written=[]
    robot=SimpleNamespace(data=SimpleNamespace(body_com_pose_b=SimpleNamespace(torch=torch.zeros(n,1,7))),
                          set_coms_index=lambda **kw: written.append(kw['coms']))
    obj=SimpleNamespace(N=n,device='cpu',xi=torch.zeros(n,33),dr=ranges,robot=robot,torso_id=0,randomize=randomize)
    node=next(x for x in function('__init__',old).body if isinstance(x,ast.If) and ast.unparse(x.test)=='not self.randomize')
    run_block(node,dict(self=obj,torch=torch,dev='cpu',FRICTION_RANGE=(.6,1.6)))
    return obj.xi, written


@pytest.mark.parametrize('randomize',[True,False])
def test_base_draws_exact_values_and_rng_order(randomize):
    ranges=randomization_ranges('tiangong3','dexevt')
    torch.manual_seed(123);a,aw=base_draws(ranges,True,randomize);arng=torch.get_rng_state()
    torch.manual_seed(123);b,bw=base_draws(ranges,False,randomize);brng=torch.get_rng_state()
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert torch.equal(arng,brng) and len(aw)==len(bw)
    if aw: torch.testing.assert_close(aw[0],bw[0],rtol=0,atol=0)
    changed=replace(ranges,friction=(.7,.9),torso_com=(0.,0.,0.))
    x,_=base_draws(changed)
    assert torch.all((x[:,0]>=.7)&(x[:,0]<=.9)) and not x[:,30:].any()


def reset_draws(ranges,old=False):
    n=32
    obj=SimpleNamespace(dr=ranges,reset_noise=True,randomize=True,xi=torch.zeros(n,33))
    scope=dict(self=obj,torch=torch,dev='cpu',n=n,p=torch.zeros(n,3),R=torch.eye(3).repeat(n,1,1),
               v=torch.zeros(n,3),w=torch.zeros(n,3),q=torch.zeros(n,29),so3_exp=so3_exp,env_ids=torch.arange(n))
    fn=function('reset',old)
    for name in ('self.reset_noise','self.randomize'):
        node=next(x for x in fn.body if isinstance(x,ast.If) and ast.unparse(x.test)==name)
        run_block(node,scope)
    return [scope[k] for k in ('p','R','v','w','q')]+[obj.xi]


def test_reset_noise_and_joint_offset_exact_values_and_rng_order():
    ranges=randomization_ranges('tiangong3','dexevt')
    torch.manual_seed(321);old=reset_draws(ranges,True);rng=torch.get_rng_state()
    torch.manual_seed(321);new=reset_draws(ranges)
    assert torch.equal(rng,torch.get_rng_state())
    for x,y in zip(old,new):torch.testing.assert_close(x,y,rtol=0,atol=0)
    changed=replace(ranges,reset_joint_pos=0.,reset_position=(0.,0.,0.),joint_offset=(-.002,.002))
    result=reset_draws(changed)
    assert not result[0].any() and not result[4].any()
    assert result[-1][:,1:30].abs().max()<=.002


def test_robot_and_dynamics_cli_override_range_defaults(monkeypatch):
    args=build_parser().parse_args(['--config','tiangong3_full_wbc','--dynamics','default'])
    cfg=requested_randomization_ranges(args)
    assert cfg.friction==(.6,1.6) and cfg.ankle_armature==(.01,.06) and not cfg.viscous_friction
    args=build_parser().parse_args(['--config','tiangong3_full_wbc','--robot','g1','--dynamics','default','--dr','none'])
    assert requested_randomization_ranges(args)==randomization_ranges('g1')
    from supermimic.configs import experiment
    env,runner=experiment.load_config('tiangong3_full_wbc')
    env.events.ranges=replace(env.events.ranges,kp_scale=(.8,1.2))
    monkeypatch.setattr(experiment,'load_config',lambda name: (env,runner))
    assert requested_randomization_ranges(args).kp_scale==(.8,1.2)
