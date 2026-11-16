"""Task declarations must drive CLI, real/imagined rewards and checkpoint scoring."""
import json
import os
from pathlib import Path
import subprocess
import sys
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from supermimic.configs.experiment import cli_defaults, config_snapshot, load_config, resolve_reward_config
from supermimic.configs.train import build_parser
from supermimic.tasks.tracking.config import from_task_cfg, reward_cfg
from supermimic.tasks.tracking.tracking_env_cfg import RewardsCfg, TerminationsCfg

ROOT = Path(__file__).resolve().parents[2]


def test_legacy_namespace_values_are_unchanged(monkeypatch):
    monkeypatch.setenv('SUPERMIMIC_ROBOT', 'g1')
    # Actual CLI schema captured before the package refactor.
    baseline = json.loads((ROOT/'tests/fixtures/layout_baseline.json').read_text())['cli']
    args = build_parser().parse_args([])
    for entry in baseline:
        if entry['dest'] == 'help':
            continue
        expected = entry['default']
        if isinstance(expected, str):
            expected = expected.replace('<ROOT>', str(ROOT))
        assert getattr(args, entry['dest']) == expected, entry['dest']
    assert args.goal_start is args.goal_stride is None
    assert set(cli_defaults()) == set(vars(args)) - {'config', 'print_config'}


def test_named_preset_overrides_and_no_shared_mutable_instances():
    parser = build_parser()
    a = parser.parse_args(['--num_envs', '128', '--config', 'tiangong3_full_wbc', '--wm_batch', '256'])
    assert (a.robot, a.num_envs, a.wm_batch, a.pi_batch, a.dr) == ('tiangong3', 128, 256, 4096, 'all,contact')
    assert (a.motion_active_ratio, a.motion_replace_ratio, a.horizon, a.ckpt_every) == (.3, .2, 48, 2500)
    assert parser.parse_args([]).wm_batch == 2048  # no preset leak on a reused parser
    env, _ = load_config('tiangong3_full_wbc')
    env.rewards.body_pos.weight = 8
    assert load_config('tiangong3_full_wbc')[0].rewards.body_pos.weight == 1
    snap = config_snapshot(a)
    assert snap['runner']['world_model']['wm_batch'] == 256
    assert snap['env']['scene']['num_envs'] == 128
    assert snap['env']['rewards']['joint_acc']['weight'] == -3e-6
    assert not snap['resolved']


def test_config_print_does_not_import_simulator_or_torch():
    code = ('import sys; from supermimic.runners.train import main; '
            'main(["--config", "tiangong3_full_wbc", "--print-config"]); '
            'assert "torch" not in sys.modules and "isaaclab" not in sys.modules')
    result = subprocess.run([sys.executable, '-c', code], env={**os.environ, 'PYTHONPATH': str(ROOT/'src')},
                            check=True, capture_output=True, text=True)
    assert json.loads(result.stdout)['env']['commands']['motion_dir'].endswith('full_wbc_validated')


def test_old_and_custom_reward_resume_and_json_snapshot():
    args = build_parser().parse_args([])
    args.dynamics = 'dexevt'
    cfg = resolve_reward_config(args)
    assert cfg.w_joint_acc == -3e-6 and cfg.max_step_reward(.01) == .05
    # Legacy checkpoints lack the newly exposed positive weights.
    old = {k:v for k,v in asdict(cfg).items() if not k.startswith(('w_body_', 'w_root_'))}
    restored = resolve_reward_config(args, {'args': {'dynamics': 'dexevt', 'reward_cfg': old}})
    assert asdict(restored) == asdict(cfg)
    saved = asdict(cfg); saved['w_root_pos'] = .25; saved['term_height'] = .3
    restored = resolve_reward_config(args, {'args': {'reward_cfg': saved}})
    assert asdict(restored) == saved
    args.simulation_timing = {'control_dt': .01, 'physics_dt': .0025, 'physics_substeps': 4}
    args.max_step_reward = restored.max_step_reward(.01)
    snap = json.loads(json.dumps(config_snapshot(args)))
    assert snap['resolved'] and snap['env']['rewards']['root_pos']['weight'] == .25
    assert snap['env']['terminations']['term_height'] == .3
    preset = build_parser().parse_args(['--config', 'tiangong3_full_wbc'])
    with pytest.raises(ValueError, match='differ from checkpoint'):
        resolve_reward_config(preset, {'args': {'reward_cfg': saved}})
    with pytest.raises(ValueError, match='another implementation'):
        reward_cfg(saved={'unknown_term': 1})


@pytest.mark.parametrize('field,value', [('body_pos_std', 0), ('w_body_pos', -1), ('w_joint_acc', 1),
                                         ('term_height', float('nan')), ('soft_limit_factor', 1.1)])
def test_invalid_rewards_rejected(field, value):
    with pytest.raises(ValueError):
        reward_cfg(saved={field: value})


def reward_case():
    import torch
    from supermimic.robots.kinematics import G1Kinematics
    from supermimic.tasks.tracking.mdp import MDP
    from supermimic.utils.rotations import quat_to_mat
    torch.manual_seed(214)
    kin = G1Kinematics('cpu')
    B = 8
    p = torch.randn(B, 3) * .02 + torch.tensor([0., 0., .8])
    quat = torch.randn(B, 4); quat /= quat.norm(dim=-1, keepdim=True)
    R = quat_to_mat(quat)
    v,w = torch.randn(B,3)*.1, torch.randn(B,3)*.1
    q = (kin.q_lo + kin.q_hi).unsqueeze(0)/2 + torch.randn(B,29)*.05
    qd = torch.randn(B,29)*.1
    ref = [p, R, v, w, q, qd]
    bodies = kin.forward(*ref)
    motion = SimpleNamespace(root_p=p, root_R=R, q=q, qd=qd,
                             body_reference=lambda f: tuple(x[f] for x in bodies))
    state = [x.detach().clone().requires_grad_() for x in ref]
    state[0] = (state[0] + .03).detach().requires_grad_()
    a = torch.randn(B,29,requires_grad=True)
    prev = torch.randn(B,29)
    return MDP(kin, motion, torch.zeros(29), reward_cfg('dexevt')), state, torch.arange(B), a, prev, qd-.01


def old_reward_function():
    # Original pre-layout implementation, independent of current configuration wiring.
    from supermimic.utils.rotations import rot_angle_sq, yaw_mat
    import torch
    entries = json.loads((ROOT/'tests/fixtures/package_layout_before.json').read_text())['functions']['reward']
    code = next(e['code'] for e in entries if e['source'] == 'mdp.py')
    scope = dict(torch=torch, rot_angle_sq=rot_angle_sq, yaw_mat=yaw_mat, CONTROL_DT=.01)
    exec(compile(code, '<original reward>', 'exec'), scope)
    return scope['reward']


def test_default_reward_and_gradients_exactly_match_original():
    import torch
    mdp,S,f,a,prev,qdprev = reward_case()
    old_fn = old_reward_function()
    for dynamics in ('default', 'dexevt'):
        mdp.cfg = reward_cfg(dynamics)
        for with_terms in (False, True):
            old = old_fn(mdp,S,f,a,prev,terms=with_terms,qd_prev=qdprev)
            new = mdp.reward(S,f,a,prev,terms=with_terms,qd_prev=qdprev)
            if with_terms:
                assert old[1].keys() == new[1].keys()
                for key in old[1]: torch.testing.assert_close(old[1][key],new[1][key],rtol=0,atol=0)
                old,new = old[0],new[0]
            torch.testing.assert_close(old,new,rtol=0,atol=0)
            go = torch.autograd.grad(old.sum(),[*S,a],retain_graph=True,allow_unused=True)
            gn = torch.autograd.grad(new.sum(),[*S,a],retain_graph=True,allow_unused=True)
            for x,y in zip(go,gn):
                if x is None: assert y is None
                else: torch.testing.assert_close(x,y,rtol=0,atol=0)


def test_configured_weights_drive_reward_bound_metrics_and_compiled_path():
    import torch
    from supermimic.tasks.tracking.mdp import MDP
    from supermimic.utils.metrics import RealMetrics
    from supermimic.logging.schema import dashboard_scalars
    mdp,S,f,a,prev,qdprev = reward_case()
    baseline,terms = mdp.reward(S,f,a,prev,terms=True,qd_prev=qdprev)
    rewards = RewardsCfg(); rewards.root_pos.weight = 2.; rewards.joint_acc.weight=-3e-6
    cfg = from_task_cfg(rewards, TerminationsCfg())
    changed = MDP(mdp.kin, mdp.motion, mdp.default_q, cfg)
    actual,t = changed.reward(S,f,a,prev,terms=True,qd_prev=qdprev)
    torch.testing.assert_close(actual,baseline+3*terms['reward_root_pos'])
    assert changed.max_step_reward == .065
    torch.testing.assert_close(actual,sum(t[k] for k in t if k.startswith(('reward_', 'pen_')) and k!='pen_smooth'))
    metrics = RealMetrics('cpu',changed.max_step_reward)
    metrics.update(dict(r=actual,terms=t))
    row = metrics.flush()
    assert row['sim_reward_normalized'] == pytest.approx(actual.double().mean().item()/.065)
    scalars = dashboard_scalars(row,.01,12,6.5)
    assert scalars['Reward/total'] == pytest.approx(actual.double().mean().item()/.01)
    compiled = torch.compile(changed.reward,backend='aot_eager',fullgraph=True)
    result = compiled(S,f,a,prev,qd_prev=qdprev)
    torch.testing.assert_close(result,actual,rtol=0,atol=0)
    g1 = torch.autograd.grad(result.sum(),a,retain_graph=True)[0]
    g2 = torch.autograd.grad(actual.sum(),a,retain_graph=True)[0]
    torch.testing.assert_close(g1,g2,rtol=0,atol=0)
