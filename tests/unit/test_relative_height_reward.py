"""Reward ablation invariants using the robot's real differentiable kinematics."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from supermimic.configs.train import build_parser
from supermimic.robots import default_joint_pos_list, set_dynamics
from supermimic.robots.kinematics import G1Kinematics
from supermimic.tasks.tracking.config import (
    RewardCfg, configure_reward, checkpoint_reward_cfg, reward_cfg,
)
from supermimic.tasks.tracking.mdp import MDP, MAX_STEP_REWARD
from supermimic.motions.library import CONTROL_DT
from supermimic.utils.metrics import RealMetrics


@pytest.fixture
def example():
    torch.manual_seed(84)
    set_dynamics('dexevt')
    kin = G1Kinematics('cpu')
    q0 = torch.tensor(default_joint_pos_list())
    n = 4
    p = torch.tensor([.1, -.2, .9]).repeat(n, 1).requires_grad_()
    R = torch.eye(3).repeat(n, 1, 1)
    v, w = [torch.randn(n, 3) * .1 for _ in range(2)]
    q = (q0.repeat(n, 1) + torch.randn(n, 29) * .03).requires_grad_()
    qd = (torch.randn(n, 29) * .1).requires_grad_()
    S = (p, R, v, w, q, qd)
    bp, bR, bv, bw = [x.detach().clone() for x in kin.forward(*S)]
    motion = SimpleNamespace(root_p=p.detach().clone(), root_R=R.clone(), q=q.detach().clone(),
                             qd=qd.detach().clone(), body_p=bp, body_R=bR, body_v=bv, body_w=bw)
    def bodies(f, positions_only=False):
        return ((motion.body_p[f], None, None, None) if positions_only else
                (motion.body_p[f], motion.body_R[f], motion.body_v[f], motion.body_w[f]))
    motion.body_reference = bodies
    old = MDP(kin, motion, q0, reward_cfg('dexevt'))
    new = MDP(kin, motion, q0, replace(old.cfg, root_pos_z_weight=.1, body_pos_z_mode='relative_root'))
    a = torch.zeros(n, 29, requires_grad=True)
    return old, new, motion, S, torch.arange(n), a


def test_default_equivalence_and_perfect_reward(example):
    old, new, motion, S, f, a = example
    for mdp in (old, new):
        r = mdp.reward(S, f, a, a, qd_prev=S[5])
        torch.testing.assert_close(r, torch.full_like(r, MAX_STEP_REWARD))
    # Exercise nonzero errors; new code's default must match the previous formula.
    motion.root_p.add_(torch.tensor([.05, -.07, .12]))
    motion.body_p.add_(torch.tensor([.05, -.07, .12]))
    _, terms = old.reward(S, f, a, a, terms=True, qd_prev=S[5])
    e_rp = (motion.root_p-S[0]).square().sum(-1)
    torch.testing.assert_close(terms['reward_root_pos'], .5*torch.exp(-e_rp/.3**2)*CONTROL_DT, rtol=0, atol=0)
    torch.testing.assert_close(terms['e_body_pos'], terms['e_body_pos_ref_z'])


def test_global_height_offset_relative_links_and_weak_root(example):
    old, new, m, S, f, a = example
    dz = .1
    m.root_p[:, 2] += dz
    m.body_p[:, :, 2] += dz
    _, t_old = old.reward(S, f, a, a, terms=True, qd_prev=S[5])
    _, t_new = new.reward(S, f, a, a, terms=True, qd_prev=S[5])
    torch.testing.assert_close(t_new['e_body_pos'], torch.zeros(4), atol=1e-12, rtol=0)
    torch.testing.assert_close(t_old['e_body_pos'], torch.full((4,), dz**2))
    expected = .5*torch.exp(torch.tensor(-.1*dz**2/.3**2))*CONTROL_DT
    torch.testing.assert_close(t_new['reward_root_pos'], expected.expand(4))
    assert (t_new['reward_root_pos'] < .5*CONTROL_DT).all()
    assert (t_new['reward_root_pos'] > t_old['reward_root_pos']).all()
    torch.testing.assert_close(t_new['e_root_pos'], t_old['e_root_pos'])
    torch.testing.assert_close(t_new['e_body_pos_ref_z'], t_old['e_body_pos'])
    for key in ('reward_body_linvel', 'reward_body_angvel', 'reward_body_ori', 'reward_root_ori',
                'pen_action_rate', 'pen_joint_acc', 'pen_joint_limit'):
        torch.testing.assert_close(t_new[key], t_old[key], rtol=0, atol=0)


def test_relative_articulation_still_matters(example):
    _, new, m, S, f, a = example
    # Raising one wrist relative to pelvis must still lower body-position reward.
    m.body_p[:, -1, 2] += .1
    _, terms = new.reward(S, f, a, a, terms=True)
    torch.testing.assert_close(terms['e_body_pos'], torch.full((4,), .1**2 / m.body_p.shape[1]))
    torch.testing.assert_close(terms['e_body_z_relative'], terms['e_body_pos'])
    assert (terms['reward_body_pos'] < CONTROL_DT).all()


def test_xy_tracking_and_gradients_remain(example):
    old, new, m, S, f, a = example
    m.root_p += torch.tensor([.08, -.06, .1])
    m.body_p += torch.tensor([.08, -.06, .1])
    _, terms = new.reward(S, f, a, a, terms=True)
    expected = .5*torch.exp(torch.tensor(-(.08**2+.06**2+.1*.1**2)/.3**2))*CONTROL_DT
    torch.testing.assert_close(terms['reward_root_pos'], expected.expand(4))
    grad = torch.autograd.grad(terms['reward_root_pos'].sum(), S[0], retain_graph=True)[0]
    assert torch.isfinite(grad).all() and (grad[:, 2] > 0).all()
    bg = torch.autograd.grad(terms['reward_body_pos'].sum(), S[0], retain_graph=True)[0]
    torch.testing.assert_close(bg, torch.zeros_like(bg), atol=1e-10, rtol=0)
    m.body_v += .1
    r = new.reward(S, f, a+.1, a, qd_prev=S[5]-.01)
    grads = torch.autograd.grad(r.sum(), (S[4], S[5], a))
    assert all(torch.isfinite(g).all() for g in grads)
    assert grads[1].abs().sum() > 0
    # Cached imagination references use exactly the same reward.
    torch.testing.assert_close(r, new.reward(S, f, a+.1, a, qd_prev=S[5]-.01, ref_bodies=m.body_reference(f)))


def test_termination_unchanged(example):
    old, new, m, S, f, a = example
    m.root_p[:, 2] += torch.tensor([.0, .1, .24, .3])
    torch.testing.assert_close(old.termination(S, f), new.termination(S, f))
    torch.testing.assert_close(old.soft_termination(S, f), new.soft_termination(S, f))
    assert new.termination(S, f).tolist() == [False, False, False, True]


def test_checkpoint_resume_and_legacy():
    a = build_parser().parse_args(['--dynamics', 'dexevt', '--root_pos_z_weight', '.1', '--body_pos_z_mode', 'relative_root'])
    cfg = configure_reward(a)
    resumed = build_parser().parse_args(['--dynamics', 'dexevt'])
    assert configure_reward(resumed, {'args': vars(a)}) == cfg
    assert checkpoint_reward_cfg(vars(a)) == cfg
    legacy = {'dynamics': 'dexevt', 'reward_cfg': {'w_joint_acc': -3e-6}}
    assert checkpoint_reward_cfg(legacy) == reward_cfg('dexevt')
    with pytest.raises(ValueError, match='Cannot resume'):
        configure_reward(a, {'args': legacy})
    for weight in (-1, float('nan'), float('inf')):
        with pytest.raises(ValueError):
            RewardCfg(root_pos_z_weight=weight)


def test_metrics_keep_absolute_and_relative_height(example):
    _, new, m, S, f, a = example
    m.root_p[:, 2] += .1
    m.body_p[:, :, 2] += .1
    r, terms = new.reward(S, f, a, a, terms=True, qd_prev=S[5])
    metrics = RealMetrics('cpu')
    metrics.update({'r': r, 'terms': terms})
    out = metrics.flush()
    assert out['error_root_z_m'] == pytest.approx(.1, abs=1e-6)
    assert out['error_body_pos_ref_z_m'] == pytest.approx(.1, abs=1e-6)
    assert out['error_body_z_relative_m'] < 1e-6
