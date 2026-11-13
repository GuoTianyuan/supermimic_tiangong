"""Actual reference-slot semantics and legacy compatibility, without simulator initialization."""
from types import SimpleNamespace

import pytest
import torch

from supermimic.configs.train import build_parser
from supermimic.motions.library import MotionLib
from supermimic.motions.pool import MotionPool
from supermimic.motions.timing import configure_reference_timing, checkpoint_goal_stride, goal_offsets


def test_new_run_and_legacy_resume():
    new = build_parser().parse_args([])
    configure_reference_timing(new)
    assert new.goal_stride == 5
    assert new.goal_offsets == [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    for saved, expected in [({}, 1), ({"goal_stride": 5}, 5)]:
        resumed = build_parser().parse_args([])
        configure_reference_timing(resumed, {"args": saved})
        assert resumed.goal_stride == expected
        assert checkpoint_goal_stride(saved) == expected
    with pytest.raises(ValueError, match="Cannot resume"):
        configure_reference_timing(SimpleNamespace(goal_frames=10, goal_stride=5), {"args": {}})
    with pytest.raises(ValueError, match="Cannot resume"):
        configure_reference_timing(SimpleNamespace(goal_frames=10, goal_stride=1), {"args": {"goal_stride": 5}})


@pytest.mark.parametrize("bad", [0, -1, 2.5, True, None])
def test_invalid_timing(bad):
    with pytest.raises(ValueError):
        goal_offsets(10, bad)
    with pytest.raises(ValueError):
        checkpoint_goal_stride({"goal_stride": bad})


@pytest.mark.parametrize("cls", [MotionLib, MotionPool])
def test_sparse_goal_and_clip_boundaries(cls):
    # Two adjacent clips with distinct frame-valued joint poses expose cross-clip indexing errors.
    motion = object.__new__(cls)
    motion.clip_id = torch.tensor([0] * 60 + [1] * 12)
    motion.clip_end = torch.tensor([59, 71])
    motion.root_R = torch.eye(3).repeat(72, 1, 1)
    motion.root_p = torch.zeros(72, 3)
    motion.root_p[:, 0] = torch.arange(72) * .01
    motion.root_v_local = torch.ones(72, 3)
    motion.root_w_local = torch.zeros(72, 3)
    motion.q = torch.arange(72).float()[:, None].repeat(1, 29)
    motion.qd = torch.zeros(72, 29)
    f = torch.tensor([0, 55, 59, 60, 71])
    # Backward-compatible standalone loader defaults, exactly the old index expression.
    legacy = torch.minimum(f[:, None] + torch.arange(1, 11), motion.clip_end[motion.clip_id[f]][:, None])
    assert torch.equal(motion.future_frames(f, 10), legacy)
    motion.goal_stride = 5
    expected = torch.tensor([[5,10,15,20,25,30,35,40,45,50], [59]*10, [59]*10,
                             [65,70,71,71,71,71,71,71,71,71], [71]*10])
    assert torch.equal(motion.future_frames(f, 10), expected)
    goal = motion.goal(f, 10).reshape(5, 10, 73)
    assert torch.equal(goal[:, :, 15], expected.float())  # first joint of every slot
    torch.testing.assert_close(goal[:, :, 0], (expected - f[:, None]) * .01)
    assert motion.goal(f, 10, ("root_pos",)).shape == (5, 700)
    # Advancing the simulation still advances by one control frame, not five.
    assert motion.future_frames(torch.tensor([1]), 10)[0, 0].item() == 6
