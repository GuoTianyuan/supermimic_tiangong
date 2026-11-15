"""Actual reference-slot semantics and legacy compatibility, without simulator initialization."""
from types import SimpleNamespace

import pytest
import torch

from supermimic.configs.train import build_parser
from supermimic.motions.library import MotionLib
from supermimic.motions.pool import MotionPool
from supermimic.motions.timing import configure_reference_timing, checkpoint_goal_start, checkpoint_goal_stride, goal_offsets


def test_new_run_and_legacy_resume():
    new = build_parser().parse_args([])
    configure_reference_timing(new)
    assert new.goal_stride == 5
    assert new.goal_start == 1
    assert new.goal_offsets == [1, 6, 11, 16, 21, 26, 31, 36, 41, 46]
    for saved, expected in [({}, 1), ({"goal_stride": 5}, 5)]:
        resumed = build_parser().parse_args([])
        configure_reference_timing(resumed, {"args": saved})
        assert resumed.goal_stride == expected
        assert resumed.goal_start == expected
        assert resumed.goal_offsets == list(range(expected, 11 * expected, expected))
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


@pytest.mark.parametrize("start", [None, 1, 5])
@pytest.mark.parametrize("cls", [MotionLib, MotionPool])
def test_sparse_goal_and_clip_boundaries(cls, start):
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
    motion.goal_start = start
    expected = (torch.tensor([[1,6,11,16,21,26,31,36,41,46], [56,59,59,59,59,59,59,59,59,59], [59]*10,
                              [61,66,71,71,71,71,71,71,71,71], [71]*10]) if start == 1 else
                torch.tensor([[5,10,15,20,25,30,35,40,45,50], [59]*10, [59]*10,
                              [65,70,71,71,71,71,71,71,71,71], [71]*10]))
    assert torch.equal(motion.future_frames(f, 10), expected)
    goal = motion.goal(f, 10).reshape(5, 10, 73)
    assert torch.equal(goal[:, :, 15], expected.float())  # first joint of every slot
    torch.testing.assert_close(goal[:, :, 0], (expected - f[:, None]) * .01)
    assert motion.goal(f, 10, ("root_pos",)).shape == (5, 700)
    # Advancing the simulation still advances by one control frame, not five.
    assert motion.future_frames(torch.tensor([1]), 10)[0, 0].item() == (2 if start == 1 else 6)


def test_explicit_resume_migration_and_next_resume():
    import copy
    checkpoint = {"args": {"goal_frames": 10, "goal_stride": 5}, "iteration": 63242,
                  "pi": {"normalizer": torch.tensor([1., 2.])}, "opt_pi": {"step": 123}}
    before = copy.deepcopy(checkpoint)
    args = build_parser().parse_args(["--goal_start", "1", "--resume", "before.pt"])
    with pytest.raises(ValueError, match="Cannot resume"):
        configure_reference_timing(args, checkpoint)
    args.allow_reference_timing_change = True
    configure_reference_timing(args, checkpoint)
    assert args.goal_offsets == list(range(1, 47, 5))
    change = args.reference_timing_changes[0]
    assert change["from_offsets"] == list(range(5, 51, 5))
    assert change["to_offsets"] == args.goal_offsets
    assert change["checkpoint_iteration"] == 63242
    assert change["checkpoint_path"] == "before.pt"
    assert checkpoint["args"] == before["args"]
    assert checkpoint["opt_pi"] == before["opt_pi"]
    assert torch.equal(checkpoint["pi"]["normalizer"], before["pi"]["normalizer"])
    # A later ordinary resume retains the new timing without carrying the override flag.
    resumed = build_parser().parse_args([])
    configure_reference_timing(resumed, {"args": vars(args), "iteration": 64000})
    assert resumed.goal_offsets == args.goal_offsets
    assert resumed.reference_timing_changes == args.reference_timing_changes
    assert not resumed.allow_reference_timing_change
    assert checkpoint_goal_start(vars(args)) == 1


def test_timing_override_cannot_change_network_dimensions():
    args = build_parser().parse_args(["--goal_frames", "9", "--allow_reference_timing_change"])
    with pytest.raises(ValueError, match="input dimensions"):
        configure_reference_timing(args, {"args": {"goal_frames": 10, "goal_stride": 5}})


@pytest.mark.parametrize("bad", [0, -1, 2.5, True])
def test_invalid_start(bad):
    with pytest.raises(ValueError):
        checkpoint_goal_start({"goal_start": bad})
    with pytest.raises(ValueError):
        goal_offsets(10, 5, bad)
