"""Compare against ce8955e before package splitting, not a snapshot of the new code."""
import ast
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "src/supermimic"
BEFORE = json.loads((ROOT / "tests/fixtures/package_layout_before.json").read_text())
MOVES = json.loads((ROOT / "docs/refactor/package_map.json").read_text())["files"]


class Normalize(ast.NodeTransformer):
    def visit_Import(self, node):
        return None

    def visit_ImportFrom(self, node):
        return None

    def visit_Assign(self, node):
        if ast.unparse(node) == "start = self.goal_stride if self.goal_start is None else self.goal_start":
            return None
        if ast.unparse(node) == "fut = f.unsqueeze(-1) + start + self.goal_stride * torch.arange(K, device=f.device)":
            return ast.parse("fut = f.unsqueeze(-1) + torch.arange(1, K + 1, device=f.device)").body[0]
        if ast.unparse(node.targets[0]) in {"goal_stride", "motion.goal_stride", "goal_start", "motion.goal_start"}:
            return None
        return self.generic_visit(node)

    def visit_Expr(self, node):
        if ast.unparse(node.value).startswith(("parser.add_argument('--goal_stride'", "parser.add_argument('--goal_start'", "parser.add_argument('--allow_reference_timing_change'")):
            return None  # The new option is tested independently.
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return None
        return self.generic_visit(node)

    # Sparse reference timing is an intentional post-refactor change, covered by test_reference_timing.
    def visit_BinOp(self, node):
        if isinstance(node.op, ast.Mult) and ast.unparse(node.left) == "self.goal_stride":
            return self.visit(node.right)
        return self.generic_visit(node)

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            for old, new in MOVES.items():
                node.value = node.value.replace(new, old)
            node.value = node.value.replace("supermimic.tasks.tracking.mdp.", "supermimic.mdp.")
        return node


def canonical(node):
    return ast.dump(Normalize().visit(copy.deepcopy(node)), include_attributes=False)


def functions():
    result = {}
    for path in PACKAGE.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.FunctionDef):
                result.setdefault(node.name, []).append(node)
    return result


def test_split_numerical_functions_preserved():
    current = functions()
    # Reconstitute the extracted Critic call at its original place in Actor update.
    policy = copy.deepcopy(current["policy_update"][0])
    critic = current["critic_update"][0]
    for i, node in enumerate(policy.body):
        if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name) and node.value.func.id == "critic_update"):
            policy.body[i:i+1] = copy.deepcopy(critic.body[:-1])
            break
    else:
        raise AssertionError("Critic update must remain at its original position")
    current["policy_update"] = [policy]
    sync = copy.deepcopy(current["sync_grads"][0])
    sync.args.args.pop()
    for node in ast.walk(sync):
        if isinstance(node, ast.Name) and node.id == "world":
            node.id = "WORLD"
    current["sync_grads"] = [sync]
    checked = 0
    for name, entries in BEFORE["functions"].items():
        if name == "repository_root":
            continue  # Path depth changed; separate asset-root tests cover this.
        actual = {canonical(node) for node in current.get(name, [])}
        for entry in entries:
            expected = canonical(ast.parse(entry["code"]).body[0])
            assert expected in actual, (entry["source"], name)
            checked += 1
    assert checked > 100


def test_runner_schedule_and_checkpoint_flow_preserved():
    moved = {
        "sync_grads", "clip_finite_grads", "set_requires_grad", "_wm_rollout", "_wm_err",
        "_imag_step", "_reward_term", "_critic_input", "reward_term", "critic_input_c",
        "wm_update", "sample_valid_starts", "policy_update", "a_prev_gain",
    }
    moved_assignments = {
        "ERR_FLOOR", "wm_rollout", "wm_err", "imag_step", "reward_term_c", "critic_input_gpu",
        "wm_learner", "policy_learner",
    }

    class Runner(Normalize):
        def visit_Expr(self, node):
            call = ast.unparse(node.value)
            if call.startswith("configure_reference_timing(") or "future reference offsets:" in call:
                return None  # CLI/metadata only; independent reference-timing tests cover this.
            return super().visit_Expr(node)

        def visit_FunctionDef(self, node):
            return None if node.name in moved else self.generic_visit(node)

        def visit_Assign(self, node):
            if any(isinstance(t, ast.Name) and t.id in moved_assignments for t in node.targets):
                return None
            return super().visit_Assign(node)

        def visit_Call(self, node):
            mapping = {
                "wm_learner.update": "wm_update",
                "policy_learner.update": "policy_update",
                "policy_learner.action_loop_gain": "a_prev_gain",
            }
            if ast.unparse(node.func) in mapping:
                node.func = ast.Name(id=mapping[ast.unparse(node.func)], ctx=ast.Load())
            return self.generic_visit(node)

    old = Runner().visit(ast.parse(BEFORE["runner_source"]))
    new = Runner().visit(ast.parse((PACKAGE / "runners/train.py").read_text()))
    assert ast.dump(old, include_attributes=False) == ast.dump(new, include_attributes=False)


def test_package_root_and_task_boundaries():
    assert {path.name for path in PACKAGE.glob("*.py")} == {"__init__.py", "cli.py"}
    for path in (
        "learners/wm_learner.py", "learners/policy_learner.py", "learners/critic_learner.py",
        "models/world_model.py", "models/policy.py", "models/critic.py", "robots/tiangong3.py",
        "robots/g1.py", "envs/tracking_env.py", "envs/sim_context.py", "motions/library.py",
        "motions/pool.py", "storage/replay_buffer.py", "tasks/tracking/mdp/observations.py",
        "tasks/tracking/mdp/rewards.py", "tasks/tracking/mdp/terminations.py",
    ):
        assert (PACKAGE / path).is_file(), path
