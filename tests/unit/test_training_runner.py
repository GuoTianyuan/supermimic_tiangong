"""Compare moved code to the actual pre-refactor runner, not to rewritten fixtures."""
import ast
import copy
import os
from pathlib import Path
import subprocess
import sys
from config_refactor_compat import ConfigDefaults

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / 'src/supermimic/runners'
OLD = ast.parse((ROOT / 'tests/fixtures/runner_before_logging.py.txt').read_text())


def node(source, name):
    return next(n for n in ast.walk(source) if isinstance(n, ast.FunctionDef) and n.name == name)


def tree(name):
    return ast.parse((PKG / name).read_text())


def dump(nodes):
    class IgnoreDocstrings(ast.NodeTransformer):
        def visit_Expr(self, n):
            if isinstance(n.value, ast.Constant) and isinstance(n.value.value, str):
                return None
            return self.generic_visit(n)
    tree=IgnoreDocstrings().visit(copy.deepcopy(ast.Module(body=nodes, type_ignores=[])))
    return ast.dump(tree, include_attributes=False)


def test_setup_initialization_and_restore_order_exact():
    old = OLD.body
    begin = next(i for i,n in enumerate(old) if ast.unparse(n).startswith('args.motion_dir ='))
    end = next(i for i,n in enumerate(old) if ast.unparse(n) == 'del ck')
    new = node(tree('setup.py'), 'build_training').body
    # Compare network/RNG/restore order after removing only config serialization.
    class NormalizeSetup(ConfigDefaults):
        def visit_Assign(self, n):
            target = ast.unparse(n.targets[0])
            if target in {'G_MAX', 'args.max_step_reward', 'args.training_config', 'args.reward_cfg'}:
                return None
            if target in {'task_reward_cfg', 'dr_ranges'}:
                return None
            return super().visit_Assign(n)
        def visit_Name(self, n):
            if n.id == 'task_reward_cfg':
                return ast.parse('reward_cfg(args.dynamics)', mode='eval').body
            return n
        def visit_Expr(self, n):
            if ast.unparse(n).startswith('json.dump(args.training_config,'):
                return None
            return self.generic_visit(n)
    def normalize(items):
        return NormalizeSetup().visit(ast.Module(body=copy.deepcopy(items),type_ignores=[])).body
    assert dump(normalize(old[begin:end+1])) == dump(normalize(new[1:-1]))


def test_collection_and_rng_order_exact():
    old = copy.deepcopy(node(OLD, 'collect'))
    new = copy.deepcopy(node(tree('collector.py'), 'collect'))
    new.body = new.body[4:]  # explicit dependency aliases
    old.body = old.body[1:]  # old global declaration
    class Normalize(ConfigDefaults):
        def visit_Attribute(self, n):
            if isinstance(n.value, ast.Name) and n.value.id == 'self' and n.attr in {'obs_hist','sim_steps','real_metrics'}:
                return ast.Name(id=n.attr, ctx=n.ctx)
            return super().visit_Attribute(n)
        def visit_Expr(self, n):
            if ast.unparse(n).startswith('self.episode_metrics.update('):
                return None  # detached diagnostic only, numerically tested separately
            return self.generic_visit(n)
    new=Normalize().visit(new)
    assert dump(old.body) == dump(new.body)


def test_checkpoint_payload_and_pool_collectives_exact():
    old=node(OLD,'save_checkpoint').body
    new=node(tree('checkpoints.py'),'save_checkpoint').body
    new=next(new[i:] for i,n in enumerate(new) if ast.unparse(n).startswith('path ='))
    assert dump(old) == dump(new)


def test_update_schedule_and_refresh_exact():
    def schedule(source):
        loop=copy.deepcopy(next(n for n in ast.walk(source) if isinstance(n,ast.For) and ast.unparse(n.iter)=='range(start_it, args.iterations)'))
        # Logging is a separate independently tested reducer. Keep the checkpoint trigger below it.
        loop.body=[n for n in loop.body if not (isinstance(n,ast.If) and ast.unparse(n.test)=='done_it % args.log_every == 0')]
        return loop
    class Normalize(ast.NodeTransformer):
        def visit_Call(self,n):
            s=ast.unparse(n.func)
            if s=='stop.requested':n.func=ast.Name(id='stop_requested',ctx=ast.Load())
            if s=='collector.collect':n.func=ast.Name(id='collect',ctx=ast.Load())
            if s=='learning_rate_at':
                n.func=ast.Name(id='lr_at',ctx=ast.Load());n.args=n.args[1:]
            if s=='save_checkpoint':n.keywords=[]
            return self.generic_visit(n)
    assert dump([schedule(OLD)]) == dump([Normalize().visit(schedule(tree('supermimic_runner.py')))])


def test_lr_schedule_exact():
    old=node(OLD,'lr_at');new=node(tree('supermimic_runner.py'),'learning_rate_at')
    assert dump(old.body)==dump(new.body)


def test_cli_import_does_not_parse_or_load_simulator():
    code='import sys; import supermimic.runners.train; import supermimic.cli; assert "torch" not in sys.modules; assert "isaaclab" not in sys.modules'
    subprocess.run([sys.executable,'-c',code],check=True,env={**os.environ,'PYTHONPATH':str(ROOT/'src')})
