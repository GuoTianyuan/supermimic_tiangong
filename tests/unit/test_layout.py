"""Refactor contracts: defaults, algorithm AST, script paths and recording timing.

The AST snapshot is from bc_100hz before relocation, not from refactored code.
Simulator integration tests live separately and are not mocked into this suite.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import pytest
from config_refactor_compat import ConfigDefaults

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT/'tests/fixtures/layout_baseline.json'
MOVES = json.loads((ROOT/'docs/refactor/path_map.json').read_text())


def schema(parser):
    def plain(value):
        if isinstance(value,str):return value.replace(str(ROOT),'<ROOT>')
        if isinstance(value,(list,tuple)):return [plain(v) for v in value]
        return value
    return [{k:plain(v) for k,v in dict(dest=a.dest,options=a.option_strings,default=a.default,
             nargs=a.nargs,const=a.const,choices=a.choices,required=a.required,
             type=getattr(a.type,'__name__',None),action=type(a).__name__).items()} for a in parser._actions]


class Canonical(ConfigDefaults):
    """Ignore only relocation/UI plumbing, never numerical learning expressions."""
    def __init__(self, runner=False):self.runner=runner
    def visit_Import(self,node):return None
    def visit_ImportFrom(self,node):return None
    # Sparse reference timing is an intentional post-refactor change, covered by test_reference_timing.
    def visit_BinOp(self, node):
        if isinstance(node.op, ast.Mult) and ast.unparse(node.left) == "self.goal_stride":
            return self.visit(node.right)
        return super().visit_BinOp(node)

    def visit_Constant(self,node):
        if isinstance(node.value,str):
            for old,new in sorted(MOVES.items(),key=lambda p:len(p[1]),reverse=True):
                node.value=node.value.replace(new,old)
        return node
    def visit_Expr(self,node):
        if isinstance(node.value,ast.Constant) and isinstance(node.value.value,str):return None
        s=ast.unparse(node.value)
        if s.startswith(('sys.path.insert(', 'parser.add_argument(', 'add_launcher_args(', 'TB_TAGS.update(')):return None
        return self.generic_visit(node)
    def visit_Assign(self,node):
        if ast.unparse(node) == "start = self.goal_stride if self.goal_start is None else self.goal_start":
            return None
        if ast.unparse(node) == "fut = f.unsqueeze(-1) + start + self.goal_stride * torch.arange(K, device=f.device)":
            return ast.parse("fut = f.unsqueeze(-1) + torch.arange(1, K + 1, device=f.device)").body[0]
        if ast.unparse(node.targets[0]) in {"goal_stride", "motion.goal_stride", "goal_start", "motion.goal_start"}:return None
        names={n.id for target in node.targets for n in ast.walk(target) if isinstance(n,ast.Name)}
        if names & {'REPO','DATA_DIR'}:return None
        if self.runner and (names & {'parser','CSV_FIELDS','SMOOTH_KEYS','TB_TAGS'} or ast.unparse(node.targets[0]).startswith('TB_TAGS[')):return None
        return super().visit_Assign(node)
    def visit_For(self,node):
        if self.runner and ast.unparse(node.iter) in ('EXTRA_FIELDS','CRITIC_FIELDS'):return None
        return self.generic_visit(node)
    def visit_FunctionDef(self,node):
        if self.runner and node.name in ('print_progress','_rss_gb','_hms'):return None
        return super().visit_FunctionDef(node)
    def visit_Call(self,node):
        name=ast.unparse(node.func)
        if self.runner and name=='print_progress':node.keywords=[]
        # New recording metadata changes neither state tensors nor simulation.
        if name=='np.savez_compressed':node.keywords=[k for k in node.keywords if k.arg!='control_dt']
        # Resolve the single relocated ground USD to the same semantic asset.
        for kw in node.keywords:
            if kw.arg=='usd_path' and 'default_ground_plane.usda' in ast.unparse(kw.value):
                kw.value=ast.Constant(value='asset:ground/default_ground_plane.usda')
        return super().visit_Call(node)


def fingerprint(source,runner=False):
    tree=Canonical(runner).visit(ast.parse(source))
    return hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest()


def test_all_relocated_files_present():
    for old,new in MOVES.items():
        assert (ROOT/new).is_file(),new
        assert not (ROOT/old).exists(),old


def test_training_cli_unchanged(monkeypatch):
    monkeypatch.setenv('SUPERMIMIC_ROBOT','g1')
    from supermimic.configs.train import build_parser
    expected=json.loads(BASELINE.read_text())['cli']
    assert [x for x in schema(build_parser()) if x["dest"] not in {"goal_stride", "goal_start", "allow_reference_timing_change", "wm_optimizer", "config", "print_config", "root_pos_z_weight", "body_pos_z_mode"}]==expected
    p=build_parser()
    a=p.parse_args(['--robot','tiangong3','--num_envs','4096','--wm_prediction','velocity_integrated','--no-act_soft_clip'])
    assert (a.robot,a.num_envs,a.wm_prediction,a.act_soft_clip)==('tiangong3',4096,'velocity_integrated',False)


@pytest.mark.parametrize('old', [p for p in [*MOVES][:15] if Path(p).name not in {'mdp.py','networks.py','robot.py','env.py','train.py','dr.py'}])
def test_numerical_modules_and_training_flow_unchanged(old):
    baseline=json.loads(BASELINE.read_text())['ast']
    if old not in baseline:pytest.skip('Non-source mapping')
    assert fingerprint((ROOT/MOVES[old]).read_text(),old.endswith(('/train.py','/evaluate.py')))==baseline[old],old


def test_python_syntax_and_script_references():
    for directory in ('src','scripts','tests','experiments'):
        for p in (ROOT/directory).rglob('*.py'):
            text=p.read_text();ast.parse(text,filename=str(p))
            if directory=='tests':continue
            for match in re.finditer(r'''['"]((?:scripts|tests|src|experiments)/[A-Za-z0-9_./-]+\.(?:py|sh))['"]''',text):
                assert (ROOT/match[1]).is_file(),(p,match[1])


def test_shell_syntax():
    scripts=list(ROOT.glob('*.sh'))
    for folder in ('scripts','experiments'):scripts.extend((ROOT/folder).rglob('*.sh'))
    for p in scripts:subprocess.run(['bash','-n',str(p)],check=True,capture_output=True,text=True)


def test_asset_roots_and_import_side_effects(tmp_path):
    env={**os.environ,'PYTHONPATH':str(ROOT/'src'),'PYTHONDONTWRITEBYTECODE':'1'}
    code='''import sys
import supermimic
from supermimic.utils.paths import REPO_ROOT, DATA_ROOT
from supermimic.configs.train import build_parser
assert "torch" not in sys.modules and "isaaclab" not in sys.modules
assert (DATA_ROOT / "tiangong3_urdf/urdf/tiangong3.urdf").is_file()
print(REPO_ROOT)
'''
    p=subprocess.run([sys.executable,'-c',code],cwd=tmp_path,env=env,check=True,capture_output=True,text=True)
    assert p.stdout.strip()==str(ROOT)


@pytest.mark.parametrize('script',['scripts/training/train.py','scripts/evaluation/evaluate.py','scripts/evaluation/play_native.py'])
def test_launcher_help_with_lab_cli_stub(tmp_path,script):
    # Only validates CLI wiring. This deliberately cannot launch a simulator.
    lab=tmp_path/'isaaclab';lab.mkdir();(lab/'__init__.py').write_text('')
    (lab/'app.py').write_text('def add_launcher_args(parser):\n    parser.add_argument("--device",default="cuda:0")\n    parser.add_argument("--visualizer")\ndef launch_simulation(*a,**kw):\n    raise AssertionError("Simulator must not launch for --help")\n')
    env={**os.environ,'SM_EXTRA_PYTHONPATH':str(tmp_path),'PYTHONDONTWRITEBYTECODE':'1','SM_PYTHON':sys.executable,
         'MPLCONFIGDIR':str(tmp_path/'mpl'),'WARP_CACHE_PATH':str(tmp_path/'warp'),'TORCHINDUCTOR_CACHE_DIR':str(tmp_path/'torch')}
    proc=subprocess.run(['bash',str(ROOT/'sm_native.sh'),'run',sys.executable,script,'--help'],cwd=tmp_path,env=env,capture_output=True,text=True,timeout=30)
    assert proc.returncode==0,proc.stderr
    assert 'usage:' in proc.stdout and '--robot' in proc.stdout if script!='scripts/evaluation/play_native.py' else '--control-hz' in proc.stdout


def test_training_process_patterns_cover_old_and_new_paths():
    text=(ROOT/'sm_native.sh').read_text()
    assert 'local cmd=("$PY" scripts/training/train.py' in text
    pattern=r'(scripts/training/train.py|supermimic/train.py) --log_dir logs/example( |$)'
    for path in ('scripts/training/train.py','supermimic/train.py'):
        assert re.search(pattern,f'python {path} --log_dir logs/example --robot tiangong3')
        assert not re.search(pattern,f'python {path} --log_dir logs/example_2 --robot tiangong3')


def test_recording_frequency_and_video_duration():
    from supermimic.utils.recording import recording_dt,render_indices
    for hz in (50,100):
        dt=recording_dt({'control_dt':np.array(1/hz)})
        for fps in (25,30,120):
            idx=render_indices(hz*2,dt,0,2,fps)
            assert len(idx)==fps*2
            np.testing.assert_allclose(idx*dt,np.arange(2*fps)/fps,atol=dt)
            assert idx.min()==0 and idx.max()<2*hz
        assert recording_dt({},hz)==dt
        assert len(render_indices(2*hz,dt,.5,1,25))==25
        assert len(render_indices(2*hz,dt,3,1,25))==0
    with pytest.raises(ValueError,match='no control_dt'):recording_dt({})
    with pytest.raises(ValueError,match='conflicts'):recording_dt({'control_dt':.01},50)
    for dt in (0,-1,float('nan')):
        with pytest.raises(ValueError):recording_dt({'control_dt':dt})


@pytest.mark.parametrize('name', ['test_sampling_and_critic_metrics.py','test_wm_prediction_and_history.py','test_wm_loss_diagnostics.py'])
def test_tensor_regression_suites(name):
    env={**os.environ,'PYTHONPATH':'','PYTHONDONTWRITEBYTECODE':'1'}
    p=subprocess.run([sys.executable,str(Path(__file__).with_name(name))],env=env,capture_output=True,text=True,timeout=120)
    assert p.returncode==0,p.stdout+'\n'+p.stderr


def robot_snapshot(repo, package, robot, dynamics, monkeypatch):
    """Run real robot parameter code with only Isaac Lab config containers stubbed."""
    import importlib.util
    import types
    class Config:
        def __init__(self,**kwargs):self.__dict__.update(kwargs)
    Config.InitialStateCfg=Config
    lab=types.ModuleType('isaaclab');lab.__path__=[]
    sim=types.ModuleType('isaaclab.sim')
    for name in ('UsdFileCfg','RigidBodyPropertiesCfg','ArticulationRootPropertiesCfg'):setattr(sim,name,Config)
    actuators=types.ModuleType('isaaclab.actuators');actuators.ImplicitActuatorCfg=Config
    assets=types.ModuleType('isaaclab.assets');assets.ArticulationCfg=Config
    lab.sim=sim
    for name,value in [('isaaclab',lab),('isaaclab.sim',sim),('isaaclab.actuators',actuators),('isaaclab.assets',assets)]:
        monkeypatch.setitem(sys.modules,name,value)
    monkeypatch.setenv('SUPERMIMIC_ROBOT',robot)
    prefix='_layout_asset_'+robot+'_'+dynamics+'_'+str(len(package))
    pkg=types.ModuleType(prefix);pkg.__path__=[str(repo/package)];monkeypatch.setitem(sys.modules,prefix,pkg)
    spec=importlib.util.spec_from_file_location(prefix+'.robots',repo/package/'robots/__init__.py',submodule_search_locations=[str(repo/package/'robots')])
    module=importlib.util.module_from_spec(spec);monkeypatch.setitem(sys.modules,spec.name,module);spec.loader.exec_module(module)
    module.set_dynamics(dynamics)
    result={
        'joint_names':module.JOINT_NAMES,'tracked_bodies':module.TRACKED_BODIES,
        'q0':module.default_joint_pos_list(),'action_scale':[module.action_scale(j) for j in module.JOINT_NAMES],
        'actuators':{k:v.__dict__ for k,v in module.G1_CFG.actuators.items()},
        'self_collision':module.G1_CFG.spawn.articulation_props.enabled_self_collisions,
        'friction':list(module.FRICTION_RANGE),
        'paths':{k:str(Path(getattr(module,k)).resolve().relative_to((repo/'data').resolve())) for k in ('URDF_PATH','MJCF_PATH','USD_PATH')},
    }
    for key in ('URDF_PATH','MJCF_PATH','USD_PATH'):assert Path(getattr(module,key)).is_file()
    return result,module


@pytest.mark.parametrize('robot,dynamics',[('g1','default'),('tiangong3','default'),('tiangong3','dexevt')])
def test_robot_assets_pd_actions_and_collisions_unchanged(monkeypatch,robot,dynamics):
    result,module=robot_snapshot(ROOT,'src/supermimic',robot,dynamics,monkeypatch)
    expected=json.loads(BASELINE.read_text())['robots'][robot+'/'+dynamics]
    assert result==expected
    # Parse actual robot model/meshes on CPU. Config-container stubs do not stand
    # in for a real Isaac Lab articulation or GPU dynamics test.
    import mujoco
    model=mujoco.MjModel.from_xml_path(module.MJCF_PATH)
    assert (model.nq,model.nv)==(36,35)


@pytest.mark.parametrize('all_alive',[True,False])
def test_failure_analysis_accepts_completed_and_failed_recordings(tmp_path,all_alive):
    n=5
    data=dict(control_dt=.01,names=np.array(['test']),alive=np.array([[True],[True],[True],[all_alive],[all_alive]]),
              f=np.arange(n)[:,None],clip_start=np.array([0]),clip_len=np.array([n]),
              root_p=np.zeros((n,1,3)),root_quat=np.tile([1.,0,0,0],(n,1,1)),q=np.zeros((n,1,29)),
              ref_root_p=np.zeros((n,3)),ref_root_R=np.tile(np.eye(3),(n,1,1)),ref_q=np.zeros((n,29)))
    path=tmp_path/'record.npz';np.savez(path,**data)
    proc=subprocess.run([sys.executable,str(ROOT/'scripts/evaluation/analyze_failure.py'),str(path),'test'],capture_output=True,text=True)
    assert proc.returncode==0,proc.stderr
    assert ('record ended while alive' if all_alive else 'terminated') in proc.stdout
