"""Check the installed runtime and bundled assets before starting a long training."""
from pathlib import Path
import importlib.metadata as metadata
import os
import sys
import hashlib
import json
import subprocess

subprocess.run([sys.executable, "-m", "pip", "check"], check=True)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
for name, expected in json.loads((ROOT / 'assets-manifest.json').read_text()).items():
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
print('Bundled asset checksums OK')
for name, expected in {
    'torch': '2.11.0', 'mujoco': '3.15.0', 'mujoco-warp': '3.15.0',
    'warp-lang': '1.17.0', 'newton': '1.8.0.dev0',
    'newton-usd-schemas': '0.5.0', 'usd-exchange': '3.0.1',
}.items():
    version = metadata.version(name)
    assert version.split('+')[0] == expected, (name, version, expected)
    print(name, version)
import torch
import mujoco
import newton
import isaaclab.sim
from pxr import Usd

for name in ['g1_description/g1_29dof.xml', 'tiangong3_generated/tiangong3.xml']:
    model = mujoco.MjModel.from_xml_path(str(ROOT / 'data' / name))
    assert model.nq == 36 and model.nv == 35, (name, model.nq, model.nv)
    print('MJCF OK:', name)
for name in ['g1_usd/g1_29dof/g1_29dof.usda', 'tiangong3_usd/tiangong3/tiangong3.usda', 'ground/default_ground_plane.usda']:
    stage = Usd.Stage.Open(str(ROOT / 'data' / name))
    assert stage and stage.GetDefaultPrim(), name
    if stage.GetDefaultPrim().GetVariantSet('Physics').GetVariantNames():
        assert stage.GetDefaultPrim().GetVariantSet('Physics').SetVariantSelection('mujoco')
    assert not stage.GetCompositionErrors(), (name, stage.GetCompositionErrors())
    print('USD OK:', name)
if os.environ.get('SM_SKIP_GPU_CHECK') != '1':
    assert torch.cuda.is_available(), 'No CUDA GPU. Check the host NVIDIA driver.'
    print('GPU:', torch.cuda.get_device_name(0))
    print('CUDA tensor:', torch.ones(1, device='cuda').cpu().item())
print('Install checks passed. See docs/DEPLOY_CONDA.md for simulator and training smoke checks.')
