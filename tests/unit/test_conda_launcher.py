"""Environment selection must not silently fall back to the old simulator stack."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]


def invoke(tmp_path, **overrides):
    env = {k: v for k, v in os.environ.items() if not k.startswith(('SM_', 'CONDA_'))}
    env.update(CONDA_EXE=str(tmp_path / 'fake_conda'), PYTHONPATH='/unrelated/python310',
               MPLCONFIGDIR=str(tmp_path / 'mpl'), WARP_CACHE_PATH=str(tmp_path / 'warp'),
               TORCHINDUCTOR_CACHE_DIR=str(tmp_path / 'torch'))
    env.update(overrides)
    return subprocess.run(
        ['bash', str(ROOT / 'sm_native.sh'), 'run', 'bash', '-c',
         'printf "%s\\n%s\\n%s\\n" "$SM_PYTHON" "$PYTHONPATH" "$PYTHONNOUSERSITE"'],
        env=env, text=True, capture_output=True, timeout=30,
    )


def fake_environment(tmp_path, name):
    prefix = tmp_path / name
    (prefix / 'bin').mkdir(parents=True)
    (prefix / 'bin/python').symlink_to(sys.executable)
    return prefix


def fake_conda(tmp_path, prefixes):
    path = tmp_path / 'fake_conda'
    path.write_text('#!/bin/sh\ncat <<\'JSON\'\n' + json.dumps({'envs': list(map(str, prefixes))}) + '\nJSON\n')
    path.chmod(0o755)


def test_named_env_overrides_unrelated_active_env(tmp_path):
    target = fake_environment(tmp_path, 'tg_supermimic')
    old = fake_environment(tmp_path, 'supermimic')
    fake_conda(tmp_path, [old, target])
    result = invoke(tmp_path, CONDA_PREFIX=str(old), CONDA_DEFAULT_ENV='supermimic')
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [str(target / 'bin/python'), str(ROOT / 'src'), '1']


@pytest.mark.parametrize('selector', ['SM_CONDA_PREFIX', 'SM_PYTHON'])
def test_explicit_environment_override(tmp_path, selector):
    target = fake_environment(tmp_path, 'custom')
    value = target / 'bin/python' if selector == 'SM_PYTHON' else target
    result = invoke(tmp_path, **{selector: str(value)}, SM_EXTRA_PYTHONPATH='/explicit/addition')
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[:2] == [str(target / 'bin/python'), str(ROOT / 'src') + ':/explicit/addition']


def test_custom_named_env(tmp_path):
    target = fake_environment(tmp_path, 'custom')
    fake_conda(tmp_path, [target])
    result = invoke(tmp_path, SM_CONDA_ENV='custom')
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[0] == str(target / 'bin/python')


def test_missing_named_env_does_not_use_arbitrary_active_env(tmp_path):
    old = fake_environment(tmp_path, 'supermimic')
    fake_conda(tmp_path, [old])
    result = invoke(tmp_path, CONDA_PREFIX=str(old), CONDA_DEFAULT_ENV='supermimic')
    assert result.returncode != 0
    assert 'not found' in result.stderr
