"""Validate two-rank learning, pool rotation, resume and production-sized updates."""
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
ART = Path('/media/supermimic/artifacts/relative_height_20261116')
CFG = ROOT / 'experiments/relative_height_20261116/config.json'
DATA = Path('/media/supermimic/artifacts/full_wbc_pool_20261111/smoke_dataset')


def inspect(run, iteration):
    import torch
    path = ROOT / 'logs' / run
    ck = torch.load(path/'last.pt', map_location='cpu', weights_only=False)
    assert ck['iteration'] == iteration
    a = ck['args']
    assert a['world_size'] == 2 and a['goal_offsets'] == [1,6,11,16,21,26,31,36,41,46]
    assert a['reward_cfg']['root_pos_z_weight'] == .1
    assert a['reward_cfg']['body_pos_z_mode'] == 'relative_root'
    assert a['reward_cfg']['term_height'] == .25
    assert len(ck['motion_pool']) == 2
    assert all(p['refresh_count'] >= iteration//2 for p in ck['motion_pool'])
    for key in ('wm', 'pi', 'critic', 'critic_targ'):
        assert all(torch.isfinite(t).all() for t in ck[key].values() if t.is_floating_point()), key
    for key in ('opt_wm', 'opt_pi', 'opt_v'):
        assert ck[key]['state'], key
        for state in ck[key]['state'].values():
            assert all(torch.isfinite(t).all() for t in state.values() if isinstance(t, torch.Tensor)), key
    rows = list(csv.DictReader((path/'train_log.csv').open()))
    assert rows and all(math.isfinite(float(v)) for row in rows for v in row.values())
    assert all(float(row['sim_nonfinite']) == 0 for row in rows)
    assert 'error_body_pos_ref_z_m' in rows[-1] and 'error_body_z_relative_m' in rows[-1]
    return dict(passed=True, run=run, iteration=iteration, reward=a['reward_cfg'],
                finite_networks_optimizers_metrics=True, refreshes=[p['refresh_count'] for p in ck['motion_pool']],
                cuda_memory=ck.get('cuda_memory'))


if __name__ == '__main__':
    ART.mkdir(parents=True, exist_ok=True)
    result = dict(passed=False, started=time.strftime('%F %T'))
    base = [sys.executable, 'experiments/full_wbc_pool_20261111/scripts/run_pool_training.py',
            '--gpus', '4,5', '--config', str(CFG), '--smoke', '--motion-dir', str(DATA)]
    def stage(name, args, iteration):
        print(f'[preflight] {name}', flush=True)
        subprocess.run(base + args, cwd=ROOT, check=True)
        run = args[args.index('--run')+1]
        result[name] = inspect(run, iteration)
        (ART/'preflight.json').write_text(json.dumps(result, indent=2)+'\n')
    try:
        run = 'relz_ddp_smoke_20261116'
        stage('update_and_rotation', ['--run', run, '--iterations', '2', '--eager'], 2)
        stage('resume', ['--run', run, '--iterations', '4', '--eager', '--resume', str(ROOT/'logs'/run/'last.pt')], 4)
        stage('compiled_4096', ['--run', 'relz_capacity_20261116', '--iterations', '2', '--envs', '4096', '--batch', '4096'], 2)
        result.update(passed=True, completed=time.strftime('%F %T'))
    except BaseException as exc:
        result['error'] = repr(exc)
        raise
    finally:
        (ART/'preflight.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result, indent=2), flush=True)
