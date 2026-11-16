"""Validate two-rank learning, pool rotation, resume and production-sized updates."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
ART = Path('/media/supermimic/artifacts/sole_point_ablation_20261116')
CFG = ROOT / 'experiments/sole_point_ablation_20261116/config.json'
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
    assert a['reward_cfg']['body_point_offsets'] == {name: (.09, 0., -.058) for name in ('ankle_roll_l_link', 'ankle_roll_r_link')}
    assert a['config'] == 'tiangong3_full_wbc'
    assert len(ck['motion_pool']) == 2
    assert all(p['refresh_count'] >= iteration//2 for p in ck['motion_pool'])
    for key in ('wm', 'pi', 'critic', 'critic_targ'):
        assert all(torch.isfinite(t).all() for t in ck[key].values() if t.is_floating_point()), key
    for key in ('opt_wm', 'opt_pi', 'opt_v'):
        assert ck[key]['state'], key
        for state in ck[key]['state'].values():
            assert all(torch.isfinite(t).all() for t in state.values() if isinstance(t, torch.Tensor)), key
    rows = list(csv.DictReader((path/'train_log.csv').open()))
    assert rows
    for row in rows:
        for key, value in row.items():
            value = float(value)
            # Episode means are intentionally absent when this window has no completed episodes.
            empty_episode_mean = key.startswith('episode_') and float(row['episodes_ended']) == 0
            assert math.isfinite(value) or (empty_episode_mean and math.isnan(value)), (key, value)
    assert all(float(row['sim_nonfinite']) == 0 for row in rows)
    assert 'error_body_pos_ref_z_m' in rows[-1] and 'error_body_z_relative_m' in rows[-1]
    return dict(passed=True, run=run, iteration=iteration, reward=a['reward_cfg'],
                finite_networks_optimizers_training_metrics=True, refreshes=[p['refresh_count'] for p in ck['motion_pool']],
                cuda_memory=ck.get('cuda_memory'))


if __name__ == '__main__':
    ART.mkdir(parents=True, exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--continue', dest='continue_run', action='store_true')
    options = parser.parse_args()
    result = json.loads((ART/'preflight.json').read_text()) if options.continue_run else dict(passed=False, started=time.strftime('%F %T'))
    if 'error' in result:
        result.setdefault('previous_errors', []).append(result.pop('error'))
    base = [sys.executable, 'experiments/full_wbc_pool_20261111/scripts/run_pool_training.py',
            '--gpus', '0,1', '--config', str(CFG), '--smoke', '--motion-dir', str(DATA)]
    def stage(name, args, iteration):
        print(f'[preflight] {name}', flush=True)
        if options.continue_run and result.get(name, {}).get('passed'):
            return
        run = args[args.index('--run')+1]
        checkpoint = ROOT/'logs'/run/'last.pt'
        completed = False
        if options.continue_run and checkpoint.exists():
            import torch
            completed = torch.load(checkpoint, map_location='cpu', weights_only=False)['iteration'] == iteration
        if not completed:
            subprocess.run(base + args, cwd=ROOT, check=True)
        result[name] = inspect(run, iteration)
        (ART/'preflight.json').write_text(json.dumps(result, indent=2)+'\n')
    try:
        run = 'sole090_ddp_smoke_20261116'
        stage('update_and_rotation', ['--run', run, '--iterations', '2', '--eager'], 2)
        stage('resume', ['--run', run, '--iterations', '4', '--eager', '--resume', str(ROOT/'logs'/run/'last.pt')], 4)
        stage('compiled_4096', ['--run', 'sole090_capacity_20261116', '--iterations', '2', '--envs', '4096', '--batch', '4096'], 2)
        result.update(passed=True, completed=time.strftime('%F %T'))
    except BaseException as exc:
        result['error'] = repr(exc)
        raise
    finally:
        (ART/'preflight.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result, indent=2), flush=True)
