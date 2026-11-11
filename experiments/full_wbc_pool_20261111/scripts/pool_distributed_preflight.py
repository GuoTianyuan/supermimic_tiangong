"""Four-rank update/rotation/checkpoint/resume test, then the requested fresh production run."""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
ART=Path('/media/supermimic/artifacts/full_wbc_pool_20261111')

def inspect_checkpoint(path,world,iteration):
    import torch
    ck=torch.load(path,map_location='cpu',weights_only=False)
    assert ck['iteration']==iteration
    assert len(ck['motion_pool'])==world
    assert all(s['refresh_count']>=iteration//2 for s in ck['motion_pool'])
    for name in ('wm','pi','critic','critic_targ'):
        assert all(torch.isfinite(t).all() for t in ck[name].values() if t.is_floating_point()),name
    for name in ('opt_wm','opt_pi','opt_v'):
        assert ck[name]['state'],name
    rows=list(csv.DictReader((path.parent/'train_log.csv').open()))
    assert rows and all(math.isfinite(float(v)) for row in rows for v in row.values())
    return dict(passed=True,iteration=iteration,world_size=world,refreshes=[s['refresh_count'] for s in ck['motion_pool']],
                cuda_memory=ck.get('cuda_memory'),finite_weights_and_metrics=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--gpus',required=True);p.add_argument('--check-only',type=Path);p.add_argument('--world',type=int,default=1);p.add_argument('--iteration',type=int,default=4)
    a=p.parse_args()
    if a.check_only:
        print(json.dumps(inspect_checkpoint(a.check_only,a.world,a.iteration),indent=2));return
    groups=a.gpus.split(',');assert len(groups)==4
    run='pool_ddp_preflight_'+time.strftime('%Y%m%d_%H%M%S')
    data=ART/'smoke_dataset';assert len(list(data.glob('*.npz')))>=32
    cmd=[sys.executable,'experiments/full_wbc_pool_20261111/scripts/run_pool_training.py','--gpus',a.gpus,'--smoke','--run',run,'--motion-dir',str(data),'--iterations','2','--eager']
    subprocess.run(cmd,cwd=ROOT,check=True)
    last=ROOT/'logs'/run/'last.pt';inspect_checkpoint(last,4,2)
    cmd[cmd.index('--iterations')+1]='4';cmd+=['--resume',str(last)]
    subprocess.run(cmd,cwd=ROOT,check=True)
    report=inspect_checkpoint(last,4,4);report.update(run=run,gpus=groups,scope='small eager four-rank updates/rotation/checkpoint/resume; production compilation remains enabled')
    (ART/'distributed_preflight.json').write_text(json.dumps(report,indent=2)+'\n')
    del report
    os.chdir(ROOT)
    os.execv(sys.executable,[sys.executable,'experiments/full_wbc_pool_20261111/scripts/run_pool_training.py','--gpus',a.gpus])

if __name__=='__main__':main()
