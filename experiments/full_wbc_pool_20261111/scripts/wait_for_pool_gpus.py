"""Wait for four unoccupied GPUs; share advisory locks with the physics-frequency queues."""
from __future__ import annotations
import argparse
import fcntl
import itertools
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT=Path(__file__).resolve().parents[3]

def idle_gpus():
    lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()
    busy={s.strip() for s in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True).splitlines()}
    idle=[]
    for line in lines:
        i,u,m,p=[s.strip() for s in line.split(',')]
        if u not in busy and int(m)<1024 and int(p)<5:idle.append(int(i))
    return idle

def acquire(gpus):
    locks=[]
    try:
        for gpu in sorted(gpus):
            f=open(f'/tmp/tg-physics-gpu{gpu}.lock','a')
            locks.append(f)
            fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return locks
    except BlockingIOError:
        for f in locks:f.close()
        return []

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--ready',required=True);p.add_argument('--state',required=True);p.add_argument('--poll',type=float,default=20)
    a=p.parse_args();state=Path(a.state);state.parent.mkdir(parents=True,exist_ok=True)
    run=json.loads((ROOT/'experiments/full_wbc_pool_20261111/config.json').read_text())['run']
    stopped=False;child=None
    def save(status,**extra):
        tmp=state.with_suffix('.tmp');tmp.write_text(json.dumps(dict(status=status,run=run,repo=str(ROOT),required_gpus=4,updated=time.strftime('%Y-%m-%d %H:%M:%S'),**extra),indent=2)+'\n');tmp.replace(state)
    def stop(sig,frame):
        nonlocal stopped
        stopped=True
        if child is not None and child.poll() is None:os.killpg(child.pid,signal.SIGINT)
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    streak={}
    while not stopped:
        ready=Path(a.ready)
        if not ready.exists() or not json.loads(ready.read_text()).get('passed'):
            save('waiting_for_validation');time.sleep(a.poll);continue
        try:idle=idle_gpus()
        except (ValueError,subprocess.SubprocessError) as e:
            save('gpu_query_failed',error=str(e));time.sleep(a.poll);continue
        streak={g:streak.get(g,0)+1 for g in idle}
        candidates=[g for g in idle if streak[g]>=3]
        for group in itertools.combinations(candidates,4):
            locks=acquire(group)
            if not locks:continue
            if not set(group)<=set(idle_gpus()):
                for f in locks:f.close()
                continue
            if (ROOT/'logs'/run/'last.pt').exists():
                save('existing_checkpoint_requires_explicit_resume',gpus=group)
                for f in locks:f.close()
                return
            cmd=['/media/supermimic/envs/tg_supermimic/bin/python','experiments/full_wbc_pool_20261111/scripts/pool_distributed_preflight.py','--gpus',','.join(map(str,group))]
            save('distributed_preflight',gpus=group,command=cmd)
            child=subprocess.Popen(cmd,cwd=ROOT,start_new_session=True)
            while child.poll() is None:time.sleep(2)
            save('stopped' if stopped else ('completed' if child.returncode==0 else 'failed'),gpus=group,exit_code=child.returncode)
            for f in locks:f.close()
            return
        save('waiting_for_four_idle_gpus',idle_gpu_indices=idle)
        print(f'{time.strftime("%H:%M:%S")} {run}: waiting for four idle GPUs; now idle {idle}',flush=True)
        time.sleep(a.poll)
    save('cancelled_before_start')

if __name__=='__main__':main()
