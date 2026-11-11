"""Finish isolated GPU7 capacity tests; only emit ready.json after all checks pass."""
from __future__ import annotations
import json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
ART=Path('/media/supermimic/artifacts/full_wbc_pool_20261111')
sys.path.insert(0,str(Path(__file__).resolve().parent))
from pool_distributed_preflight import inspect_checkpoint

def main():
    started=time.time();run='pool_probe4096_20261111';last=ROOT/'logs'/run/'last.pt'
    while True:
        pane=subprocess.check_output(['tmux','list-panes','-t','supermimic:pool-batch-probe','-F','#{pane_dead} #{pane_dead_status}'],text=True).strip()
        if pane.startswith('1 '):
            if pane!='1 0':raise RuntimeError('Initial 4096 batch probe failed: '+pane)
            break
        time.sleep(10)
    first=inspect_checkpoint(last,1,4)
    cmd=[sys.executable,'experiments/full_wbc_pool_20261111/scripts/run_pool_training.py','--gpus','7','--smoke','--envs','4096','--batch','4096',
         '--iterations','6','--run',run,'--motion-dir',str(ART/'smoke_dataset'),'--resume',str(last)]
    with (ART/'capacity_resume.log').open('w') as log:subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    resumed=inspect_checkpoint(last,1,6)
    while not (ART/'catalog.json').exists():
        print('Waiting for the four full-catalog action-statistics caches.',flush=True);time.sleep(20)
    stress_run='pool_fullrank4096_20261111'
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='7',OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='1',
        LD_LIBRARY_PATH='/media/supermimic/runtime/cuda-compat/usr/local/cuda-13.0/compat:/usr/local/nvidia/lib:/usr/local/nvidia/lib64',
        WARP_CACHE_PATH='/media/supermimic/cache/a800_1/full_wbc_pool/warp',
        TORCHINDUCTOR_CACHE_DIR='/media/supermimic/cache/a800_1/full_wbc_pool/inductor_'+run)
    env.pop('TORCHDYNAMO_DISABLE',None)
    # Training and the stress test use the same compiled shapes; cache can be reused.
    cmd=[sys.executable,'experiments/full_wbc_pool_20261111/scripts/stress_pool_rank.py','--envs','4096','--batch','4096','--run',stress_run]
    samples=[]
    with (ART/'capacity_fullrank.log').open('w') as log:
        proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        while proc.poll() is None:
            row=subprocess.check_output(['nvidia-smi','-i','7','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
            samples.append(dict(t=time.time(),gpu=row));time.sleep(5)
        (ART/'capacity_gpu_samples.json').write_text(json.dumps(samples,indent=2)+'\n')
        if proc.returncode:raise RuntimeError(f'Full rank 4096 stress test exited {proc.returncode}; see capacity_fullrank.log')
    full=inspect_checkpoint(ROOT/'logs'/stress_run/'last.pt',1,2)
    record=dict(passed=True,environment_per_gpu=4096,wm_batch_per_gpu=4096,actor_batch_per_gpu=4096,
                initial=first,resume=resumed,full_rank=full,residency=json.loads((ART/(stress_run+'_residency.json')).read_text()),
                elapsed_seconds=time.time()-started,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                scope='Single GPU actual compiled WM12/Actor48 updates after worst first pool refresh; four-rank checks run when four free cards are reserved.')
    (ART/'capacity.json').write_text(json.dumps(record,indent=2)+'\n')
    for name in ['reward.json','windows.json','pool_cpu.json','pool_gpu.json','dataset_validation.json','catalog.json']:
        assert (ART/name).exists(),name
    (ART/'ready.json').write_text(json.dumps(dict(passed=True,source_commit=record['source_commit'],capacity='capacity.json',four_rank_preflight_required=True),indent=2)+'\n')
    print(json.dumps(record,indent=2),flush=True)

if __name__=='__main__':
    try:main()
    except Exception as e:
        (ART/'capacity_failure.json').write_text(json.dumps(dict(passed=False,error=str(e)),indent=2)+'\n')
        raise
