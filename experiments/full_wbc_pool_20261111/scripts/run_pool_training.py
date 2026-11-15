"""Explicit full-WBC pool preset; --smoke exercises updates, refresh and resume separately."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
ART = Path('/media/supermimic/artifacts/full_wbc_pool_20261111')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpus',required=True)
    p.add_argument('--config',type=Path,default=ROOT/'experiments/full_wbc_pool_20261111/config.json')
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--envs',type=int)
    p.add_argument('--batch',type=int)
    p.add_argument('--motion-dir')
    p.add_argument('--run')
    p.add_argument('--resume')
    p.add_argument('--allow-reference-timing-change', action='store_true')
    p.add_argument('--iterations',type=int)
    p.add_argument('--eager',action='store_true')
    a=p.parse_args()
    cfg=json.loads(a.config.read_text())
    gpus=[int(x) for x in a.gpus.split(',')]
    if len(gpus)!=len(set(gpus)):raise ValueError('Each rank needs a distinct GPU')
    if not a.smoke and len(gpus)!=cfg['world_size']:raise ValueError('Production preset requires four GPUs')
    run=a.run or cfg['run']
    if not all(ch.isalnum() or ch in '._-' for ch in run):raise ValueError('Invalid run name')
    if a.smoke and run==cfg['run']:raise ValueError('Use a distinct smoke run name')
    dest=ROOT/'logs'/run
    if not a.resume and any(dest.glob('*.pt')):raise RuntimeError('Existing checkpoint: explicit --resume required')
    if a.resume and not Path(a.resume).is_file():raise FileNotFoundError(a.resume)
    pars=dict(cfg['training'],**cfg['per_rank'])
    # Historical manifests predate sparse future references. New configs set goal_stride explicitly.
    pars.setdefault('goal_stride', 1)
    pars.setdefault('goal_start', pars['goal_stride'])  # Keep historical presets unchanged.
    pars.update(motion_active_ratio=cfg['pool']['active_ratio'],motion_replace_ratio=cfg['pool']['replace_ratio'],
                motion_refresh_iters=cfg['pool']['refresh_iters'],motion_dir=a.motion_dir or cfg['dataset'],
                motion_stats_cache='/media/supermimic/cache/tg_supermimic/full_wbc_pool_stats',
                dr=cfg['dr'],dr_curriculum=cfg['dr_curriculum'],dr_ramp=cfg['dr_ramp'],dynamics=cfg['dynamics'],
                amp=cfg['amp'],action_noise=.1,act_norm='ref',reset_a_prev='pose',reset_temp=25)
    if a.smoke:
        pars.update(iterations=4,num_envs=16,wm_batch=32,pi_batch=32,buffer_steps=64,wm_batches=1,
                    motion_refresh_iters=2,warmup_steps=32,wm_only_iters=0,ckpt_every=2,log_every=1,
                    dr_curriculum='',dr_ramp=0)
    if a.envs:pars['num_envs']=a.envs
    if a.batch:pars['wm_batch']=pars['pi_batch']=a.batch
    if a.iterations:pars['iterations']=a.iterations
    # Stress tests allocate the same replay capacity as production.
    if a.smoke and pars['num_envs']>=1024:pars['buffer_steps']=1000;pars['wm_batches']=10;pars['warmup_steps']=64
    if a.resume:pars['resume']=a.resume
    env=os.environ.copy()
    cache=Path('/media/supermimic/cache')/cfg.get('server','a800_1')/'full_wbc_pool'
    env.update(SM_PYTHON=env.get('SM_PYTHON','/media/supermimic/envs/tg_supermimic/bin/python'),OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='1',
               LD_LIBRARY_PATH='/media/supermimic/runtime/cuda-compat/usr/local/cuda-13.0/compat:/usr/local/nvidia/lib:/usr/local/nvidia/lib64',
               WARP_CACHE_PATH=env.get('WARP_CACHE_PATH',str(cache/'warp')),
               TORCHINDUCTOR_CACHE_DIR=env.get('TORCHINDUCTOR_CACHE_DIR',str(cache/('inductor_'+run))),
               MPLCONFIGDIR=env.get('MPLCONFIGDIR',str(cache/'matplotlib')))
    if a.eager:env['TORCHDYNAMO_DISABLE']='1'
    else:env.pop('TORCHDYNAMO_DISABLE',None)
    cmd=['bash','sm_native.sh','_train',run,'--gpu',a.gpus,'--robot','tiangong3','--no_tb']
    for k,v in pars.items():cmd += ['--'+k,str(v)]
    cmd+=['--critic_layer_norm','--act_soft_clip']
    if a.allow_reference_timing_change: cmd += ['--allow_reference_timing_change']
    dest.mkdir(parents=True,exist_ok=True)
    manifest=dict(config=cfg,run=run,gpus=gpus,world_size=len(gpus),actual_args=pars,smoke=a.smoke,eager=a.eager,
                  source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                  global_envs=pars['num_envs']*len(gpus),global_wm_batch=pars['wm_batch']*len(gpus),global_pi_batch=pars['pi_batch']*len(gpus),command=cmd)
    (dest/('resume_manifest.json' if a.resume else 'launch_manifest.json')).write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:manifest[k] for k in ['run','gpus','global_envs','global_wm_batch','global_pi_batch','source_commit']},indent=2),flush=True)
    os.chdir(ROOT)
    os.execvpe('bash',cmd,env)

if __name__=='__main__':main()
