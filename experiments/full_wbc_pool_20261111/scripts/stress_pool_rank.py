"""Memory test: real full-WBC rank shard, all old active clips pinned, then full learning updates.

Only used as a test entry point. It overrides library construction in this process so the
normal train.py executes with the worst first-refresh resident set already allocated.
"""
from pathlib import Path
import os
os.environ['SUPERMIMIC_ROBOT']='tiangong3'
import sys
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/"src"))
import argparse,json,runpy
from types import SimpleNamespace
import torch
from supermimic.robots.kinematics import G1Kinematics
from supermimic.motions.pool import MotionPool

p=argparse.ArgumentParser();p.add_argument('--rank',type=int,default=0);p.add_argument('--envs',type=int,default=4096);p.add_argument('--batch',type=int,default=4096);p.add_argument('--run',required=True);a=p.parse_args()
ART=Path('/media/supermimic/artifacts/full_wbc_pool_20261111')
kin=G1Kinematics('cuda:0')
pool=MotionPool.from_dir('/media/supermimic/data/full_wbc_validated',kin,'cuda:0',a.rank,4,
    active_ratio=.3,replace_ratio=.2,seed=a.rank,stats_cache='/media/supermimic/cache/tg_supermimic/full_wbc_pool_stats')
initial=pool.metrics()
f=pool.clip_start.clone()
env=SimpleNamespace(f=f.clone(),start_bin=pool.frame_bin[f].clone(),bin_G=torch.zeros(pool.num_bins,device='cuda:0'),bin_w_avg=torch.ones(pool.num_bins,device='cuda:0')/pool.num_bins)
replay=SimpleNamespace(reference_frames=lambda frames=f:frames)
pool.refresh(env,replay)
resident=pool.metrics()
(ART/(a.run+'_residency.json')).write_text(json.dumps(dict(initial=initial,pinned_resident=resident,rank=a.rank,scenario='all old active clips pinned across the first 30/20 refresh'),indent=2)+'\n')
del env,replay,f
# Return this exact library to the real trainer; all sampling, WM and actor math stay unchanged.
MotionPool.from_dir=classmethod(lambda cls,*args,**kwargs:pool)
args=['scripts/training/train.py','--robot','tiangong3','--log_dir',str(ROOT/'logs'/a.run),'--num_envs',str(a.envs),
 '--wm_batch',str(a.batch),'--pi_batch',str(a.batch),'--iterations','2','--buffer_steps','1000','--utd','.4',
 '--wm_window','12','--horizon','48','--wm_batches','10','--hist','10','--goal_frames','10','--lr','.0002',
 '--gamma','.995','--lam','.95','--action_noise','.1','--critic_epochs','5','--critic_minibatches','4',
 '--critic_tau','.05','--critic_layer_norm','--act_soft_clip','--act_norm','ref','--reset_a_prev','pose',
 '--reset_temp','25','--warmup_steps','64','--wm_only_iters','0','--ckpt_every','2','--log_every','1',
 '--seed','0','--amp','bf16','--dr','all,contact','--dr_ramp','0','--dynamics','dexevt',
 '--motion_active_ratio','.3','--motion_replace_ratio','.2','--motion_refresh_iters','0',
 '--motion_dir','/media/supermimic/data/full_wbc_validated']
assert not (ROOT/'logs'/a.run/'last.pt').exists()
sys.argv=args
runpy.run_path(str(ROOT/'scripts/training/train.py'),run_name='__main__')
