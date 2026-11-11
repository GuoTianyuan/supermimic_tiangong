"""Stream full-catalog fixed action statistics on CPU and estimate the four rank pools."""
from pathlib import Path
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/"src"))
os.environ['SUPERMIMIC_ROBOT']='tiangong3'
import concurrent.futures
import hashlib
import json
import numpy as np
import torch
from supermimic.motion_pool import MotionPool
from supermimic.motion import CONTROL_DT
from supermimic.robot import JOINT_NAMES,ROBOT_NAME
ART=Path('/media/supermimic/artifacts/full_wbc_pool_20261111')
DATA=Path('/media/supermimic/data/full_wbc_validated')

def rank_stats(rank):
    torch.set_num_threads(2)
    files=sorted(DATA.glob('*.npz'))[rank::4]
    lengths=[];fingerprint=hashlib.sha256()
    for f in files:
        meta=json.loads(Path(str(f)+'.metadata.json').read_text());st=f.stat()
        lengths.append(int(np.floor((meta['frames']-1)/meta['fps']/CONTROL_DT))+1)
        fingerprint.update(json.dumps([f.name,meta,st.st_size,st.st_mtime_ns],sort_keys=True).encode())
    fingerprint.update(json.dumps(['motion_pool_v1',CONTROL_DT,ROBOT_NAME,JOINT_NAMES]).encode())
    pool=MotionPool.__new__(MotionPool)
    pool.files=list(map(str,files));pool.catalog_hash=fingerprint.hexdigest();pool.catalog_len=torch.tensor(lengths)
    pool.catalog_num_frames=sum(lengths);pool.device='cpu';pool.stats_cache='/media/supermimic/cache/tg_supermimic/full_wbc_pool_stats'
    var,mean,count=pool.reference_joint_stats()
    rng=np.random.default_rng(rank);active=sorted(rng.choice(len(files),round(len(files)*.3),replace=False).tolist())
    available=sorted(set(range(len(files)))-set(active))
    drop=set(rng.choice(active,round(len(files)*.2),replace=False).tolist())
    add=rng.choice(available,round(len(files)*.2),replace=False).tolist()
    resident=sorted(set(active)|set(add))
    # 83 float32 per GPU root/joint/goal row; tracked body states are on CPU.
    frames_initial=sum(lengths[i] for i in active);frames_pinned=sum(lengths[i] for i in resident)
    report=dict(rank=rank,clips=len(files),frames=count,hash=pool.catalog_hash,
                active_clips=len(active),replace_clips=len(add),initial_frames=frames_initial,
                all_old_active_pinned_frames_after_first_refresh=frames_pinned,
                root_joint_goal_gpu_gib_initial=frames_initial*83*4/2**30,
                root_joint_goal_gpu_gib_pinned=frames_pinned*83*4/2**30,
                stats_mean=mean.tolist(),stats_var=var.tolist())
    (ART/f'catalog_rank{rank}.json').write_text(json.dumps(report,indent=2)+'\n')
    if rank==0:
        shard=ART/'rank0_dataset';shard.mkdir(exist_ok=True)
        for f in files:
            for source in (f,Path(str(f)+'.metadata.json')):
                dst=shard/source.name
                if not dst.is_symlink():dst.symlink_to(source)
        smoke=ART/'smoke_dataset';smoke.mkdir(exist_ok=True)
        eligible=[i for i,n in enumerate(lengths) if n>=400]
        chosen=np.random.default_rng(71).choice(eligible,64,replace=False)
        for i in chosen:
            f=files[i]
            for source in (f,Path(str(f)+'.metadata.json')):
                dst=smoke/source.name
                if not dst.is_symlink():dst.symlink_to(source)
    print(json.dumps({k:report[k] for k in ('rank','clips','frames','active_clips','initial_frames','root_joint_goal_gpu_gib_pinned')}),flush=True)
    return report

if __name__=='__main__':
    ART.mkdir(parents=True,exist_ok=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as ex:reports=list(ex.map(rank_stats,range(4)))
    (ART/'catalog.json').write_text(json.dumps(dict(passed=True,clips=sum(x['clips'] for x in reports),frames=sum(x['frames'] for x in reports),ranks=reports),indent=2)+'\n')
