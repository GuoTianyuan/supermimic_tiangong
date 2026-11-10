"""Reuse the colleague's original validator against the exact original SONIC bytes."""
from pathlib import Path
import contextlib, io, json, multiprocessing as mp, sys, time
import xml.etree.ElementTree as ET
import mujoco
sys.path.insert(0,str(Path(__file__).resolve().parent))
import prepare_tiangong3 as prep
ROBOT=MODEL=None

def validate(job):
    src,dst=job
    meta=Path(str(dst)+'.metadata.json')
    if dst.is_symlink() and meta.is_file():return 'skip',src.name,''
    try:
        if not dst.exists():dst.symlink_to(src)
        with contextlib.redirect_stdout(io.StringIO()):prep.validate(ROBOT,MODEL,dst)
        return 'ok',src.name,''
    except Exception as e:
        dst.unlink(missing_ok=True);meta.unlink(missing_ok=True)
        return 'fail',src.name,f'{type(e).__name__}: {e}'

def main():
    global ROBOT,MODEL
    src=Path('/media_li/wbc_full');out=Path('/media/supermimic/data/full_wbc_validated');out.mkdir(exist_ok=True)
    urdf=Path(__file__).resolve().parents[1]/'data/tiangong3_urdf/urdf/tiangong3.urdf'
    tree=ET.parse(urdf);ROBOT=tree.getroot()
    for mesh in ROBOT.findall('.//mesh'):mesh.set('filename',str((urdf.parent/mesh.get('filename')).resolve()))
    for old in ROBOT.findall('mujoco'):ROBOT.remove(old)
    ET.SubElement(ET.SubElement(ROBOT,'mujoco'),'compiler',balanceinertia='true',discardvisual='false',fusestatic='false',strippath='false')
    spec=mujoco.MjSpec.from_string(ET.tostring(ROBOT,encoding='unicode'));list(spec.bodies)[1].add_freejoint(name='floating_base_joint');MODEL=spec.compile()
    files=sorted(src.glob('*.npz'));assert len(files)==306948
    started=time.monotonic();counts={'ok':0,'skip':0,'fail':0};failures=[]
    print('Original colleague validation:',len(files),'clips, 32 workers',flush=True)
    with mp.get_context('fork').Pool(32) as pool:
        for i,(status,name,error) in enumerate(pool.imap_unordered(validate,((p,out/p.name) for p in files),chunksize=16),1):
            counts[status]+=1
            if error:failures.append(name+'\t'+error)
            if i%2000==0 or i==len(files):print(i,'/',len(files),counts,f'elapsed={time.monotonic()-started:.0f}s',flush=True)
    (out/'prepare_failures.txt').write_text('\n'.join(failures))
    report={'passed':not failures,'clips':len(files),'counts':counts,'elapsed_seconds':time.monotonic()-started,'directory':str(out),'validator':'original tools/prepare_tiangong3.py validate()'}
    Path('/media/supermimic/artifacts/full_wbc_pool_20261111/dataset_validation.json').write_text(json.dumps(report,indent=2))
    if failures:raise SystemExit('Some clips failed original validation; training was not started.')

if __name__=='__main__':main()
