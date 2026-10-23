"""Convert the supplied Tiangong3 URDF, validate NPZ ordering, and create USD."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]

def convert():
    src = ROOT/'data/tiangong3_urdf/urdf/tiangong3.urdf'
    out = ROOT/'data/tiangong3_generated'
    out.mkdir(parents=True, exist_ok=True)
    tree = ET.parse(src)
    robot = tree.getroot()
    for mesh in robot.findall('.//mesh'):
        path = (src.parent/mesh.get('filename')).resolve()
        if not path.is_file(): raise FileNotFoundError(path)
        mesh.set('filename', str(path))
    for old in robot.findall('mujoco'): robot.remove(old)
    compiler = ET.SubElement(ET.SubElement(robot,'mujoco'),'compiler',
        balanceinertia='true', discardvisual='false', fusestatic='false', strippath='false')
    tmp = out/'tiangong3.import.urdf'
    tree.write(tmp)
    m = mujoco.MjModel.from_xml_path(str(tmp))
    xml = out/'tiangong3.xml'
    mujoco.mj_saveLastXML(str(xml),m)
    t = ET.parse(xml)
    pelvis = t.getroot().find("./worldbody/body[@name='pelvis']")
    if pelvis is None: raise RuntimeError('Importer did not preserve pelvis')
    ET.SubElement(pelvis,'freejoint',name='floating_base_joint')
    t.write(xml)
    m = mujoco.MjModel.from_xml_path(str(xml))
    if m.nq != 36 or m.nv != 35: raise ValueError((m.nq,m.nv))
    print('Converted MJCF:',xml,'mass:',m.body_mass.sum(),flush=True)
    return robot,m,xml

def validate(robot,m,path):
    with np.load(path,allow_pickle=False) as a:
        data={k:a[k] for k in a.files}
    for k in ('joint_pos','joint_vel','body_pos_w','body_quat_w','body_lin_vel_w','body_ang_vel_w'):
        if not np.isfinite(data[k]).all(): raise ValueError(f'Nonfinite {k}')
    fps=float(data['fps'].reshape(-1)[0])
    if fps<=0 or data['joint_pos'].shape[1]!=29: raise ValueError('Invalid fps/joint count')
    children={}
    for j in robot.findall('joint'): children.setdefault(j.find('parent').get('link'),[]).append(j)
    order=[];queue=['pelvis']
    while queue:
        for j in children.get(queue.pop(0),[]):
            queue.append(j.find('child').get('link'))
            if j.get('type')!='fixed': order.append(j.get('name'))
    if 'joint_names' in data:
        order=[str(x) for x in data['joint_names']]
    targets=data['body_pos_w'];quats=data['body_quat_w'];T=len(targets)
    ids=np.linspace(0,T-1,min(T,41)).astype(int)
    model_names=[m.body(i).name for i in range(1,m.nbody)]
    canonical=[j.get('name') for j in robot.findall('joint') if j.get('type')!='fixed']
    d=mujoco.MjData(m)
    pred=[];pred_q=[]
    for f in ids:
        d.qpos[:3]=targets[f,0];d.qpos[3:7]=quats[f,0]
        for n in canonical:
            d.qpos[m.jnt_qposadr[m.joint(n).id]]=data['joint_pos'][f,order.index(n)]
        mujoco.mj_forward(m,d)
        pred.append(d.xpos[1:].copy());pred_q.append(d.xquat[1:].copy())
    pred=np.array(pred);pred_q=np.array(pred_q)
    cost=np.linalg.norm(pred[:,:,None,:]-targets[ids,None,:,:],axis=-1).mean(0)
    # Orientations disambiguate nearby ankle and fixed-link origins.
    dots=np.abs(np.einsum('tbi,tci->tbc',pred_q,quats[ids]))
    angles=2*np.arccos(dots.clip(0,1))
    r,c=linear_sum_assignment(cost+0.05*angles.mean(0))
    if len(r)!=len(model_names) or int(c[r.tolist().index(0)])!=0:
        raise ValueError('Body count or root mapping mismatch')
    pos_error=np.linalg.norm(pred[:,r]-targets[ids][:,c],axis=-1)
    rot_error=angles[:,r,c]
    print('Body rotation errors:', {model_names[int(i)]:float(rot_error[:,k].max()) for k,i in enumerate(r) if rot_error[:,k].max()>.03}, flush=True)
    moving={j.find('child').get('link') for j in robot.findall('joint') if j.get('type')!='fixed'} | {'pelvis'}
    moving_rot=rot_error[:,[k for k,i in enumerate(r) if model_names[int(i)] in moving]]
    if pos_error.max()>0.015 or moving_rot.max()>0.03:
        raise ValueError(f'NPZ/URDF mismatch: pos={pos_error.max()} rot={rot_error.max()}')
    body_names=[None]*len(model_names)
    for i,k in zip(r,c): body_names[int(k)]=model_names[int(i)]
    meta={'robot':'tiangong3','joint_names':order,'body_names':body_names,
        'quaternion_format':'wxyz','root_body_index':0,'fps':fps,'frames':T,
        'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'validation_frames':len(ids),'max_position_error_m':float(pos_error.max()),
        'mean_position_error_m':float(pos_error.mean()),'max_rotation_error_rad':float(rot_error.max()),'max_moving_rotation_error_rad':float(moving_rot.max())}
    Path(str(path)+'.metadata.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps({k:v for k,v in meta.items() if k not in ('joint_names','body_names')},indent=2),flush=True)
    # Enforce joint ranges before training; small retargeting overshoots are reported.
    q=data['joint_pos'][:,[order.index(n) for n in canonical]]
    ranges=np.array([m.jnt_range[m.joint(n).id] for n in canonical])
    violation=np.maximum(ranges[:,0]-q,0)+np.maximum(q-ranges[:,1],0)
    print('Max joint-limit overshoot (rad):',float(violation.max()),flush=True)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--motion_file',type=Path,required=True)
    ap.add_argument('--skip_usd',action='store_true')
    args=ap.parse_args()
    robot,m,xml=convert()
    validate(robot,m,args.motion_file.resolve())
    if not args.skip_usd:
        from pxr import Plug
        import isaacsim.asset.importer.utils.impl as iu
        Plug.Registry().RegisterPlugins(os.path.join(os.path.dirname(iu.__file__),'schemas'))
        from isaacsim.asset.importer.mjcf import MJCFImporter,MJCFImporterConfig
        cfg=MJCFImporterConfig(mjcf_path=str(xml),usd_path=str(ROOT/'data/tiangong3_usd'))
        print(MJCFImporter(cfg).import_mjcf(),flush=True)

if __name__=='__main__':main()
