"""Tiangong3 29-DoF asset. Gains, initial pose and action scale follow the
DexEVT config in cheerleading_competition (robots/dex_evt.py).

Effort/velocity limits come from the supplied URDF, except the parallel ankle.
This uses implicit PD without a torque-speed model; actuator delay, joint
friction and gain/mass/armature errors are opt-in randomization (supermimic/dr.py).
"""
import os
import re
import xml.etree.ElementTree as ET

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
URDF_PATH = os.path.join(DATA_DIR, 'tiangong3_urdf', 'urdf', 'tiangong3.urdf')
MJCF_PATH = os.path.join(DATA_DIR, 'tiangong3_generated', 'tiangong3.xml')
USD_PATH = os.path.join(DATA_DIR, 'tiangong3_usd', 'tiangong3', 'tiangong3.usda')
USD_DIR = os.path.dirname(os.path.dirname(USD_PATH))
_tree = ET.parse(URDF_PATH).getroot()
# MJCF/Isaac Lab kitless expose moving joints in tree (depth-first) order.
G1_JOINT_NAMES = [j.get('name') for j in _tree.findall('joint') if j.get('type') != 'fixed']
JOINT_NAMES = G1_JOINT_NAMES
TRACKED_BODIES = [
    'pelvis', 'hip_roll_l_link', 'knee_pitch_l_link', 'ankle_roll_l_link',
    'hip_roll_r_link', 'knee_pitch_r_link', 'ankle_roll_r_link', 'waist_pitch_link',
    'shoulder_roll_l_link', 'elbow_pitch_l_link', 'wrist_roll_l_link',
    'shoulder_roll_r_link', 'elbow_pitch_r_link', 'wrist_roll_r_link',
]
TORSO_BODY = 'waist_pitch_link'
DEFAULT_MOTION = os.path.join(DATA_DIR, 'tiangong_motion')
DEFAULT_JOINT_POS = {
    'hip_pitch_[lr]_joint': -0.25, 'knee_pitch_[lr]_joint': 0.5,
    'ankle_pitch_[lr]_joint': -0.25, 'elbow_pitch_[lr]_joint': -0.3,
    'shoulder_roll_l_joint': 0.3, 'shoulder_roll_r_joint': -0.3,
}
# (Kp, Kd)
_PD = {
    'hip_pitch': (900.,57.), 'hip_roll': (900.,57.), 'hip_yaw': (660.,42.),
    'knee_pitch': (1020.,72.), 'ankle_pitch': (55.,3.4), 'ankle_roll': (55.,3.4),
    'waist_yaw': (670.,41.), 'waist_roll': (670.,41.), 'waist_pitch': (670.,41.),
    'shoulder_pitch': (100.,7.4), 'shoulder_roll': (100.,7.4), 'shoulder_yaw': (90.,5.9),
    'elbow_pitch': (90.,5.9), 'elbow_yaw': (100.,2.4), 'wrist_pitch': (100.,2.4), 'wrist_roll': (100.,2.4),
}
# Each serial ankle axis is driven by two 50 N*m crank motors; the action scale
# keeps DexEVT's earlier 55 N*m per-joint limit.
ANKLE_EFFORT, ANKLE_SCALE_EFFORT = 100., 55.
# Midpoints of DexEVT's start-up armature ranges (0.002-0.06, ankle 0.01-0.06).
ARMATURE, ANKLE_ARMATURE = .031, .035
_limits = {j.get('name'): j.find('limit') for j in _tree.findall('joint') if j.get('type') != 'fixed'}

# Joint dynamics presets (train.py --dynamics):
#   default: 100 N*m ankles, ankle armature 0.01-0.06, Coulomb joint friction only;
#   dexevt: the current DexEVT (cheerleading_competition robots/dex_evt.py + EventCfg): 55 N*m ankles,
#           armature 0.002-0.06 on every joint, and viscous joint friction from the Coulomb range too
#           (Isaac Lab on Isaac Sim 5 draws static, dynamic and viscous friction; here one draw per joint).
DYNAMICS = ('default', 'dexevt')
_dynamics = 'default'

def _key(name):
    return re.sub(r'_[lr]_joint$', '', name).removesuffix('_joint')

def motor_params(name):
    kp, kd = _PD[_key(name)]
    ankle = _key(name).startswith('ankle')
    lim = _limits[name]
    if _dynamics == 'dexevt':
        return ANKLE_SCALE_EFFORT if ankle else float(lim.get('effort')), float(lim.get('velocity')), kp, kd, ARMATURE
    effort = ANKLE_EFFORT if ankle else float(lim.get('effort'))
    return effort, float(lim.get('velocity')), kp, kd, ANKLE_ARMATURE if ankle else ARMATURE

def action_scale(name):
    effort, _, kp, _, _ = motor_params(name)
    return 0.5 * (ANKLE_SCALE_EFFORT if _key(name).startswith('ankle') else effort) / kp

def default_joint_pos_list():
    return [next((v for pat,v in DEFAULT_JOINT_POS.items() if re.fullmatch(pat,n)),0.) for n in JOINT_NAMES]

# Rigid segments for mass randomization (supermimic/dr.py): the root and the child link of each moving
# joint. Links on fixed joints (IMUs, cameras, head, TCPs) scale with the segment they hang from.
_parent = {j.find('child').get('link'): (j.find('parent').get('link'), j.get('type')) for j in _tree.findall('joint')}
MASS_SEGMENTS = [l.get('name') for l in _tree.findall('link') if l.get('name') not in _parent] + \
    [j.find('child').get('link') for j in _tree.findall('joint') if j.get('type') != 'fixed']

def mass_segment(link):
    while _parent.get(link, (None, None))[1] == 'fixed':
        link = _parent[link][0]
    return MASS_SEGMENTS.index(link)

# Robot-ground friction drawn per environment: DexEVT's static friction range (EventCfg physics_material),
# which holds xsim_mujoco's 1.5.
FRICTION_RANGE = (0.6, 1.6)

def _actuators():
    acts = {}
    for name in JOINT_NAMES:
        effort, vel, kp, kd, arm = motor_params(name)
        acts[name] = ImplicitActuatorCfg(joint_names_expr=[name], effort_limit_sim=effort,
            velocity_limit_sim=vel, stiffness=kp, damping=kd, armature=arm)
    return acts

def set_dynamics(name):
    """Switch the joint dynamics preset (see DYNAMICS); call before building the simulator."""
    global _dynamics
    if name not in DYNAMICS:
        raise ValueError(f"unknown dynamics {name!r}; choose from {', '.join(DYNAMICS)}")
    _dynamics = name
    G1_CFG.actuators = _actuators()

def get_dynamics():
    return _dynamics

acts = _actuators()
G1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(usd_path=USD_PATH, variants={'Physics':'mujoco'},
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False,linear_damping=0.,angular_damping=0.),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=True)),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.,0.,.95),joint_pos=DEFAULT_JOINT_POS,joint_vel={'.*':0.}),
    soft_joint_pos_limit_factor=.9, actuators=acts,
)
