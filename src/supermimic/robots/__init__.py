"""Select the robot once per process, before importing the environment or motion library."""
import os
ROBOT_NAME = os.environ.get("SUPERMIMIC_ROBOT", "g1")
if ROBOT_NAME == "g1":
    from .g1 import *
elif ROBOT_NAME == "tiangong3":
    from .tiangong3 import *
else:
    raise ValueError(f"Unknown robot: {ROBOT_NAME}")
JOINT_NAMES = G1_JOINT_NAMES
ROBOT_CFG = G1_CFG
