"""
States:
 - X_WG trajectory (Piecewise Pose?)
 - WSG trajectory (PiecewisePolynomial?)
 - Times?

Input Ports:
 - Grasp pose
 - Current pose

Output Ports: 
 - 

"""

from pydrake.all import (
    AbstractValue,
    Concatenate,
    Trajectory,
    InputPortIndex,
    LeafSystem,
    PointCloud,
    RandomGenerator,
    RigidTransform,
    RollPitchYaw,
    Simulator,
    StartMeshcat,
    UniformlyRandomRotationMatrix,
)
import numpy as np

class MotionPlanner(LeafSystem):
    """
    Perform Constrained Optimization to find optimal trajectory for iiwa to move
    to the grasping position.
    """

    def __init__(self, plant):
        LeafSystem.__init__(self)

        self._gripper_body_index = plant.GetBodyByName("body").index()

        X_WG_grasp = AbstractValue.Make(RigidTransform())
        body_poses = AbstractValue.Make([RigidTransform()])  # Use this to get current gripper pose using body_poses.Eval(context)[self._gripper_body_index]
        
        self.DeclareAbstractInputPort("grasp_selections", X_WG_grasp)
        self.DeclareAbstractInputPort("body_poses", body_poses)