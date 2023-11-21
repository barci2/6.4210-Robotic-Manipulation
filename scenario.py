from pydrake.all import (
    RigidTransform,
    RotationMatrix,
    SpatialVelocity,
)
from manipulation.station import add_directives

import numpy as np
import os
import math

"""
List of tuples of YCB obj filenames and the link name within the SDF file

Potential future YCB Objects to add:
 - 11: banana
 - 54: golf ball
"""
OBJECTS = [
    # ("003_cracker_box.sdf", "base_link_cracker"),
    ("004_sugar_box.sdf", "base_link_sugar"),
    ("005_tomato_soup_can.sdf", "base_link_soup"),
    # ("006_mustard_bottle.sdf", "base_link_mustard"),
    # ("009_gelatin_box.sdf", "base_link_gelatin"),
    ("010_potted_meat_can.sdf", "base_link_meat"),
    ("tennis_ball", "Tennis_ball")
    ]

tennis_ball_file = os.path.join(os.getcwd(), "object_files/Tennis_ball.sdf")

# --------------------------------- DIRECTIVES ---------------------------------
"""
Notes:
 - If using iiwa7, weld `iiwa_link_0` to world. If using iiwa14, weld `base` to world.
   Only `link` tags can be welded (?)
"""
scenario_data = """
directives:
- add_model:
    name: iiwa
    file: package://drake/manipulation/models/iiwa_description/urdf/iiwa14_spheres_dense_collision.urdf
    default_joint_positions:
        iiwa_joint_1: [-1.57]
        iiwa_joint_2: [0.5]
        iiwa_joint_3: [0]
        iiwa_joint_4: [-1.2]
        iiwa_joint_5: [0]
        iiwa_joint_6: [ 1.6]
        iiwa_joint_7: [0]
- add_weld:
    parent: world
    child: iiwa::base

- add_model:
    name: wsg
    file: package://drake/manipulation/models/wsg_50_description/sdf/schunk_wsg_50_with_tip.sdf
- add_weld:
    parent: iiwa::iiwa_link_7
    child: wsg::body
    X_PC:
        translation: [0, 0, 0.09]
        rotation: !Rpy { deg: [90, 0, 90]}

- add_model:
    name: camera0
    file: package://manipulation/camera_box.sdf
- add_weld:
    parent: world
    child: camera0::base
    X_PC:
        translation: [2, 1.5, 1.5]
        rotation: !Rpy {deg: [-30, 0, 135]}

model_drivers:
    iiwa: !IiwaDriver
      hand_model_name: wsg
    wsg: !SchunkWsgDriver {}

cameras:
    camera0:
        name: camera0
        clipping_near: 0.1 # To clip out the camera object
        X_PB:
            base_frame: camera0::base
            rotation: !Rpy { deg: [-90, 0, 0]}
    camera1:
        name: camera1
        clipping_near: 0.1 # To clip out the camera object
        X_PB:
            base_frame: camera0::base
            rotation: !Rpy { deg: [-90, 0, 0]}
"""

def throw_object(plant, plant_context, rng, obj_name):
    """
    Move object to throwing position, generate initial velocity for object, then unfreeze its dynamics

    Args:
        plant: MultbodyPlant from hardware station
        plant_context: plant's context
        rng: np.random.default_rng object to generate random positions/velocities
        obj_name: string of the object's name in the scenario YAML (i.e. 'ycb2')
    """

    # Getting relevant data from plant
    model_instance = plant.GetModelInstanceByName(obj_name)  # ModelInstance object
    joint_idx = plant.GetJointIndices(model_instance)[0]  # JointIndex object
    joint = plant.get_joint(joint_idx)  # Joint object

    # Generate random object pose
    z = 0.75  # fixed z for now
    x = rng.uniform(2, 3) * rng.choice([-1, 1])
    y = rng.uniform(2, 3) * rng.choice([-1, 1])

    # Set object pose
    body_idx = plant.GetBodyIndices(model_instance)[0]  # BodyIndex object
    body = plant.get_body(body_idx)  # Body object
    pose = RigidTransform(RotationMatrix(), [x,y,z])
    plant.SetFreeBodyPose(plant_context, body, pose)

    # Unlock joint so object is subject to gravity
    joint.Unlock(plant_context)

    v_magnitude = rng.uniform(5.0, 10.0)
    angle_perturb = rng.uniform(0.1, 0.2) * rng.choice([-1, 1])  # must perturb by at least 0.1 rad to avoid throwing directly at iiwa
    # ensure the perturbation is applied such that it directs the obj away from iiwa
    if x*y > 0:  # x and y have same sign
        cos = x / math.sqrt(x**2 + y**2) + angle_perturb
        sin = y / math.sqrt(x**2 + y**2) - angle_perturb
    else:
        cos = x / math.sqrt(x**2 + y**2) + angle_perturb
        sin = y / math.sqrt(x**2 + y**2) + angle_perturb
    z_perturb = rng.uniform(-0.5, 0.5)
    v_x = -v_magnitude * cos
    v_y = -v_magnitude * sin 
    v_z = 2 + z_perturb

    # Define the spatial velocity
    spatial_velocity = SpatialVelocity(
        v = np.array([v_x, v_y, v_z]),  # m/s
        w = np.array([0, 0, 0]),  # rad/s
    )
    plant.SetFreeBodySpatialVelocity(context = plant_context, body = body, V_WB = spatial_velocity)