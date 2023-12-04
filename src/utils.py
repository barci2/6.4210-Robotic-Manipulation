""" Miscellaneous Utility functions """
from enum import Enum
from typing import BinaryIO, Optional, Union
from pydrake.all import (
    DiagramBuilder,
    Diagram,
    RigidTransform,
    RotationMatrix,
    SpatialVelocity,
    MultibodyPlant,
    Context,
    CameraConfig
)
from pydrake.common.yaml import yaml_load_typed
import numpy as np
import pydot
import matplotlib.pyplot as plt


def diagram_update_meshcat(diagram, context=None) -> None:
    if context is None:
        context = diagram.CreateDefaultContext()
    diagram.ForcedPublish(context)


def diagram_visualize_connections(diagram: Diagram, file: Union[BinaryIO, str]) -> None:
    """
    Create SVG file of system diagram.
    """
    if type(file) is str:
        file = open(file, "bw")
    graphviz_str = diagram.GetGraphvizString()
    svg_data = pydot.graph_from_dot_data(
        diagram.GetGraphvizString())[0].create_svg()
    file.write(svg_data)


def visualize_camera_plt(diagram: Diagram, camera_name: str, context=None, plt_show: bool = True) -> Optional[plt.Figure]:
    """
    Show Camera view using matplotlib.
    """
    if context is None:
        context = diagram.CreateDefaultContext()
    image = diagram.GetOutputPort(
        f"{camera_name}.rgb_image").Eval(context).data
    fig, ax = plt.subplots()
    ax.imshow(image)
    if not plt_show:
        return ax
    plt.show()


def throw_object(plant: MultibodyPlant, plant_context: Context, obj_name: str) -> None:
    """
    Move object to throwing position, generate initial velocity for object, then unfreeze its dynamics

    Args:
        plant: MultbodyPlant from hardware station
        plant_context: plant's context
        obj_name: string of the object's name in the scenario YAML (i.e. 'ycb2')
    """

    # Getting relevant data from plant
    model_instance = plant.GetModelInstanceByName(
        obj_name)  # ModelInstance object
    joint_idx = plant.GetJointIndices(model_instance)[0]  # JointIndex object
    joint = plant.get_joint(joint_idx)  # Joint object

    # Generate random object pose
    z = 0.75  # fixed z for now
    x = np.random.uniform(3.5, 4) * np.random.choice([-1, 1])
    y = np.random.uniform(3.5, 4) * np.random.choice([-1, 1])

    # Set object pose
    body_idx = plant.GetBodyIndices(model_instance)[0]  # BodyIndex object
    body = plant.get_body(body_idx)  # Body object
    pose = RigidTransform(RotationMatrix(), [x, y, z])
    plant.SetFreeBodyPose(plant_context, body, pose)

    # Unlock joint so object is subject to gravity
    joint.Unlock(plant_context)

    v_magnitude = np.random.uniform(5.0, 5.5)
    angle_perturb = np.random.uniform(0.075, 0.1) * np.random.choice(
        [-1, 1]
    )  # must perturb by at least 0.1 rad to avoid throwing directly at iiwa
    # ensure the perturbation is applied such that it directs the obj away from iiwa
    if x * y > 0:  # x and y have same sign
        cos_alpha = x / np.sqrt(x**2 + y**2) + angle_perturb
        sin_alpha = y / np.sqrt(x**2 + y**2) - angle_perturb
    else:
        cos_alpha = x / np.sqrt(x**2 + y**2) + angle_perturb
        sin_alpha = y / np.sqrt(x**2 + y**2) + angle_perturb
    z_perturb = np.random.uniform(-0.5, 0.5)
    v_x = -v_magnitude * cos_alpha
    v_y = -v_magnitude * sin_alpha
    v_z = 4 + z_perturb

    # Define the spatial velocity
    spatial_velocity = SpatialVelocity(
        v=np.array([v_x, v_y, v_z]),  # m/s
        w=np.array([0, 0, 0]),  # rad/s
    )
    plant.SetFreeBodySpatialVelocity(
        context=plant_context, body=body, V_WB=spatial_velocity
    )
