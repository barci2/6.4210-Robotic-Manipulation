""" Miscellaneous Utility functions """
from enum import Enum
from typing import BinaryIO, Optional, Union, Tuple
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
from dataclasses import dataclass, field
from pydrake.common.yaml import yaml_load_typed
import numpy as np
import numpy.typing as npt
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


def throw_object1(plant: MultibodyPlant, plant_context: Context, obj_name: str) -> None:
    """
    Version 1 (from further away, higher velocity)

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
    z = 0.5  # fixed z for now
    x = np.random.uniform(2.6, 2.7) * np.random.choice([-1, 1])
    y = np.random.uniform(2.6, 2.7) * np.random.choice([-1, 1])

    # Set object pose
    body_idx = plant.GetBodyIndices(model_instance)[0]  # BodyIndex object
    body = plant.get_body(body_idx)  # Body object
    pose = RigidTransform(RotationMatrix(), [x, y, z])
    plant.SetFreeBodyPose(plant_context, body, pose)

    # Unlock joint so object is subject to gravity
    joint.Unlock(plant_context)

    v_magnitude = np.random.uniform(4.75, 5.0)
    angle_perturb = np.random.uniform(0.11, 0.12) * np.random.choice(
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
    v_z = 3.8 + z_perturb

    # Define the spatial velocity
    spatial_velocity = SpatialVelocity(
        v=np.array([v_x, v_y, v_z]),  # m/s
        w=np.array([0, 0, 0]),  # rad/s
    )
    plant.SetFreeBodySpatialVelocity(
        context=plant_context, body=body, V_WB=spatial_velocity
    )


def throw_object2(plant: MultibodyPlant, plant_context: Context, obj_name: str) -> None:
    """
    Version 2 (closer, lower velocity, catching closer to peak of traj)

    Move object to throwing position, generate initial velocity for object, then unfreeze its dynamics

    Args:
        plant: MultbodyPlant from hardware station
        plant_context: plant's context
        obj_name: string of the object's name in the scenario YAML (i.e. 'ycb2')
    """

    # Getting relevant data from plant
    model_instance = plant.GetModelInstanceByName(
        obj_name)  # ModelInstancendex object
    joint_idx = plant.GetJointIndices(model_instance)[0]  # JointIndex object
    joint = plant.get_joint(joint_idx)  # Joint object

    # Generate random object pose
    z = 0.25  # fixed z for now
    x = np.random.uniform(1.7, 1.9) * np.random.choice([-1, 1])
    y = np.random.uniform(1.7, 1.9) * np.random.choice([-1, 1])

    # Set object pose
    body_idx = plant.GetBodyIndices(model_instance)[0]  # BodyIndex object
    body = plant.get_body(body_idx)  # Body object
    pose = RigidTransform(RotationMatrix(), [x, y, z])
    plant.SetFreeBodyPose(plant_context, body, pose)

    # Unlock joint so object is subject to gravity
    joint.Unlock(plant_context)

    v_magnitude = np.random.uniform(3.9, 4.0)
    angle_perturb = np.random.uniform(0.10, 0.11) * np.random.choice(
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
    v_z = 3.8 + z_perturb

    # Define the spatial velocity
    spatial_velocity = SpatialVelocity(
        v=np.array([v_x, v_y, v_z]),  # m/s
        w=np.array([0, 0, 0]),  # rad/s
    )
    plant.SetFreeBodySpatialVelocity(
        context=plant_context, body=body, V_WB=spatial_velocity
    )

@dataclass
class ObjectTrajectory:
    x: Tuple[np.float32, np.float32, np.float32] = (0, 0, 0)
    y: Tuple[np.float32, np.float32, np.float32] = (0, 0, 0)
    z: Tuple[np.float32, np.float32, np.float32] = (0, 0, 0)

    @staticmethod
    def _solve_single_traj(
            a: np.float32,
            x1: np.float32,
            t1: np.float32,
            x2: np.float32,
            t2: np.float32
        ) -> Tuple[np.float32, np.float32, np.float32]:
        return (a, *np.linalg.solve([[t1, 1], [t2, 1]], [x1 - a * t1 ** 2, x2 - a * t2 ** 2]))

    @staticmethod
    def CalculateTrajectory(
            X1: RigidTransform,
            t1: np.float32,
            X2: RigidTransform,
            t2: np.float32,
            g: np.float32 = 9.81,
        ) -> "ObjectTrajectory":
        p1 = X1.translation()
        p2 = X2.translation()
        return ObjectTrajectory(
            ObjectTrajectory._solve_single_traj(0, p1[0], t1, p2[0], t2),
            ObjectTrajectory._solve_single_traj(0, p1[1], t1, p2[1], t2),
            ObjectTrajectory._solve_single_traj(-g/2, p1[2], t1, p2[2], t2)
        )

    def value(self, t: np.float32) -> RigidTransform:
        return RigidTransform([
            self.x[0] * t ** 2 + self.x[1] * t + self.x[2],
            self.y[0] * t ** 2 + self.y[1] * t + self.y[2],
            self.z[0] * t ** 2 + self.z[1] * t + self.z[2],
        ])
    
    def EvalDerivative(self, t: np.float32) -> npt.NDArray[np.float32]:
        return np.array([
            2 * self.x[0] * t + self.x[1],
            2 * self.y[0] * t + self.y[1],
            2 * self.z[0] * t + self.z[1]
        ])