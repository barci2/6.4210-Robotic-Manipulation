"""
Output ports:
 - Trajectory object
 - A downsampled PointCloud object (containing just the object) in Object frame
"""

from pydrake.all import (
    DiagramBuilder,
    RgbdSensor,
    MultibodyPlant,
    CameraConfig,
    RigidTransform,
    Diagram,
    RollPitchYaw,
    MakeRenderEngineGl,
    LeafSystem
)
import numpy as np
import itertools


def add_cameras(builder: DiagramBuilder, station: Diagram, plant: MultibodyPlant, h_num: int, v_num: int) -> list[Diagram]:
    camera_config = CameraConfig()
    scene_graph = station.GetSubsystemByName("scene_graph")
    if not scene_graph.HasRenderer(camera_config.renderer_name):
        scene_graph.AddRenderer(camera_config.renderer_name, MakeRenderEngineGl())

    camera_systems = []
    thetas = np.linspace(0, 2*np.pi, h_num, endpoint=False)
    phis = np.linspace(0, -np.pi, v_num + 2)[1:-1]
    for idx, (theta, phi) in enumerate(itertools.product(thetas, phis)):
        name = f"camera{idx}"
        transform = RigidTransform(RollPitchYaw(0, 0, theta).ToRotationMatrix() @ RollPitchYaw(phi, 0, 0).ToRotationMatrix(), np.zeros(3)) @ RigidTransform([0, 0, -2])

        _, depth_camera = camera_config.MakeCameras()
        camera_sys = builder.AddSystem(RgbdSensor(
            parent_id=plant.GetBodyFrameIdIfExists(
                plant.world_frame().body().index()),
            X_PB=transform,
            depth_camera=depth_camera
        ))
        builder.Connect(
            station.GetOutputPort("query_object"), camera_sys.query_object_input_port()
        )
        builder.ExportOutput(
            camera_sys.color_image_output_port(), f"{name}.rgb_image"
        )
        builder.ExportOutput(
            camera_sys.depth_image_32F_output_port(), f"{name}.depth_image"
        )
        camera_sys.set_name(name)
        camera_systems.append(camera_sys)


class TrajectoryPredictor(LeafSystem):
    """ Performs ICP after first keying out the objects
    """

    def __init__(self):
        pass