"""
Output ports:
 - Trajectory object
 - A downsampled PointCloud object (containing just the object) in Object frame
"""

from typing import Optional, Tuple, List
from pydrake.all import (
    AbstractValue,
    Trajectory,
    PointCloud,
    DiagramBuilder,
    RgbdSensor,
    MultibodyPlant,
    CameraConfig,
    RigidTransform,
    Diagram,
    RollPitchYaw,
    MakeRenderEngineGl,
    LeafSystem,
    AbstractValue,
    Context,
    InputPort,
    ImageDepth32F,
    ImageLabel16I,
    PointCloud,
    OutputPort,
    Rgba,
    Meshcat,
    PiecewisePolynomial
)
import numpy as np
import numpy.typing as npt
import itertools
import matplotlib.pyplot as plt


def add_cameras(
        builder: DiagramBuilder,
        station: Diagram,
        plant: MultibodyPlant,
        camera_width: int,
        camera_height: int,
        horizontal_num: int,
        vertical_num: int,
        camera_distance: float
    ) -> Tuple[List[RgbdSensor], List[RigidTransform]]:
    camera_config = CameraConfig()
    camera_config.width = camera_width
    camera_config.height = camera_height
    scene_graph = station.GetSubsystemByName("scene_graph")
    if not scene_graph.HasRenderer(camera_config.renderer_name):
        scene_graph.AddRenderer(
            camera_config.renderer_name, MakeRenderEngineGl())

    camera_systems = []
    camera_transforms = []
    thetas = np.linspace(0, 2*np.pi, horizontal_num, endpoint=False)
    phis = np.linspace(0, -np.pi, vertical_num + 2)[1:-1]
    for idx, (theta, phi) in enumerate(itertools.product(thetas, phis)):
        name = f"camera{idx}"
        transform = RigidTransform(RollPitchYaw(0, 0, theta).ToRotationMatrix(
        ) @ RollPitchYaw(phi, 0, 0).ToRotationMatrix(), np.zeros(3)) @ RigidTransform([0, 0, -camera_distance])

        _, depth_camera = camera_config.MakeCameras()
        camera_sys = builder.AddSystem(RgbdSensor(
            parent_id=plant.GetBodyFrameIdIfExists(
                plant.world_frame().body().index()),
            X_PB=transform,
            depth_camera=depth_camera
        ))
        builder.Connect(
            station.GetOutputPort(
                "query_object"), camera_sys.query_object_input_port()
        )
        builder.ExportOutput(
            camera_sys.color_image_output_port(), f"{name}.rgb_image"
        )
        builder.ExportOutput(
            camera_sys.depth_image_32F_output_port(), f"{name}.depth_image"
        )
        builder.ExportOutput(
            camera_sys.label_image_output_port(), f"{name}.label_image"
        )
        camera_sys.set_name(name)
        camera_systems.append(camera_sys)
        camera_transforms.append(transform)

    return camera_systems, camera_transforms


class TrajectoryPredictor(LeafSystem):
    """
    Performs ICP after first keying out the objects
    """
    def __init__(
            self,
            cameras: List[RgbdSensor],
            camera_transforms: List[RigidTransform],
            pred_thresh: int,
            thrown_model_name: int,
            plant: MultibodyPlant,
            meshcat: Optional[Meshcat] = None
        ):
        super().__init__()
        self._pred_thresh = pred_thresh
        self._num_cameras = len(cameras)
        self._camera_infos = [camera.depth_camera_info() for camera in cameras]
        self._camera_transforms = camera_transforms
        self._meshcat = meshcat

        # Index of the object
        model_idx = plant.GetModelInstanceByName(thrown_model_name)
        body_idx, = map(int, plant.GetBodyIndices(model_idx))
        self._obj_idx = body_idx

        # Camera inputs
        self._camera_depth_inputs = [self.DeclareAbstractInputPort(
            f"camera{i}.depth_input", AbstractValue.Make(ImageDepth32F(camera_info.width(), camera_info.height())))
            for i, camera_info in enumerate(self._camera_infos)]

        self._camera_label_inputs = [self.DeclareAbstractInputPort(
            f"camera{i}.label_input", AbstractValue.Make(ImageLabel16I(camera_info.width(), camera_info.height())))
            for i, camera_info in enumerate(self._camera_infos)]

        # Saved previous poses
        self._poses_state_index = self.DeclareAbstractState(AbstractValue.Make([0.0, 0.0, 0.0]))

        # Update Event
        self.DeclarePeriodicPublishEvent(0.1, 0.001, self.PredictTrajectory)

        # Michael commented out `lambda c, o: None` and added `self.CreateOutput`
        port = self.DeclareAbstractOutputPort(
            "object_trajectory",
            lambda: AbstractValue.Make((Trajectory())),
            # lambda c, o: None,
            self.CreateOutput,
        )

    def camera_input_ports(self, camera_idx: int) -> Tuple[InputPort, InputPort]:
        return self._camera_depth_inputs[camera_idx], self._camera_label_inputs[camera_idx]

    def point_cloud_output_port(self) -> OutputPort:
        return self._point_cloud_output

    def PredictTrajectory(self, context: Context):
        points = np.concatenate([self.ExtractPointCloudForCamera(context, camera_idx) for camera_idx in range(self._num_cameras)], axis=1)
        if self._meshcat is not None:
            self.PublishMeshcat(points, self._meshcat)

    @staticmethod
    def PublishMeshcat(points: npt.NDArray[np.float32], meshcat: Meshcat):
        cloud = PointCloud(points.shape[1])
        if points.shape[1] > 0:
            cloud.mutable_xyzs()[:] = points
        meshcat.SetObject("TrajectoryPredictorPointCloud", cloud, point_size=0.01, rgba=Rgba(1, 0.5, 0.5))

    def ExtractPointCloudForCamera(self, context: Context, camera_idx: int) -> npt.NDArray[np.float32]:
        camera_info = self._camera_infos[camera_idx]
        transform = self._camera_transforms[camera_idx]

        height = camera_info.height()
        width = camera_info.width()
        center_x = camera_info.center_x()
        center_y = camera_info.center_y()
        focal_x = camera_info.focal_x()
        focal_y = camera_info.focal_y()

        depth_img = self._camera_depth_inputs[camera_idx].Eval(context).data
        label_img = self._camera_label_inputs[camera_idx].Eval(context).data
        u_coords, v_coords, _ = np.meshgrid(np.arange(width), np.arange(height), [0], copy=False)
        distances_coords = np.stack([u_coords, v_coords, depth_img], axis=-1)
        depth_pixel = distances_coords[np.logical_and(label_img == self._obj_idx, np.abs(depth_img) != np.inf)]

        u = depth_pixel[:, 0]
        v = depth_pixel[:, 1]
        z = depth_pixel[:, 2]

        x = (u - center_x) * z / focal_x
        y = (v - center_y) * z / focal_y
        pC = np.stack([x, y, z])
        return (transform @ pC)

    # Michael added this function to test connecting the two leafsystems
    def CreateOutput(self, context, output):
        t = 0
        test_obj_traj = PiecewisePolynomial.FirstOrderHold(
                    [t, t + 1],  # Time knots
                    np.array([[-1, 0.65], [-1, 0], [0.75, 0.75], [0, 0], [0, 0], [0, 0], [1, 1]])
                    )