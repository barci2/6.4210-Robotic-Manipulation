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
    PiecewisePolynomial,
    Value
)
import numpy as np
import numpy.typing as npt
import itertools
import matplotlib.pyplot as plt


group_idx = 0
def add_cameras(
        builder: DiagramBuilder,
        station: Diagram,
        plant: MultibodyPlant,
        camera_width: int,
        camera_height: int,
        horizontal_num: int,
        vertical_num: int,
        camera_distance: float,
        cameras_center: npt.NDArray[np.float32]
    ) -> Tuple[List[RgbdSensor], List[RigidTransform]]:
    global group_idx
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
        name = f"camera{idx}_group{group_idx}"
        transform = RigidTransform(RollPitchYaw(0, 0, theta).ToRotationMatrix(
        ) @ RollPitchYaw(phi, 0, 0).ToRotationMatrix(), cameras_center) @ RigidTransform([0, 0, -camera_distance])

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

    group_idx += 1
    return camera_systems, camera_transforms

class CameraBackedSystem(LeafSystem):
    def __init__(
            self,
            cameras: List[RgbdSensor],
            camera_transforms: List[RigidTransform],
            pred_thresh: int,
            thrown_model_name: str,
            plant: MultibodyPlant,
            meshcat: Optional[Meshcat] = None,
        ):
        super().__init__()

        self._pred_thresh = pred_thresh
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

    def camera_input_ports(self, camera_idx: int) -> Tuple[InputPort, InputPort]:
        return self._camera_depth_inputs[camera_idx], self._camera_label_inputs[camera_idx]

    def ConnectCameras(self, builder: DiagramBuilder, cameras: List[RgbdSensor]):
        for camera, depth_input, label_input in zip(cameras, self._camera_depth_inputs, self._camera_label_inputs):
            builder.Connect(camera.depth_image_32F_output_port(), depth_input)
            builder.Connect(camera.label_image_output_port(), label_input)

    def GetCameraPoints(self, context: Context) -> npt.NDArray[np.float32]:
        total_point_cloud = np.zeros((3, 0))
        for camera_info, transform, depth_input, label_input in zip(
                self._camera_infos,
                self._camera_transforms,
                self._camera_depth_inputs,
                self._camera_label_inputs
            ):
            height = camera_info.height()
            width = camera_info.width()
            center_x = camera_info.center_x()
            center_y = camera_info.center_y()
            focal_x = camera_info.focal_x()
            focal_y = camera_info.focal_y()

            depth_img = depth_input.Eval(context).data
            label_img = label_input.Eval(context).data
            u_coords, v_coords, _ = np.meshgrid(np.arange(width), np.arange(height), [0], copy=False)
            distances_coords = np.stack([u_coords, v_coords, depth_img], axis=-1)
            depth_pixel = distances_coords[np.logical_and(label_img == self._obj_idx, np.abs(depth_img) != np.inf)]

            u = depth_pixel[:, 0]
            v = depth_pixel[:, 1]
            z = depth_pixel[:, 2]

            x = (u - center_x) * z / focal_x
            y = (v - center_y) * z / focal_y
            pC = np.stack([x, y, z])

            if pC.shape[1] >= self._pred_thresh:
                total_point_cloud = np.concatenate([total_point_cloud, (transform @ pC)], axis=-1)
        return total_point_cloud

    def PublishMeshcat(self, points: npt.NDArray[np.float32]):
        cloud = PointCloud(points.shape[1])
        if points.shape[1] > 0:
            cloud.mutable_xyzs()[:] = points
        if self._meshcat is not None:
            self._meshcat.SetObject("TrajectoryPredictorPointCloud", cloud, point_size=0.01, rgba=Rgba(1, 0.5, 0.5))

class PointCloudGenerator(CameraBackedSystem):
    def __init__(
            self,
            cameras: List[RgbdSensor],
            camera_transforms: List[RigidTransform],
            cameras_center: npt.NDArray[np.float32],
            pred_thresh: int,
            thrown_model_name: str,
            plant: MultibodyPlant,
            meshcat: Optional[Meshcat] = None,
        ):
        super().__init__(
            cameras=cameras,
            camera_transforms=camera_transforms,
            pred_thresh=pred_thresh,
            thrown_model_name=thrown_model_name,
            plant=plant,
            meshcat=meshcat
        )

        self._cameras_center = cameras_center
        self._point_cloud = PointCloud()
        self._point_cloud_output = self.DeclareAbstractOutputPort(
            'point_cloud',
            lambda: AbstractValue.Make(PointCloud()),
            self.OutputPointCloud
        )

    @property
    def point_cloud_output_port(self) -> OutputPort:
        return self._point_cloud_output

    def OutputPointCloud(self, context: Context, output: Value) -> None:
        output.set_value(self._point_cloud)

    def CapturePointCloud(self, context: Context) -> None:
        points = (self.GetCameraPoints(context).T - self._cameras_center).T
        self._point_cloud = PointCloud(points.shape[1])
        self._point_cloud.mutable_xyzs()[:] = points
        self.PublishMeshcat(points)

class TrajectoryPredictor(CameraBackedSystem):
    """
    Performs ICP after first keying out the objects
    """
    def __init__(
            self,
            cameras: List[RgbdSensor],
            camera_transforms: List[RigidTransform],
            pred_thresh: int,
            thrown_model_name: str,
            plant: MultibodyPlant,
            meshcat: Optional[Meshcat] = None
        ):
        super().__init__(
            cameras=cameras,
            camera_transforms=camera_transforms,
            pred_thresh=pred_thresh,
            thrown_model_name=thrown_model_name,
            plant=plant,
            meshcat=meshcat
        )

        # Saved previous poses
        self._poses_state_index = self.DeclareAbstractState(AbstractValue.Make([0.0, 0.0, 0.0]))

        self._obj_point_cloud_input = self.DeclareAbstractInputPort(
            "obj_point_cloud",
            AbstractValue.Make(PointCloud())
        )

        # Update Event
        self.DeclarePeriodicPublishEvent(0.25, 0.1, self.PredictTrajectory)

        # Michael commented out `lambda c, o: None` and added `self.CreateOutput`
        port = self.DeclareAbstractOutputPort(
            "object_trajectory",
            lambda: AbstractValue.Make((Trajectory())),
            # lambda c, o: None,
            self.CreateOutput,
        )

    @property
    def point_cloud_input_port(self) -> OutputPort:
        return self._obj_point_cloud_input

    def PredictTrajectory(self, context: Context):
        self._obj_point_cloud_input.Eval(context)
        points = self.GetCameraPoints(context)
        if self._meshcat is not None:
            self.PublishMeshcat(points, self._meshcat)

    @staticmethod
    def PublishMeshcat(points: npt.NDArray[np.float32], meshcat: Meshcat):
        cloud = PointCloud(points.shape[1])
        if points.shape[1] > 0:
            cloud.mutable_xyzs()[:] = points
        meshcat.SetObject("TrajectoryPredictorPointCloud", cloud, point_size=0.01, rgba=Rgba(1, 0.5, 0.5))

    # Michael added this function to test connecting the two leafsystems
    def CreateOutput(self, context, output):
        t = 0
        test_obj_traj = PiecewisePolynomial.FirstOrderHold(
                    [t, t + 1],  # Time knots
                    np.array([[-1, 0.65], [-1, 0], [0.75, 0.75], [0, 0], [0, 0], [0, 0], [1, 1]])
                    )