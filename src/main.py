from pydrake.all import (
    DiagramBuilder,
    StartMeshcat,
    Simulator,
    ModelInstanceIndex,
    InverseDynamicsController,
    RigidTransform,
    MultibodyPlant,
    RotationMatrix,
    ConstantVectorSource,
    AbstractValue
)
# from manipulation.station import MakeHardwareStation, load_scenario
from station import MakeHardwareStation, load_scenario
from manipulation.scenarios import AddIiwa

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.use("tkagg")
import os
import time
from utils import (
    diagram_visualize_connections,
    throw_object,
    ObjectTrajectory
)
from perception import PointCloudGenerator, TrajectoryPredictor, add_cameras

from grasping_selection import GraspSelector
from motion_planner import MotionPlanner

##### Settings #####
seed = 135#135
close_button_str = "Close"
scenario_file = "data/scenario.yaml"
thrown_obj_prefix = "obj"
this_drake_module_name = "cwd"
point_cloud_cameras_center = [0, 0, 100]
simulator_runtime = 1.0

np.random.seed(seed)

#####################
### Meshcat Setup ###
#####################
meshcat = StartMeshcat()
meshcat.AddButton(close_button_str)

#####################
### Diagram Setup ###
#####################
builder = DiagramBuilder()
scenario = load_scenario(filename=scenario_file)


### Hardware station setup
station = builder.AddSystem(MakeHardwareStation(
    scenario=scenario,
    meshcat=meshcat,

    # This is to be able to load our own models from a local path
    # we can refer to this using the "package://" URI directive
    parser_preload_callback=lambda parser: parser.package_map().Add(this_drake_module_name, os.getcwd())
))
scene_graph = station.GetSubsystemByName("scene_graph")

### Extracting the object to throw
plant = station.GetSubsystemByName("plant")
(obj_name,) = [ # There should only be one uncommented object
    model_name
    for model_idx in range(plant.num_model_instances())
    for model_name in [plant.GetModelInstanceName(ModelInstanceIndex(model_idx))]
    if model_name.startswith('obj')
]


### Camera Setup
icp_cameras, icp_camera_transforms = add_cameras(
    builder=builder,
    station=station,
    plant=plant,
    camera_width=800//2,
    camera_height=600//2,
    horizontal_num=4,
    vertical_num=5,
    camera_distance=7,
    cameras_center=[0, 0, 0]
)

point_cloud_cameras, point_cloud_camera_transforms = add_cameras(
    builder=builder,
    station=station,
    plant=plant,
    camera_width=800,
    camera_height=600,
    horizontal_num=8,
    vertical_num=4,
    camera_distance=1,
    cameras_center=point_cloud_cameras_center
)

### Point Cloud Capturing Setup
obj_point_cloud_system = builder.AddSystem(PointCloudGenerator(
    cameras=point_cloud_cameras,
    camera_transforms=point_cloud_camera_transforms,
    cameras_center=point_cloud_cameras_center,
    pred_thresh=5,
    thrown_model_name=obj_name,
    plant=plant
))
obj_point_cloud_system.ConnectCameras(builder, point_cloud_cameras)

### Trajectory Prediction Setup
traj_pred_system = builder.AddSystem(TrajectoryPredictor(
    cameras=icp_cameras,
    camera_transforms=icp_camera_transforms,
    pred_thresh=5,
    pred_samples_thresh=6,  # how many views of object are needed before outputting predicted traj
    thrown_model_name=obj_name,
    ransac_iters=20,
    ransac_thresh=0.01,
    ransac_rot_thresh=0.1,
    ransac_window=30,
    plant=plant,
    meshcat=meshcat
))
traj_pred_system.ConnectCameras(builder, icp_cameras)
builder.Connect(obj_point_cloud_system.GetOutputPort("point_cloud"), traj_pred_system.point_cloud_input_port)

### Grasp Selector
grasp_selector = builder.AddSystem(GraspSelector(plant, scene_graph, meshcat, obj_name))
builder.Connect(traj_pred_system.GetOutputPort("object_trajectory"), grasp_selector.GetInputPort("object_trajectory"))
builder.Connect(obj_point_cloud_system.GetOutputPort("point_cloud"), grasp_selector.GetInputPort("object_pc"))

### Motion Planner
motion_planner = builder.AddSystem(MotionPlanner(plant, meshcat))
builder.Connect(grasp_selector.GetOutputPort("grasp_selection"), motion_planner.GetInputPort("grasp_selection"))
builder.Connect(station.GetOutputPort("body_poses"), motion_planner.GetInputPort("iiwa_current_pose"))
builder.Connect(traj_pred_system.GetOutputPort("object_trajectory"), motion_planner.GetInputPort("object_trajectory"))
builder.Connect(station.GetOutputPort("iiwa_state"), motion_planner.GetInputPort("iiwa_state"))
builder.Connect(motion_planner.GetOutputPort("wsg_command"), station.GetInputPort("wsg.position"))


# Implement inverse dynamics controller for feedforward acceleration
controller_plant = MultibodyPlant(time_step=0.001)
controller_iiwa = AddIiwa(controller_plant)
# controller_iiwa = Parser(controller_plant).AddModelsFromUrl("package://drake/manipulation/models/iiwa_description/urdf/iiwa14_spheres_dense_collision.urdf")[0]  # ModelInstance object
controller_plant.Finalize()
num_iiwa_positions = controller_plant.num_positions()
# controller = builder.AddSystem(InverseDynamicsController(controller_plant, [100]*num_iiwa_positions, [1]*num_iiwa_positions, [20]*num_iiwa_positions, True))
controller = builder.AddSystem(InverseDynamicsController(controller_plant, [200]*num_iiwa_positions, [1]*num_iiwa_positions, [20]*num_iiwa_positions, True))
builder.Connect(station.GetOutputPort("iiwa_state"), controller.GetInputPort("estimated_state"))
builder.Connect(motion_planner.GetOutputPort("iiwa_command"), controller.GetInputPort("desired_state"))
builder.Connect(motion_planner.GetOutputPort("iiwa_acceleration"), controller.GetInputPort("desired_acceleration"))
builder.Connect(controller.GetOutputPort("generalized_force"), station.GetInputPort("iiwa.actuation"))



### Finalizing diagram setup
diagram = builder.Build()
context = diagram.CreateDefaultContext()
diagram.set_name("object_catching_system")
diagram_visualize_connections(diagram, "diagram.svg")


########################
### Simulation Setup ###
########################
simulator = Simulator(diagram)
simulator_context = simulator.get_mutable_context()
station_context = station.GetMyMutableContextFromRoot(simulator_context)
plant_context = plant.GetMyMutableContextFromRoot(simulator_context)


### Testing hardware
# station.GetInputPort("iiwa.desired_state").FixValue(station_context, np.zeros(14)) # TESTING
# station.GetInputPort("wsg.position").FixValue(station_context, [1]) # TESTING


### Capturing the point cloud for the robot
obj = plant.GetModelInstanceByName(obj_name)
body_idx = plant.GetBodyIndices(obj)[0]
body = plant.get_body(body_idx)
plant.SetFreeBodyPose(plant_context, body, RigidTransform(point_cloud_cameras_center))
obj_point_cloud_system.CapturePointCloud(obj_point_cloud_system.GetMyMutableContextFromRoot(simulator_context))

# For tennis ball:
# throw_object(plant, plant_context, obj_name, RotationMatrix())
# For banana:
# throw_object(plant, plant_context, obj_name, RotationMatrix.MakeZRotation(-np.pi / 4) @ RotationMatrix.MakeXRotation(-np.pi / 4) @ RotationMatrix.MakeZRotation(-np.pi / 2))
# For pill bottle:
throw_object(plant, plant_context, obj_name, RotationMatrix.MakeZRotation(-np.pi / 6) @ RotationMatrix.MakeXRotation(np.pi / 4))

# Example camera view
# plt.imshow(icp_cameras[17].depth_image_32F_output_port().Eval(icp_cameras[17].GetMyContextFromRoot(simulator_context)).data[::-1])
# plt.show()

# scene_graph_context = scene_graph.GetMyMutableContextFromRoot(simulator_context) # TESTING
# query_object = scene_graph.get_query_output_port().Eval(scene_graph_context) # TESTING
# print(query_object.ComputeSignedDistanceToPoint([-2, 0, 0])) # TESTING

####################################
### Running Simulation & Meshcat ###
####################################
simulator.set_target_realtime_rate(1)
simulator.set_publish_every_time_step(True)
plt.show()

meshcat.StartRecording()
simulator.AdvanceTo(simulator_runtime)
meshcat.PublishRecording()

while not meshcat.GetButtonClicks(close_button_str):
    pass
