from pydrake.all import (
    DiagramBuilder,
    StartMeshcat,
    Simulator,
    ModelInstanceIndex,
    PiecewisePolynomial,
    PortSwitch,
    TrajectorySource,
    RigidTransform
)
from manipulation.station import MakeHardwareStation, load_scenario
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
from motion_tests import motion_test  # TEMPORARY

##### Settings #####
seed = 135
close_button_str = "Close"
scenario_file = "data/scenario.yaml"
thrown_obj_prefix = "obj"
this_drake_module_name = "cwd"
point_cloud_cameras_center = [0, 0, 100]
simulator_runtime = 1

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
    camera_width=800,
    camera_height=600,
    horizontal_num=8,
    vertical_num=5,
    camera_distance=5,
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
    camera_distance=5,
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
    pred_samples_thresh=4,
    thrown_model_name=obj_name,
    ransac_iters=20,
    ransac_thresh=0.01,
    ransac_window=30,
    plant=plant,
    meshcat=meshcat
))
traj_pred_system.ConnectCameras(builder, icp_cameras)
builder.Connect(obj_point_cloud_system.GetOutputPort("point_cloud"), traj_pred_system.point_cloud_input_port)

### Grasp Selector
grasp_selector = builder.AddSystem(GraspSelector(plant, scene_graph, meshcat))
builder.Connect(traj_pred_system.GetOutputPort("object_trajectory"), grasp_selector.GetInputPort("object_trajectory"))
builder.Connect(obj_point_cloud_system.GetOutputPort("point_cloud"), grasp_selector.GetInputPort("object_pc"))

### Motion Planner
motion_planner = builder.AddSystem(MotionPlanner(plant, meshcat))
builder.Connect(grasp_selector.GetOutputPort("grasp_selection"), motion_planner.GetInputPort("grasp_selection"))
builder.Connect(station.GetOutputPort("body_poses"), motion_planner.GetInputPort("iiwa_current_pose"))
builder.Connect(station.GetOutputPort("iiwa.velocity_estimated"), motion_planner.GetInputPort("iiwa_current_vel"))
builder.Connect(station.GetOutputPort("iiwa.position_measured"), motion_planner.GetInputPort("iiwa_current_pos"))
builder.Connect(traj_pred_system.GetOutputPort("object_trajectory"), motion_planner.GetInputPort("object_trajectory"))
builder.Connect(motion_planner.GetOutputPort("iiwa_position_command"), station.GetInputPort("iiwa.position"))


# # Port Switch to switch from stationary trajectory to catching trajectory
# switch = builder.AddSystem(PortSwitch(7))
# builder.Connect(
#     diff_ik.get_output_port(), switch.DeclareInputPort("diff_ik")  # For normal use case
# )
# builder.Connect(
#     planner.GetOutputPort("iiwa_position_command"),  # for Planner's GoHome state
#     switch.DeclareInputPort("position"),
# )
# builder.Connect(switch.get_output_port(), station.GetInputPort("iiwa.position"))
# builder.Connect(
#     planner.GetOutputPort("control_mode"),
#     switch.get_port_selector_input_port(),
# )


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

# print(station.GetOutputPort("body_poses").Eval(station_context)[plant.GetBodyByName("body", plant.GetModelInstanceByName("wsg")).index()])


### Opening the gripper
# station.GetInputPort("iiwa.position").FixValue(station_context, np.zeros(7)) # TESTING
station.GetInputPort("wsg.position").FixValue(station_context, [1]) # TESTING


### Capturing the point cloud for the robot
obj = plant.GetModelInstanceByName(obj_name)
body_idx = plant.GetBodyIndices(obj)[0]
body = plant.get_body(body_idx)
plant.SetFreeBodyPose(plant_context, body, RigidTransform(point_cloud_cameras_center))
obj_point_cloud_system.CapturePointCloud(obj_point_cloud_system.GetMyMutableContextFromRoot(simulator_context))



throw_object(plant, plant_context, obj_name)

# plt.imshow(icp_cameras[17].depth_image_32F_output_port().Eval(icp_cameras[17].GetMyContextFromRoot(simulator_context)).data[::-1])
# plt.show()

# scene_graph_context = scene_graph.GetMyMutableContextFromRoot(simulator_context) # TESTING
# query_object = scene_graph.get_query_output_port().Eval(scene_graph_context) # TESTING
# print(query_object.ComputeSignedDistanceToPoint([-2, 0, 0])) # TESTING

####################################
### Running Simulation & Meshcat ###
####################################
simulator.set_target_realtime_rate(0.1)
simulator.set_publish_every_time_step(True)
plt.show()

meshcat.StartRecording()
simulator.AdvanceTo(simulator_runtime)
meshcat.PublishRecording()

while not meshcat.GetButtonClicks(close_button_str):
    pass
