from pydrake.all import (
    DiagramBuilder,
    StartMeshcat,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    Simulator,
    ModelInstanceIndex,
    PiecewisePolynomial,
    TrajectorySource
)
# from pydrake.systems.analysis import Simulator
from manipulation.meshcat_utils import WsgButton
from manipulation.station import MakeHardwareStation, load_scenario
from manipulation.scenarios import AddRgbdSensors, AddShape
from manipulation.meshcat_utils import AddMeshcatTriad, PublishPositionTrajectory
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from utils import (
    visualize_camera_plt,
    diagram_visualize_connections,
    throw_object,
)
from perception import TrajectoryPredictor, add_cameras

from grasping_selection import GraspSelector
from motion_tests import motion_test  # TEMPORARY

##### Settings #####
seed = 135
close_button_str = "Close"
scenario_file = "data/scenario.yaml"
thrown_obj_prefix = "obj"
this_drake_module_name = "cwd"
simulator_runtime = 3

np.random.seed(seed)

##### Meshcat Setup #####
meshcat = StartMeshcat()
meshcat.AddButton(close_button_str)

##### Diagram Setup #####
builder = DiagramBuilder()
scenario = load_scenario(filename=scenario_file)

##### Hardware station setup #####
station = builder.AddSystem(MakeHardwareStation(
    scenario=scenario,
    meshcat=meshcat,

    # This is to be able to load our own models from a local path
    # we can refer to this using the "package://" URI directive
    parser_preload_callback=lambda parser: parser.package_map().Add(this_drake_module_name, os.getcwd())
))
scene_graph = station.GetSubsystemByName("scene_graph")

### Object to throw
plant = station.GetSubsystemByName("plant")
(obj_name,) = [ # There should only be one uncommented object
    model_name
    for model_idx in range(plant.num_model_instances())
    for model_name in [plant.GetModelInstanceName(ModelInstanceIndex(model_idx))]
    if model_name.startswith('obj')
]

### Camera Setup
cameras, camera_transforms = add_cameras(builder, station, plant, 800, 600, 8, 4, 5)

### Perception Setup
perception_system = builder.AddSystem(TrajectoryPredictor(cameras, camera_transforms, 0, obj_name, plant, meshcat))
for i, camera in enumerate(cameras):
    depth_input, label_input = perception_system.camera_input_ports(i)
    builder.Connect(camera.depth_image_32F_output_port(), depth_input)
    builder.Connect(camera.label_image_output_port(), label_input)

### Grasp Selector
# grasp_selector = builder.AddSystem(GraspSelector(plant, scene_graph, diagram, context, meshcat))
# builder.Connect(perception_system.GetOutputPort("object_trajectory"), grasp_selector.GetInputPort("object_trajectory"))

### Motion Planning
# linear path for testing
t = 0
test_obj_traj = PiecewisePolynomial.FirstOrderHold(
            [t, t + 1],  # Time knots
            np.array([[-1, 0.55], [-1, 0], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
            )

q_traj = motion_test(plant, meshcat, test_obj_traj, 1)

# Connect output of motion system to input of iiwa's controller
q_traj_system = builder.AddSystem(TrajectorySource(q_traj))
builder.Connect(
    q_traj_system.get_output_port(), station.GetInputPort("iiwa.position")
)

### Finalizing diagram setup
diagram = builder.Build()
context = diagram.CreateDefaultContext()
diagram.set_name("object_catching_system")
diagram_visualize_connections(diagram, "diagram.svg")

##### Simulation Setup #####
simulator = Simulator(diagram)
simulator_context = simulator.get_mutable_context()
station_context = station.GetMyMutableContextFromRoot(simulator_context)
plant_context = plant.GetMyMutableContextFromRoot(simulator_context)

# station.GetInputPort("iiwa.position").FixValue(station_context, np.zeros(7))
station.GetInputPort("wsg.position").FixValue(station_context, [1])

obj = plant.GetModelInstanceByName(obj_name)
joint_idx = plant.GetJointIndices(obj)[0]  # JointIndex object
joint = plant.get_joint(joint_idx)  # Joint object
joint.Lock(plant_context)

throw_object(plant, plant_context, obj_name)

# scene_graph_context = scene_graph.GetMyMutableContextFromRoot(simulator_context)
# query_object = scene_graph.get_query_output_port().Eval(scene_graph_context)
# print(query_object.ComputeSignedDistanceToPoint([-2, 0, 0]))

##### Running Simulation & Meshcat #####
# simulator.AdvanceTo(0.2)

simulator.set_target_realtime_rate(1.0)
simulator.set_publish_every_time_step(True)
plt.show()

meshcat.StartRecording()
simulator.AdvanceTo(simulator_runtime)
meshcat.PublishRecording()

while not meshcat.GetButtonClicks(close_button_str):
    pass
