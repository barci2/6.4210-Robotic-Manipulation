import os

from pydrake.all import (
    DiagramBuilder,
    StartMeshcat,
    Simulator,
    ModelInstanceIndex
)
# from pydrake.systems.analysis import Simulator
from manipulation.meshcat_utils import WsgButton
from manipulation.station import MakeHardwareStation, load_scenario
from manipulation.scenarios import AddRgbdSensors, AddShape
from manipulation.meshcat_utils import AddMeshcatTriad
import matplotlib.pyplot as plt
import numpy as np
import os
from utils import (
    visualize_camera_plt,
    diagram_visualize_connections,
    throw_object,
)
from perception import add_cameras

##### Settings #####
seed = 135
close_button_str = "Close"
scenario_file = "data/scenario.yaml"
thrown_obj_prefix = "obj"
this_drake_module_name = "cwd"
simulator_runtime = 5

np.random.seed(seed)

##### Meshcat Setup #####
meshcat = StartMeshcat()
meshcat.AddButton(close_button_str)

##### Diagram Setup #####
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

### Camera Setup
plant = station.GetSubsystemByName("plant")
add_cameras(builder, station, plant, 8, 4)

diagram = builder.Build()
diagram.set_name("object_catching_system")

diagram_visualize_connections(diagram, "diagram.svg")

##### Simulation Setup #####
simulator = Simulator(diagram)
# simulator.set_target_realtime_rate(1.0)
simulator_context = simulator.get_mutable_context()
station_context = station.GetMyMutableContextFromRoot(simulator_context)
plant_context = plant.GetMyMutableContextFromRoot(simulator_context)

station.GetInputPort("iiwa.position").FixValue(station_context, np.zeros(7))
station.GetInputPort("wsg.position").FixValue(station_context, [1])

##### Extracting the object to throw #####
(obj_name,) = [ # There should only be one uncommented object
    model_name
    for model_idx in range(plant.num_model_instances())
    for model_name in [plant.GetModelInstanceName(ModelInstanceIndex(model_idx))]
    if model_name.startswith('obj')
]

obj = plant.GetModelInstanceByName(obj_name)
joint_idx = plant.GetJointIndices(obj)[0]  # JointIndex object
joint = plant.get_joint(joint_idx)  # Joint object
joint.Lock(plant_context)

##### Throwing the object #####
throw_object(plant, plant_context, obj_name)

# scene_graph = station.GetSubsystemByName("scene_graph")
# scene_graph_context = scene_graph.GetMyMutableContextFromRoot(simulator_context)
# query_object = scene_graph.get_query_output_port().Eval(scene_graph_context)
# print(query_object.ComputeSignedDistanceToPoint([-2, 0, 0]))

##### Running Simulation & Meshcat #####
simulator.AdvanceTo(0.2)
# for i in range(8 * 4):
#     visualize_camera_plt(diagram, f"camera{i}", simulator_context, False)
# plt.show()

meshcat.StartRecording()
simulator.AdvanceTo(simulator_runtime)
meshcat.PublishRecording()

while not meshcat.GetButtonClicks(close_button_str):
    pass