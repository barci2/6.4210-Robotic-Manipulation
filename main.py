from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    BsplineTrajectory,
    DiagramBuilder,
    Diagram,
    KinematicTrajectoryOptimization,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    MinimumDistanceLowerBoundConstraint,
    Parser,
    PositionConstraint,
    Rgba,
    RigidTransform,
    Role,
    Solve,
    JointSliders,
    StartMeshcat,
)

from pydrake.geometry import StartMeshcat
from pydrake.systems.analysis import Simulator
from manipulation.station import MakeHardwareStation, load_scenario
from manipulation.scenarios import AddRgbdSensors, AddShape
from manipulation.utils import ConfigureParser
from manipulation.meshcat_utils import AddMeshcatTriad
from utils import diagram_update_meshcat, station_visualize_camera, diagram_visualize_connections

import matplotlib.pyplot as plt
import numpy as np
import time

from scenario import *

NUM_THROWS = 5

rng = np.random.default_rng(135)  # Seeded randomness


# ------------------------------- MESHCAT SETUP -------------------------------
close_button_str = "Close"
meshcat = StartMeshcat()
meshcat.AddButton(close_button_str)

# -------------------------- HARDWARESTATION SETUP ----------------------------
# Setting up the main diagram
builder = DiagramBuilder()

# Load scenario contaning iiwa and wsg
scenario = load_scenario(data=scenario_data)

# Load objects to be thrown randomly
obj_directives = """
directives:
"""
for i in range(1):
    obj_idx = rng.integers(0, len(OBJECTS))

    print(obj_idx)
    print(OBJECTS[obj_idx])

    # Custom SDF Objects
    if OBJECTS[obj_idx][0] == "tennis_ball":
        obj_directives += f"""
- add_model:
    name: test
    file: file://{tennis_ball_file}
    default_free_body_pose:
        Tennis_ball:
            translation: [1.5, 0, 0.75]
            rotation: !Rpy {{ deg: [0, 0, 0] }}
        """
    # YCB Objects
    else:  
        obj_directives += f"""
- add_model:
    name: {OBJECTS[obj_idx][1]}
    file: package://drake/manipulation/models/ycb/sdf/{OBJECTS[obj_idx][0]}
    default_free_body_pose:
        {OBJECTS[obj_idx][1]}:
            translation: [1.5, 0, 0.75]
            rotation: !Rpy {{ deg: [0, 0, 0] }}
        """
scenario = add_directives(scenario, data=obj_directives)

# Create HardwareStation
station = builder.AddSystem(MakeHardwareStation(scenario, meshcat=meshcat))
station_context = station.CreateDefaultContext()

plant = station.GetSubsystemByName("plant")

# Create parser for MBP to accept new objects that we want to throw at iiwa
parser = Parser(plant)
ConfigureParser(parser)

controller_plant = station.GetSubsystemByName(
    "iiwa.controller"
).get_multibody_plant_for_control()

visualizer = MeshcatVisualizer.AddToBuilder(
    builder, station.GetOutputPort("query_object"), meshcat
)

# TEMPORARY TELEOP CONTROLS
teleop = builder.AddSystem(
    JointSliders(
        meshcat,
        controller_plant,
    )
)
builder.Connect(
    teleop.get_output_port(), station.GetInputPort("iiwa.position")
)
from manipulation.meshcat_utils import WsgButton
wsg_teleop = builder.AddSystem(WsgButton(meshcat))
builder.Connect(
    wsg_teleop.get_output_port(0), station.GetInputPort("wsg.position")
)


diagram = builder.Build()
diagram.set_name("object_catching_system")

# -------------------------------- SIMULATION ---------------------------------
diagram_update_meshcat(diagram)
diagram_visualize_connections(diagram, "diagram.svg")
station_visualize_camera(station, "camera0")

# Setting up the simulation
simulator = Simulator(diagram)
simulator.set_target_realtime_rate(1.0)
simulator_context = simulator.get_mutable_context()

plant_context = plant.GetMyMutableContextFromRoot(simulator_context)
q0 = plant.GetPositions(
    plant_context, plant.GetModelInstanceByName("iiwa")
)
station.GetInputPort("iiwa.position").FixValue(station_context, q0)
station.GetInputPort("wsg.position").FixValue(station_context, [0])

meshcat.StartRecording()
simulator.AdvanceTo(10.0)

while not meshcat.GetButtonClicks(close_button_str):

    q_cmd = np.ones(7)
    station.GetInputPort("iiwa.position").FixValue(station_context, q_cmd)

    station.GetInputPort("wsg.position").FixValue(station_context, [1])

    # q_current = station.GetOutputPort("iiwa.position_measured").Eval(station_context)
    # print(f"Current joint angles: {q_current}")