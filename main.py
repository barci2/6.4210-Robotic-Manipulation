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

# Load objects to be thrown randomly. Put them reasonably off screen so they don't get in the way until we need them
obj_model_instance_names = []  # list containing names for each object so we can retrieve them from the MBP
obj_directives = """
directives:
"""
for i in range(NUM_THROWS):
    obj_idx = rng.integers(0, len(OBJECTS))

    print(obj_idx)
    print(OBJECTS[obj_idx])

    # Custom SDF Objects
    if OBJECTS[obj_idx][0] == "tennis_ball":
        obj_directives += f"""
- add_model:
    name: Tennis_ball{i}
    file: file://{tennis_ball_file}
    default_free_body_pose:
        Tennis_ball:
            translation: [0, 0, 2]
            rotation: !Rpy {{ deg: [0, 0, 0] }}
        """
        obj_model_instance_names.append(f"Tennis_ball{i}")
    # YCB Objects
    else:  
        obj_directives += f"""
- add_model:
    name: ycb{i}
    file: package://drake/manipulation/models/ycb/sdf/{OBJECTS[obj_idx][0]}
    default_free_body_pose:
        {OBJECTS[obj_idx][1]}:
            translation: [0, 0, 2]
            rotation: !Rpy {{ deg: [0, 0, 0] }}
        """
        obj_model_instance_names.append(f"ycb{i}")

scenario = add_directives(scenario, data=obj_directives)

# Create HardwareStation
station = builder.AddSystem(MakeHardwareStation(scenario, meshcat=meshcat))

scene_graph = station.GetSubsystemByName("scene_graph")

plant = station.GetSubsystemByName("plant")

controller_plant = station.GetSubsystemByName(
    "iiwa.controller"
).get_multibody_plant_for_control()

# TEMPORARY TELEOP CONTROLS (comment out the stuff in the while loop below for this to work)
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

# Build diagram
diagram = builder.Build()
diagram.set_name("object_catching_system")

# -------------------------------- SIMULATION ---------------------------------
diagram_update_meshcat(diagram)
diagram_visualize_connections(diagram, "diagram.svg")
# station_visualize_camera(station, "camera0", station_context)  # TODO: figure out why this is causing segfault

# Setting up the simulation
simulator = Simulator(diagram)
simulator.set_target_realtime_rate(1.0)
simulator_context = simulator.get_mutable_context()
station_context = station.GetMyMutableContextFromRoot(simulator_context)
plant_context = plant.GetMyMutableContextFromRoot(simulator_context)

# Freeze all objects for now (we'll unfreeze them when we're ready to throw them)
for obj in obj_model_instance_names:
    joint_idx = plant.GetJointIndices(plant.GetModelInstanceByName(obj))[0]  # JointIndex object
    joint = plant.get_joint(joint_idx)  # Joint object
    joint.Lock(plant_context)

meshcat.StartRecording()
simulator.AdvanceTo(0.1)
meshcat.Flush()  # Wait for the large object meshes to get to meshcat.

while not meshcat.GetButtonClicks(close_button_str):
    simulator.AdvanceTo(simulator.get_context().get_time() + 1.0)

    q_cmd = np.ones(7)
    station.GetInputPort("iiwa.position").FixValue(station_context, q_cmd)

    station.GetInputPort("wsg.position").FixValue(station_context, [1])

    # q_current = station.GetOutputPort("iiwa.position_measured").Eval(station_context)
    # print(f"Current joint angles: {q_current}")

meshcat.DeleteButton(close_button_str)

# Publish recording so we can play it back at varying speeds for debugging
meshcat.PublishRecording()