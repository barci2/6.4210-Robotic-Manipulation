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
# meshcat.AddButton(close_button_str)

# -------------------------- HARDWARESTATION SETUP ----------------------------
# Setting up the main diagram
builder = DiagramBuilder()

# Load scenario contaning iiwa and wsg
scenario = load_scenario(data=scenario_data)

# Load objects to be thrown randomly. Put them reasonably off screen so they don't get in the way until we need them
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
            translation: [0, 0, 100]
            rotation: !Rpy {{ deg: [0, 0, 0] }}
        """
    # YCB Objects
    else:  
        obj_directives += f"""
- add_model:
    name: ycb{i}
    file: package://drake/manipulation/models/ycb/sdf/{OBJECTS[obj_idx][0]}
    default_free_body_pose:
        {OBJECTS[obj_idx][1]}:
            translation: [0, 0, 100]
            rotation: !Rpy {{ deg: [0, 0, 0] }}
        """
scenario = add_directives(scenario, data=obj_directives)

# Create HardwareStation
station = builder.AddSystem(MakeHardwareStation(scenario, meshcat=meshcat))
station_context = station.CreateDefaultContext()
scene_graph = station.GetSubsystemByName("scene_graph")

# Add visualizer so we can play back the simulation for debugging purposes
visualizer = MeshcatVisualizer.AddToBuilder(
    builder,
    station.GetOutputPort("query_object"),
    meshcat,
    MeshcatVisualizerParams(delete_on_initialization_event=False),
)

plant = station.GetSubsystemByName("plant")

controller_plant = station.GetSubsystemByName(
    "iiwa.controller"
).get_multibody_plant_for_control()

# joint_idx = plant.GetJointByName("Tennis_ball").index()
# print(joint_idx)

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
# station_visualize_camera(station, "camera0")  # TODO: figure out why this is causing segfault

# Setting up the simulation
simulator = Simulator(diagram)
simulator.set_target_realtime_rate(1.0)
# simulator_context = simulator.get_mutable_context()

# If we choose meshcat to record, we can see the simulation in realtime
# meshcat.StartRecording()
# If we choose visualizer to record, we can play back the simulation after
visualizer.StartRecording(False)
simulator.AdvanceTo(10.0)
# meshcat.PublishRecording()
visualizer.PublishRecording()

# while not meshcat.GetButtonClicks(close_button_str):

#     q_cmd = np.ones(7)
#     station.GetInputPort("iiwa.position").FixValue(station_context, q_cmd)

#     station.GetInputPort("wsg.position").FixValue(station_context, [1])

    # q_current = station.GetOutputPort("iiwa.position_measured").Eval(station_context)
    # print(f"Current joint angles: {q_current}")