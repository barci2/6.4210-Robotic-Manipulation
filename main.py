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
from manipulation.scenarios import AddRgbdSensors, AddPlanarIiwa, AddShape, AddWsg
from manipulation.utils import ConfigureParser
from manipulation.meshcat_utils import AddMeshcatTriad
from utils import diagram_update_meshcat, station_visualize_camera, diagram_visualize_connections

import matplotlib.pyplot as plt
import numpy as np

from scenario import scenario_data


# ------------------------------- MESHCAT SETUP -------------------------------
close_button_str = "Close"
meshcat = StartMeshcat()
meshcat.AddButton(close_button_str)

# -------------------------- HARDWARESTATION SETUP ----------------------------
# Setting up the main diagram
builder = DiagramBuilder()

scenario = load_scenario(data=scenario_data)
station = builder.AddSystem(MakeHardwareStation(scenario, meshcat=meshcat))
station_context = station.CreateDefaultContext()

plant = station.GetSubsystemByName("plant")
controller_plant = station.GetSubsystemByName(
    "iiwa.controller"
).get_multibody_plant_for_control()

visualizer = MeshcatVisualizer.AddToBuilder(
    builder, station.GetOutputPort("query_object"), meshcat
)

# TEMPORARY
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

    # station.GetInputPort("wsg.position").FixValue(station_context, [0])

    # q_current = station.GetOutputPort("iiwa.position_measured").Eval(station_context)
    # print(f"Current joint angles: {q_current}")