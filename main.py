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
    StartMeshcat,
)

from pydrake.geometry import StartMeshcat
from pydrake.systems.analysis import Simulator
from manipulation.station import MakeHardwareStation, load_scenario
from manipulation.scenarios import AddRgbdSensors
from utils import diagram_update_meshcat, station_visualize_camera, diagram_visualize_connections

import matplotlib.pyplot as plt
import numpy as np

from scenario import scenario_data

# Settings
close_button_str = "close"

# Setting up meshcat
meshcat = StartMeshcat()
meshcat.AddButton(close_button_str)

# Setting up the main diagram
builder = DiagramBuilder()

scenario = load_scenario(data=scenario_data)
station = builder.AddSystem(MakeHardwareStation(scenario, meshcat=meshcat))

diagram = builder.Build()
diagram.set_name("object_catching_system")

# Visualization
diagram_update_meshcat(diagram)
diagram_visualize_connections(diagram, "diagram.svg")
station_visualize_camera(station, "camera0")

# Setting up the simulation
simulator = Simulator(diagram)
context = diagram.CreateDefaultContext()

while not meshcat.GetButtonClicks(close_button_str):
    pass


