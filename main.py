from pydrake.all import DiagramBuilder, Diagram
from pydrake.geometry import StartMeshcat
from pydrake.systems.analysis import Simulator
from manipulation.station import MakeHardwareStation, load_scenario
from manipulation.scenarios import AddRgbdSensors
from utils import diagram_update_meshcat, station_visualize_camera, diagram_visualize_connections

import matplotlib.pyplot as plt
import os

from scenario import scenario_data

# Settings
scenario_path = "scenario.yaml"
close_button_str = "close"

# Setting up meshcat
meshcat = StartMeshcat()
meshcat.AddButton(close_button_str)

# Setting up the main diagram
builder = DiagramBuilder()

# scenario = load_scenario(filename=scenario_path)
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


