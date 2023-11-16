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
from manipulation.scenarios import AddRgbdSensors, AddPlanarIiwa, AddShape, AddWsg
from manipulation.utils import ConfigureParser
from manipulation.meshcat_utils import AddMeshcatTriad
from utils import diagram_update_meshcat, station_visualize_camera, diagram_visualize_connections

import matplotlib.pyplot as plt
import numpy as np

from scenario import scenario_data


# --------------------------- MULTIBODYPLANT SETUP ----------------------------
builder = DiagramBuilder()
plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)

iiwa = AddPlanarIiwa(plant)
wsg = AddWsg(plant, iiwa, roll=0.0, welded=True, sphere=True)

parser = Parser(plant)
ConfigureParser(parser)
parser.AddModelsFromString(scenario_data, ".dmd.yaml")  # sdf format string

# iiwa = plant.GetModelInstanceByName("iiwa")
tennis_ball = plant.GetModelInstanceByName("Tennis_ball")

plant.Finalize()

# ------------------------------- MESHCAT SETUP -------------------------------
close_button_str = "Close"
meshcat = StartMeshcat()
meshcat.AddButton(close_button_str)

visualizer = MeshcatVisualizer.AddToBuilder(
    builder,
    scene_graph,
    meshcat,
    MeshcatVisualizerParams(role=Role.kIllustration),
)
collision_visualizer = MeshcatVisualizer.AddToBuilder(
    builder,
    scene_graph,
    meshcat,
    MeshcatVisualizerParams(
        prefix="collision", role=Role.kProximity, visible_by_default=False
    ),
)

diagram = builder.Build()
diagram.set_name("object_catching_system")
context = diagram.CreateDefaultContext()
plant_context = plant.GetMyContextFromRoot(context)

num_q = plant.num_positions()
print(num_q)

# plant.SetVelocities(plant_context, tennis_ball, np.ones(3))
# plant.SetPositions(plant_context, tennis_ball, np.ones(7))

diagram.ForcedPublish(context)  # Publish results to Diagram and MeshCat

# AddMeshcatTriad(meshcat, "test", X_PT=RigidTransform(), opacity=0.5)


# # Visualization
diagram_update_meshcat(diagram)
diagram_visualize_connections(diagram, "diagram.svg")
# station_visualize_camera(station, "camera0")

# # Setting up the simulation
simulator = Simulator(diagram)
meshcat.StartRecording()
simulator.AdvanceTo(5.0)
visualizer.PublishRecording()

while not meshcat.GetButtonClicks(close_button_str):
    pass


