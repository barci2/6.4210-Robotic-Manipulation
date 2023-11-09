""" Miscellaneous Utility functions """
from typing import BinaryIO, Union
from pydrake.all import DiagramBuilder, Diagram
import pydot
import matplotlib.pyplot as plt

def diagram_update_meshcat(diagram, context = None) -> None:
    if context is None:
        context = diagram.CreateDefaultContext()
    diagram.ForcedPublish(context)

def diagram_visualize_connections(diagram: Diagram, file: Union[BinaryIO, str]) -> None:
    if type(file) is str:
        file = open(file, "bw")
    graphviz_str = diagram.GetGraphvizString()
    svg_data = pydot.graph_from_dot_data(diagram.GetGraphvizString())[0].create_svg()
    file.write(svg_data)

def station_visualize_camera(station: Diagram, camera_name: str, context = None) -> None:
    if context is None:
        context = station.CreateDefaultContext()
    image = station.GetOutputPort(f"{camera_name}.rgb_image").Eval(context).data
    print(image)
    print(dir(image))
    plt.imshow(image)
    plt.show()
