from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.UserParam import UserSettableParameter

from pdpython_model.generic_model.agents import Gagent
from pdpython_model.generic_model.model import GenericModel

def gen_Model_Portrayal(agent):
    if agent is None:
        return

    portrayal = {}

    if type(agent) is Gagent:
        portrayal = {"Shape": "circle",
                     "scale": 1,
                     "Color": "pink",
                     "Filled": "true",
                     "Layer": 1,
                     "r": 0.5,
                     "text": "ᕕ( ՞ ᗜ ՞ )ᕗ",
                     "text_color": "black",
                     "scale": 1
                     }

    return portrayal

canvas_element = CanvasGrid(gen_Model_Portrayal, 5, 5, 500, 500)
# chart_element = ChartModule([{"Label": "Walkers", "Color": "#AA0000"},
#                              {"Label": "Closed Boxes", "Color": "#666666"}])

model_params = {"number_of_agents": UserSettableParameter('slider', 'Number of Agents', 1, 1, 10, 1),
                }

server = ModularServer(GenericModel, [canvas_element], "Generic Model", model_params)
server.port = 8521
