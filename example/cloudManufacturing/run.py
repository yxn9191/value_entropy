import mesa
import mesa_geo
import profile
from mesa_geo.visualization.MapModule import MapModule

from mesa_geo.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import UserSettableParameter
from mesa.visualization.modules import ChartModule, TextElement
from example.cloudManufacturing.env import CloudManufacturing
from example.cloudManufacturing.orderAgent import OrderAgent
from example.cloudManufacturing.serviceAgent import ServiceAgent
from base.region import Region

class StepText(TextElement):
    """
    Display a text count of how many steps have been taken
    """

    def __init__(self):
        pass

    def render(self, model):
        return "Steps: " + str(model.schedule.steps)

def agent_portrayal(agent):
    if agent is None:
        return

    portrayal = {}

    if type(agent) is OrderAgent:
        if agent.occupied == 0:
            portrayal["Shape"] = "Order.png"

        elif agent.occupied == 1:
            portrayal["Shape"] = "Lock.png"

        portrayal["scale"] = 0.9
        portrayal["Layer"] = 1
        portrayal["text"] = agent.order_type

    elif type(agent) is ServiceAgent:
        # portrayal["text"] = agent.service_type
        # portrayal["scale"] = 0.9
        # portrayal["Layer"] = 2
        # portrayal["text_color"] = "Red"
        if agent.service_type == "A":
            portrayal["color"] = "Green"
            #portrayal["Shape"] = "ServiceA.png"
        elif agent.service_type == "B":
            portrayal["color"] = "Blue"
            #portrayal["Shape"] = "ServiceB.png"
        elif agent.service_type == "C":
            portrayal["color"] = "Black"
            #portrayal["Shape"] = "ServiceC.png"

    elif type(agent) is Region:
        portrayal["color"] = "Red"


    return portrayal


model_params = {
    # "title": mesa.visualization.StaticText("Parameters:"),
    # "num_organization": mesa.visualization.Slider(
    #     "Initial Organization Numbers", 100, 0, 5
    # ),
    "num_service": UserSettableParameter(
        "slider", "Initial Service Numbers", value=5, min_value=0, max_value=500,step=5
    ),

    "num_order":  UserSettableParameter(
        "slider", "Initial Order Numbers",value=10, min_value=0, max_value=1000,step=5),
    "ratio_low":  UserSettableParameter(
        "slider", "Initial ratio_low", value=0, min_value=0, max_value=1, step=0.1
    ),
    "ratio_medium":  UserSettableParameter(
        "slider", "Initial ratio_medium", value=1, min_value=0, max_value=1, step=0.1
    ),
    "tax_rate":   UserSettableParameter(
        "slider","Initial tax_rate", value=1, min_value=0, max_value=1, step=0.1
    ),
}

map_element = MapModule(portrayal_method=agent_portrayal, view=[52, 12], zoom=4)
step_text = StepText()
chart = mesa.visualization.ChartModule(
    [{"Label": "Social Reward", "Color": "red"},
     # {"Label": "Service Num", "Color": "green"}
     ]
)

server = ModularServer(CloudManufacturing,
                       [map_element, step_text, chart],
                       "CloudManufacturing", model_params)

profile.run(server.launch())
