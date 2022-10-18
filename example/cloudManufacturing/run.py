import mesa
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import CanvasGrid

from example.cloudManufacturing.env import CloudManufacturing
from example.cloudManufacturing.orderAgent import OrderAgent
from example.cloudManufacturing.serviceAgent import ServiceAgent


def agent_portrayal(agent):
    if agent is None:
        return

    portrayal = {}

    if type(agent) is OrderAgent:
        portrayal["Shape"] = "Order.png"
        portrayal["scale"] = 0.9
        portrayal["Layer"] = 2
        portrayal["text"] = agent.order_type

    elif type(agent) is ServiceAgent:
        portrayal["text"] = agent.service_type
        portrayal["scale"] = 0.9
        portrayal["Layer"] = 1
        portrayal["text_color"] = "Red"
        if agent.service_type == "A":
            portrayal["Shape"] = "ServiceA.png"
        elif agent.service_type == "B":
            portrayal["Shape"] = "ServiceB.png"
        elif agent.service_type == "C":
            portrayal["Shape"] = "ServiceC.png"

    return portrayal


model_params = {
    "title": mesa.visualization.StaticText("Parameters:"),
    # "num_organization": mesa.visualization.Slider(
    #     "Initial Organization Numbers", 100, 0, 5
    # ),
    "num_service": mesa.visualization.Slider(
        "Initial Service Numbers",0, 10, 200
    ),

    "num_order": mesa.visualization.Slider("Initial Order Numbers", 0, 10, 100),
    "ratio_low": mesa.visualization.Slider(
        "Initial ratio_low", 0, 0.1, 1
    ),
    "ratio_medium": mesa.visualization.Slider(
        "Initial ratio_medium", 0,0.1, 1
    ),
}

grid = CanvasGrid(agent_portrayal, 20, 20, 700, 700)
# chart = mesa.visualization.ChartModule(
#     [{"Label": "Social Reward", "Color": "Black"},
#      {"Label": "Service Num", "Color": "#666666"}
#      ], data_collector_name="collector"
# )

server = ModularServer(CloudManufacturing,
                       [grid],
                       "CloudManufacturing", model_params)

server.launch()
