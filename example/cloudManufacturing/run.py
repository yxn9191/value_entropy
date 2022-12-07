import mesa
import profile
import os
import sys

current_path = os.path.split(os.path.realpath(__file__))[0]

sys.path.append("/home/bertrand/Desktop/group-intelligence-system")

from mesa_geo.visualization.MapModule import MapModule

from mesa_geo.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import UserSettableParameter
from mesa.visualization.modules import ChartModule, TextElement
from example.cloudManufacturing.env import CloudManufacturing
from example.cloudManufacturing.orderAgent import OrderAgent
from example.cloudManufacturing.serviceAgent import ServiceAgent
from base.region import Region
from rl.doTrain import build_Trainer, process_args
from rl.utils.saving_and_loading import load_torch_model_weights
import rl.policy_model


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
        portrayal['radius'] = "1"
        if agent.occupied == 0:
            portrayal["color"] = "Gray"

        elif agent.occupied == 1:
            portrayal['radius'] = "2"
            portrayal["color"] = "Black"

        # portrayal["scale"] = 0.9
        # portrayal["Layer"] = 1
        # portrayal["text"] = agent.order_type

    elif type(agent) is ServiceAgent:
        portrayal["text"] = agent.service_type
        portrayal["scale"] = 0.9
        portrayal["Layer"] = 2
        # portrayal["text_color"] = "Red"
        portrayal['radius'] = "1"
        if agent.service_type == "A":
            portrayal["color"] = "Green"
            # portrayal["Shape"] = "ServiceA.png"
        elif agent.service_type == "B":
            portrayal["color"] = "Blue"
            # portrayal["Shape"] = "ServiceB.png"
        elif agent.service_type == "C":
            portrayal["color"] = "Red"
            # portrayal["Shape"] = "ServiceC.png"

    elif type(agent) is Region:
        portrayal["color"] = "Black"

    return portrayal


# 获取参数
run_dir, run_config = process_args()
# 创建训练器
trainer = build_Trainer(run_config)
ckpt = run_config["general"].get(
    "ckpt_path", ""
)
ckpt = os.path.join(current_path, ckpt)
trainer.restore(str(ckpt))

starting_weights_path_agents = run_config["general"].get(
    "restore_torch_weights_agents", ""
)
starting_weights_path_agents = os.path.join(current_path, starting_weights_path_agents)
load_torch_model_weights(trainer, starting_weights_path_agents)

model_params = {
    # "title": mesa.visualization.StaticText("Parameters:"),
    # "num_organization": mesa.visualization.Slider(
    #     "Initial Organization Numbers", 100, 0, 5
    # ),
    "num_service": UserSettableParameter(
        "slider", "Initial Service Numbers", value=5, min_value=0, max_value=500, step=5
    ),

    "num_order": UserSettableParameter(
        "slider", "Initial Order Numbers", value=10, min_value=0, max_value=1000, step=5),
    "ratio_low": UserSettableParameter(
        "slider", "Initial ratio_low", value=0, min_value=0, max_value=1, step=0.1
    ),
    "ratio_medium": UserSettableParameter(
        "slider", "Initial ratio_medium", value=1, min_value=0, max_value=1, step=0.1
    ),
    "tax_rate": UserSettableParameter(
        "slider", "Initial tax_rate", value=1, min_value=0, max_value=1, step=0.1
    ),
    "trainer": trainer
}

map_element = MapModule(agent_portrayal, CloudManufacturing.MAP_COORDS, 10, 500, 500)
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
