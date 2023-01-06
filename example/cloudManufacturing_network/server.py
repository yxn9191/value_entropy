import os
import mesa

from example.cloudManufacturing_network.rl.utils.saving_and_loading import load_torch_model_weights
from example.cloudManufacturing_network.orderAgent import OrderAgent
from example.cloudManufacturing_network.serviceAgent import ServiceAgent
from rl.doTrain import process_args, build_Trainer
from env import CloudManufacturing_network

import rl.policy_model

current_path = os.path.split(os.path.realpath(__file__))[0]

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


def network_portrayal(G):
    def node_color(agents):
        # if get_order_agent(agents):
        #     # 当该节点上有订单时，颜色为黄色
        #     return "Yellow"
        # else:
        #     # 否则，根据企业类型决定节点颜色
        return {"A": "#FF0000", "B": "#008000", "C": "#000080"}.get(
            get_service_agent(agents).service_type, "#808080"
        )

    def edge_color(agent1, agent2):
        # if State.RESISTANT in (agent1.state, agent2.state):
        #     return "#000000"
        return "#e8e8e8"

    def edge_width(agent1, agent2):
        # if State.RESISTANT in (agent1.state, agent2.state):
        #     return 3
        return 2

    # 分别返回source节点的serviceAgent，返回target节点的serviceAgent
    def get_agents(source, target):
        return get_service_agent(G.nodes[source]["agent"]), get_service_agent(G.nodes[target]["agent"])

    # 从agents数组中分离出order的数组,如果没有order，返回空数组
    def get_order_agent(agents):
        orders = []
        for i in agents:
            if type(i) is OrderAgent:
                orders.append(i)
        return orders

    # 从agents数组中分离出service agent（一个节点对应一个）
    def get_service_agent(agents):
        for i in agents:
            if type(i) is ServiceAgent:
                return i

    def get_size(agents):
        energy = get_service_agent(agents).energy
        if energy > 0:
            return energy / 30 if energy < 360 else 12
        else:
            print("出现了energy<0的显示问题")
            return 5

    portrayal = dict()
    portrayal["nodes"] = [
        {
            "size": get_size(agents),
            "color": node_color(agents),
            "tooltip": f"id: {get_service_agent(agents).unique_id}<br>state: {get_service_agent(agents).state}"
                       f"<br>intelligence_level:{get_service_agent(agents).intelligence_level}"
                       f"<br>service_type:{get_service_agent(agents).service_type}",
        }
        for (_, agents) in G.nodes.data("agent")
    ]

    portrayal["edges"] = [
        {
            "source": source,
            "target": target,
            "color": edge_color(*get_agents(source, target)),
            "width": edge_width(*get_agents(source, target)),
        }
        for (source, target) in G.edges
    ]

    return portrayal


network = mesa.visualization.NetworkModule(network_portrayal, 500, 800)
chart = mesa.visualization.ChartModule(
    [
        {"Label": "num_nodes", "Color": "#FF0000"},
        {"Label": "num_orders", "Color": "#008000"}
    ]
)

model_params = {
    "avg_node_degree": mesa.visualization.Slider(
        "Avg Node Degree", value=3, min_value=2, max_value=8, step=1, description="Avg Node Degree"
    ),
    "num_nodes": mesa.visualization.Slider(
        "Number of agents", description="Initial Service Numbers", value=20, min_value=0, max_value=500, step=5
    ),
    "ratio_low": mesa.visualization.Slider(
        "The ratio of low_intelligence", description="Initial ratio_low", value=0.0, min_value=0, max_value=1, step=0.1
    ),
    "ratio_medium": mesa.visualization.Slider(
        "The ratio of medium_intelligence", description="Initial ratio_medium", value=0.0, min_value=0, max_value=1,
        step=0.1
    ),
    "is_training": False,
    "trainer": trainer,
    "reset_random": False,
    "episode_length": mesa.visualization.Slider(
        "Length of one episode", description="Length of one episode", value=200, min_value=100, max_value=300,
        step=10
    )
}

server = mesa.visualization.ModularServer(
    CloudManufacturing_network,
    [network, chart],
    "CloudManufacturing_network",
    model_params,
)
server.port = 8523
# model = CloudManufacturing_network(trainer=trainer, is_training=False)
# model.run_model()
