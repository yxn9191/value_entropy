import mesa


class BaseEnvironment(mesa.Model):
    """
    """

    def __init__(self,
                 width=None,
                 height=None,
                 episode_length=None,
                 schedule=None,
                 grid=None
                 ):
        super().__init__()
        self.schedule = schedule
        self.grid = grid
        self.episode_length = episode_length  # 一次演化的时长

        self.timestep = 0  # 环境当前处于的时间点

        # 环境中每个智能体的当前奖赏值
        self.curr_optimization_metric = dict()

        # 环境中所有的企业
        self.all_agent = []
        self._agent_lookup = {str(agent.unique_id): agent for agent in self.all_agent}

        self.actions = None

    # 重置整个环境
    def reset(self):
        for agent in self.all_agent:
            agent.reset()
        obs = self.generate_observations()
        return obs

    # 生成观察值（强化学习的输入，待补充）
    def generate_observations(self):
        pass

    # 生成即时奖赏值（强化学习的输入，待补充）
    def generate_rewards(self):
        pass

    def action_parse(self, action_dict):
        self.actions = action_dict

    def step(self):
        self.timestep = self.timestep + 1
