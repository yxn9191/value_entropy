import mesa



class BaseEnvironment(mesa.Model):
    """
    """
    name = ""

    def __init__(self,
                 width=None,
                 height=None,
                 episode_length=None,
                 schedule=None,
                 grid=None,
                 ratio_low=0,
                 ratio_medium=0
                 ):
        super().__init__()

        self.schedule = schedule
        self.grid = grid
        self.episode_length = episode_length  # 一次演化的时长

        # self.timestep = 0  # 环境当前处于的时间点=》mesa有默认的，我先去掉了
        # 在model中，用self.schedule.steps可以获取当前时间步

        self.ratio_low = ratio_low  # 低智能agent的比例
        self.ratio_medium = ratio_medium  # 中智能agent的比例
        self.ratio_high = 1 - self.ratio_low - self.ratio_medium  # 高智能agent的比例
        assert self.ratio_low + self.ratio_medium <= 1
        # 环境中每个智能体的当前奖赏值
        self.curr_optimization_metric = dict()

        # 环境中所有的企业
        self.all_agents = []
        self._agent_lookup = {str(agent.unique_id): agent for agent in self.all_agents}

        self.actions = None

    # 重置整个环境
    def reset(self):
        for agent in self.all_agents:
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



