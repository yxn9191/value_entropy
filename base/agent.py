import mesa


class Agent(mesa.Agent):
    """
     Base class for Agent classes.
     Args:
        unique_id:
        model:
        pos:
        speed:
        vision:
        consumption:
        failure_prob:
        skills:
        multi_action_mode:
    """
    name = ""

    def __init__(self, unique_id,
                 model,
                 speed=None,
                 vision=None,
                 energy=None,
                 consumption=None,
                 failure_prob=0,
                 skills=None,
                 state=0,
                 multi_action_mode=False
                 ):
        assert self.name
        super().__init__(unique_id, model)
        if unique_id is None:
            unique_id = 0

        self.speed = speed
        self.vision = vision,
        self.energy = energy
        self.consumption = consumption,
        self.failure_prob = float(failure_prob),
        self.skills = skills
        self.state = state  # define the state of the agent
        self.multi_action_mode = bool(multi_action_mode)
        self.action = None
        self.done = False  # 是否可以被移除

    # 构造可变长度的技能向量，用于匹配，有了这个向量可以利用约束条件，感觉应该是比较通用的
    def match_vector(self, **skills):
        self.skills = skills
        return self.skills

    @property
    def action_space(self):
        pass

    def action_parse(self, action):
        self.action = action

    def reset(self):
        self.state = 0

    def move(self):
        self.energy = self.energy - self.consumption

    def destroy(self):
        self.done = True

    def step(self):
        pass
