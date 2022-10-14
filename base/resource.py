import mesa


class Resource(mesa.Agent):
    """
    Base class for Resource classes.

    Args:
        unique_id: The unique id of the Resource. It can be None if you don't need it.
        model: Instance of the model that contains the agent.
        pos: The position of the resource.
        time_start: The time when the resource appears in the environment.
        time_end: The time when the resource disappears in the environment.
        bonus: The bonus of the agent to collect the resource.
        cost: The cost of the agent to collect the resource.
        skills: Skills required to collect this resource.
    """
    name = None
    collectible = None  # 是否可收集

    def __init__(self,
                 unique_id=None,
                 model=None,
                 time_start=None,
                 time_end=None,
                 bonus=None,
                 cost=None,
                 skills=None,
                 ):

        assert self.name is not None
        assert self.collectible is not None
        
        super().__init__(unique_id, model)
        self.time_start = time_start
        self.time_end = time_end
        self.bonus = bonus
        self.cost = cost
        self.skills = skills

    def step(self):
        pass

    # 构造可变长度的技能向量，用于匹配，有了这个向量可以利用约束条件，感觉应该是比较通用的
    def match_vector(self, **skills):
        self.skills = skills
        return self.skills


