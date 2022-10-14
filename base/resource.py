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
        skill_requirements: Skills required to collect this resource.
    """
    name = None
    collectible = None  # 是否可收集

    def __init__(self,
                 unique_id=None,
                 model=None,
                 pos=None,
                 time_start=None,
                 time_end=None,
                 bonus=None,
                 cost=None,
                 skill_requirements=None,
                 ):

        assert self.name is not None
        assert self.collectible is not None

        self.time_start = time_start
        self.time_end = time_end
        self.bonus = bonus
        self.cost = cost
        self.skill_requirements = dict(skill_requirements)

    def step(self):
        pass



