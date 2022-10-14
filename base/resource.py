import mesa


class Resource(mesa.Agent):
    """
    Base class for Resource classes.

    Args:
        unique_id: The unique id of the Resource. It can be None if you don't need it.
        model: Instance of the model that contains the agent.
        time_start: The time when the resource appears in the environment.
        time_end: The time when the resource disappears in the environment.
        operate_type: The type of operate to finish the work. It can be None if you don't need to collect the resource.
        skill_requirements: Skills required to collect this resource。



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
                 operate_type=None,
                 skill_requirements=None,
                 ):

        assert self.name is not None
        assert self.collectible is not None
        
    def step(self):
        pass



