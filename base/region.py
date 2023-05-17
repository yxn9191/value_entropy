import random
import mesa_geo
from shapely.geometry import Point

class Region(mesa_geo.GeoAgent):
    """Region agent. Define the region that agent move."""

    def __init__(
        self, unique_id, model, shape, agent_type="A"
    ):
        """
        Create a new Region agent.
        unique_id:   Unique identifier for the agent
        model:       Model in which the agent runs
        shape:    Shape object for the agent
        agent_type:  region_type("A","B","C")
        """
        super().__init__(unique_id, model, shape)
        self.atype = agent_type

    @property
    def random_point(self):
        min_x, min_y, max_x, max_y = self.shape.bounds
        random_point = Point(random.uniform(min_x, max_x), random.uniform(min_y, max_y))
        while 1:
            random_point = Point(random.uniform(min_x, max_x), random.uniform(min_y, max_y))
            if self.shape.contains(random_point):
                break
        return random_point

    def add_person_to_region(self, person, region_id):
        person.region_id = region_id
        person.geometry = self._id_region_map[region_id].random_point()
        self._id_region_map[region_id].add_person(person)
        super().add_agents(person)
        self.num_people += 1

    def remove_person_from_region(self, person):
        self._id_region_map[person.region_id].remove_person(person)
        person.region_id = None
        super().remove_agent(person)
        self.num_people -= 1

    def step(self):
        pass

    def __repr__(self):
        return "Region " + str(self.unique_id)