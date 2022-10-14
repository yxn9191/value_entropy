import math
from copy import deepcopy
from random import randint
from tkinter import N
import numpy as np
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
    def __init__(self, unique_id ,
                 model ,
                 pos ,
                 speed=None,
                 vision=None,
                 energy=None,
                 consumption=None,
                 failure_prob=0,
                 skills=None,
                 multi_action_mode=None
                 ):

        assert self.name
        super().__init(unique_id, model, pos)
        if unique_id is None:
            unique_id = 0
        
        self.speed = speed
        self.vision = vision,
        self.energy = energy
        self.consumption = consumption,
        self.failure_prob = float(failure_prob),
        self.skills = skills
        self.state = 0 # define the state of the agent
        self.multi_action_mode = bool(multi_action_mode)

    def action_space(self):
        pass

    def reset(self):
        self.state = 0
    
    def move(self):
        self.energy = self.energy - self.consumption

    def step(self, actions=None):
        pass