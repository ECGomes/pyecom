import gymnasium as gym
import numpy as np
import random
from copy import copy
from collections import OrderedDict
from ...resources.base_resource import BaseResource
from ...resources.vehicle import Vehicle

from ray.rllib.env.multi_agent_env import MultiAgentEnv


class EVMultiAgent_v0(MultiAgentEnv):
    """
    A multi-agent environment for the EV charging problem.
    Penalties are applied for EVs not being charged at time of departure and illegal actions;
    Reward at each timestep is the negative sum of penalties and charging costs:
    reward = - (penalty + charging_cost)

    Penalties for illegal actions are applied as follows:
    penalty_action = coefficient * (action_delta)^2
    where action_delta is the difference between the current action and the allowed bounds

    Penalty for not being charged at time of departure is applied as a constant
    """

    def __init__(self, resources: list[Vehicle],
                 penalty_action_coefficient: float = 1000,
                 penalty_not_charged: float = 1000,
                 energy_price: np.array = None,
                 ):
        super().__init__()

        # Define the Resources
        self.evs = resources
        self.energy_price = energy_price

        # Define penalties
        self.penalty_action_coefficient = penalty_action_coefficient
        self.penalty_not_charged = penalty_not_charged

        # Initialize timestep counter
        self.current_timestep = 0
