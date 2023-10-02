# Generator for RL usage

import numpy as np
import gymnasium as gym
from ..generator import Generator


class GeneratorRL(Generator):

    def __init__(self,
                 name: str,
                 value: np.array,
                 lower_bound: np.array,
                 upper_bound: np.array,
                 cost: np.array,
                 cost_nde: np.array,
                 is_renewable: np.array):
        super().__init__(name, value, lower_bound, upper_bound, cost, cost_nde, is_renewable)

    def get_actions(self) -> gym.spaces.Dict:
        if self.is_renewable:
            actions = gym.spaces.Dict({
                'action': gym.spaces.Box(low=self.lower_bound, high=self.upper_bound, dtype=np.float32),
                'active': gym.spaces.Discrete(2)
            })
            return actions
        else:
            actions = gym.spaces.Dict({
                'action': gym.spaces.Box(low=self.lower_bound, high=self.upper_bound, dtype=np.float32)
            })
            return actions
