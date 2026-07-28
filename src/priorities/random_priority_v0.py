import copy

from .base_priority import Priority
from ..resources import (BaseResource, Aggregator, Generator,
                         Storage, Vehicle, Load)

import numpy as np
import pandas as pd


class RandomPriorityV0(Priority):
    """
    Class to calculate the priority of resources in the environment
    based on random permutations.
    """

    def __init__(self, data,
                 seed: int = 42):
        super().__init__(data)
        self.data = copy.deepcopy(data)
        self.seed = seed

        # Priorities initialization
        self.priorities = None

        # Initialize the random rng
        self.rng = np.random.default_rng(seed)

    def calculate_priority(self):
        pass

    def initialize_priority(self):
        """
        Initialize the priority DataFrame with the priority of each resource.
        """

        # Create a DataFrame to store the priorities
        self.priorities = list()
        self.priorities.append(np.array(self.data))

        return

    def update_priority(self):
        """
        Add a new order by permutating the last one.
        """

        self.priorities.append(np.array(self.rng.permutation(self.priorities[-1])))

        return
