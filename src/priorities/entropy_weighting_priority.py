import copy

from .base_priority import Priority

import numpy as np
import pandas as pd


class EntropyWeightingPriorityV0(Priority):
    """
    Class to calculate the priority of resources in the environment
    based on their urgency, current available capacity.
    """

    def __init__(self, data):
        super().__init__(data)
        self.data = copy.deepcopy(data)

        # Priorities initialization
        self.priorities = pd.DataFrame({})

    def calculate_priority(self):
        pass

    def initialize_priority(self):
        self.priorities = pd.DataFrame({},
                                       columns=[x.name for x in self.data],
                                       index=np.arange(len(self.data[0].value + 1)))
        return

    def update_resources(self, new_data: dict, timestep: int):
        if not isinstance(new_data, dict) or len(new_data) == 0:
            raise ValueError("new_data must be a non-empty dictionary")

        # Efficient DataFrame creation
        df = pd.DataFrame.from_dict(new_data, orient='index')

        # Vectorized normalization
        col_min = df.min()
        col_max = df.max()
        normalized_df = (df - col_min) / (col_max - col_min + 1e-9)

        # Calculate probability matrix
        p = normalized_df / (normalized_df.sum(axis=0) + 1e-9)

        # Entropy weight calculation
        k = 1 / np.log(len(df))
        entropy = -k * (p * np.log(p + 1e-9)).sum(axis=0)
        diversification = 1 - entropy
        entropy_weights = diversification / diversification.sum()

        # Final priority
        priority_values = (df * entropy_weights).sum(axis=1)

        # Assign priorities to the correct timestep
        self.priorities.loc[timestep, priority_values.index] = priority_values

        return
