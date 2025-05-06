import copy

from .base_priority import Priority
from ..resources import (BaseResource, Aggregator, Generator,
                         Storage, Vehicle, Load)

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
        """
        Update the resources with new data and calculate the priority.
        Dictionary structure:
        {
            'resource_name': {
                'discharge_cost': float,
                'charge_cost': float,
                'soc': float,
                'connected': bool,
                'max_discharge_power': float,
                'max_charge_power': float,
                'grid_balance': float
            }
        }
        """

        # Build a DataFrame with the new data
        if not isinstance(new_data, dict):
            raise ValueError("new_data must be a dictionary")

        if len(new_data) == 0:
            raise ValueError("new_data cannot be empty")

        df = pd.DataFrame({},
                          index=list(new_data.keys()),
                          columns=new_data[list(new_data.keys())[0]].keys())
        for resource in new_data:
            for key in new_data[resource]:
                df.loc[resource, key] = new_data[resource][key]

        # Calculate the priority
        normalized_df = df.copy()
        for col in df.columns:
            col_min = df[col].min()
            col_max = df[col].max()
            normalized_df[col] = (df[col] - col_min) / (col_max - col_min + 1e-9)

        p = normalized_df / (normalized_df.sum(axis=0) + 1e-9)
        p = p.astype(float)

        # Calculate the entropy
        k = 1 / np.log(len(df))
        entropy = -k * (p * np.log(p + 1e-9)).sum(axis=0)

        # Calculate diversification
        diversification = 1 - entropy

        # Calculate the weights
        entropy_weights = diversification / diversification.sum()

        # Calculate the priority
        priority_values = (df * entropy_weights.values).sum(axis=1)

        # Assign the values according to the timestep
        for i, resource in enumerate(df.index):
            self.priorities.loc[timestep, resource] = priority_values[i]


        return
