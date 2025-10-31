import copy

from .base_priority import Priority
from ..resources import (BaseResource, Aggregator, Generator,
                         Storage, Vehicle, Load)

import numpy as np
import pandas as pd


class WeightedPriorityV0(Priority):
    """
    Class to calculate the priority of resources in the environment
    based on their urgency, current available capacity.
    """

    def __init__(self, data,
                 urgency_weight: float = 1.0,
                 capacity_weight: float = 0.5,
                 log_scaling: bool = False):
        super().__init__(data)
        self.data = copy.deepcopy(data)

        # Priorities initialization
        self.priorities = None

        # Weights for the different factors
        self.urgency_weight = urgency_weight
        self.capacity_weight = capacity_weight

        # Log scaling
        self.log_scaling = log_scaling

    def calculate_priority(self):
        pass

    def initialize_priority(self):
        """
        Initialize the priority DataFrame with the priority of each resource.
        """

        # Create a DataFrame to store the priorities
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
                'urgency': float,
                'capacity': float
            }
        }
        """

        df = pd.DataFrame.from_dict(new_data, orient='index')

        # log(a) + log(b) = log(a*b)
        if self.log_scaling:
            df['priorities'] = np.log(df['urgency'] * self.urgency_weight *
                                      df['capacity'] * self.capacity_weight + 0.0001)
        else:
            df['priorities'] = (df['urgency'] * self.urgency_weight +
                                df['capacity'] * self.capacity_weight)

        df = df.transpose()
        df = df[self.priorities.columns]
        self.priorities.loc[timestep, :] = df.loc['priorities', :].values

        return
