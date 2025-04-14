import copy

from .base_priority import Priority
from ..resources import (BaseResource, Aggregator, Generator,
                         Storage, Vehicle, Load)

import numpy as np
import pandas as pd

class WeightedPriorityV0(Priority):
    """
    Class to calculate the priority of resources in the environment
    based on their urgency, SoC range.
    """

    def __init__(self, data,
                 urgency_weight: float = 1.0,
                 soc_weight: float = 0.5,
                 log_scaling: bool = False):
        super().__init__(data)
        self.data = copy.deepcopy(data)

        # Priorities initialization
        self.priorities = None

        # Weights for the different factors
        self.urgency_weight = urgency_weight
        self.soc_weight = soc_weight

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
                'urgency': int,
                'soc_range': float
            }
        }
        """

        # Update the resources
        for resource in new_data.keys():

            if self.log_scaling:
                self.priorities.loc[timestep, resource] = \
                    np.log(new_data[resource]['urgency'] + 1) * self.urgency_weight + \
                    np.log(new_data[resource]['soc_range'] + 1) * self.soc_weight

            else:
                self.priorities.loc[timestep, resource] = \
                    new_data[resource]['urgency'] * self.urgency_weight + \
                    new_data[resource]['soc_range'] * self.soc_weight

        return
