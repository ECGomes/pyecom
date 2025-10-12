import copy

from .base_priority import Priority
from .type_priority import EmpiricalPriority
from ..resources import (BaseResource, Aggregator, Generator,
                         Storage, Vehicle, Load)
import numpy as np
import pandas as pd


class ContributionPriority(Priority):
    """
    Calculate the priority of resources based on their contribution to the system.
    The contribution is calculated as the fraction of the resource's contribution to the
    total system consumption and total system generation.

    Each resource is assigned a priority based on the following formula:
    priority = (total_resource_contribution / total_system_contribution) +
               (total_resource_generation / total_system_generation)

    The priority is normalized to the range [0, 1] by dividing by the maximum priority.

    Method calculate_priority() is not used, as update_resources handles the priority update.
    """

    def __init__(self, data,
                 alpha: float = 0.5,
                 beta: float = 0.5,
                 log_scaling: bool = False):
        """
        Initialize the ContributionPriority object.
        :param data: List of resources
        :param alpha: Weight of the contribution to the total generation
        :param beta: Weight of the contribution to the total consumption
        :param log_scaling: If True, each component of priority is scaled logarithmically
        """
        super().__init__(data)
        self.data = copy.deepcopy(data)

        self.priorities = None

        # Parameters for the priority calculation
        self.alpha = alpha
        self.beta = beta
        self.log_scaling = log_scaling

    def calculate_priority(self):
        pass

    def initialize_priority(self):
        """
        Initialize the priority DataFrame with the priority of each resource.
        Since there is no contribution at timestep 0, the priority is set to empirical defaults.
        """

        initial_priority = EmpiricalPriority(self.data).calculate_priority(ascending=None)

        self.priorities = pd.DataFrame({},
                                       columns=[x.name for x in self.data],
                                       index=np.arange(len(self.data[0].value + 1)))

        for resource in self.data:
            temp_priority = initial_priority.loc[initial_priority['name'] ==
                                                 resource.name]['priority'].values[0]
            self.priorities.loc[0, resource.name] = temp_priority

        return

    def update_resources(self, new_data: dict, timestep: int):
        """
        Update the resources with new data.
        The environment must be updated with the new data before calling this method.
        Dictionary Structure:
        {
            'resource_name': {
                'generation': series of floats,
                'consumption': series of floats
            }
        }
        """

        # Sum all the generation and consumption values
        total_generation = sum([sum(x['generation']) for x in new_data.values()]) + 1
        total_consumption = sum([sum(x['consumption']) for x in new_data.values()]) + 1

        # Update the priority of each resource
        for resource in new_data.keys():

            if self.log_scaling:
                total_generation = np.log(total_generation + 1)
                total_consumption = np.log(total_consumption + 1)

                contribution = self.alpha * (np.log(sum(new_data[resource]['generation']) + 1) / total_generation) + \
                               self.beta * (np.log(sum(new_data[resource]['consumption']) + 1) / total_consumption)

                # Update the priority of the resource
                self.priorities.loc[timestep, resource] = contribution

            else:
                if total_generation == 0 and total_consumption == 0:
                    self.priorities.loc[timestep, resource] = 0
                elif total_generation == 0:
                    self.priorities.loc[timestep, resource] = sum(new_data[resource]['consumption']) / total_consumption
                elif total_consumption == 0:
                    self.priorities.loc[timestep, resource] = sum(new_data[resource]['generation']) / total_generation
                else:
                    # Calculate the contribution of the resource
                    contribution = self.alpha * (sum(new_data[resource]['generation']) / total_generation) + \
                                   self.beta * (sum(new_data[resource]['consumption']) / total_consumption)

                    # Update the priority of the resource
                    self.priorities.loc[timestep, resource] = contribution

        return
