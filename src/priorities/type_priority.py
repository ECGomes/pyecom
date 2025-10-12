import copy

from .base_priority import Priority
from ..resources import (BaseResource, Aggregator, Generator,
                         Storage, Vehicle, Load)
import numpy as np
import pandas as pd


class EmpiricalPriority(Priority):

    def __init__(self, data):
        super().__init__(data)
        self.data = copy.deepcopy(data)

    def calculate_group_priority(self):
        """
        Calculate the priority of resource types
        Renewable generator: 5
        EV: 4
        Battery: 3
        Non-renewable generator: 2
        Aggregator: 1
        Load: 0
        """

        # Create a list of priorities
        priorities = []
        for resource in self.data:
            if resource.istype(Load):
                priorities.append(5.0)
            elif resource.istype(Generator):
                if resource.is_renewable:
                    priorities.append(4.0)
                else:
                    priorities.append(1.0)
            elif resource.istype(Vehicle):
                priorities.append(3.0)
            elif resource.istype(Storage):
                priorities.append(2.0)
            elif resource.istype(Aggregator):
                priorities.append(0.0)
            else:
                raise ValueError(f"Resource {resource} not supported")

        return np.array(priorities)

    def calculate_resource_size_priority(self, type: BaseResource):
        """
        Calculate the priority of individual resources within a group
        General idea is that the group priority defines the integer part and
        the resource size defines the decimal part
        """

        # Get the resources of the group that was passed
        resources = [x for x in self.data if x.istype(type)]
        res_size = []

        # Get the size of the resources
        # If it's a generator, get the maximum generation
        if type == Generator:
            for res in resources:
                res_size.append(np.max(res.upper_bound))

        # If it's a storage, get the maximum capacity
        elif type == Storage:
            for res in resources:
                res_size.append(np.max(res.capacity_max))

        # If it's a vehicle, get the maximum capacity
        elif type == Vehicle:
            for res in resources:
                res_size.append(np.max(res.capacity_max))

        # If it's a load, get the maximum demand
        elif type == Load:
            for res in resources:
                res_size.append(np.max(res.upper_bound))

        # If it's an aggregator it's 0
        elif type == Aggregator:
            res_size = 0

        else:
            raise ValueError(f"Resource {type} not supported")

        # Calculate and attribute a decimal part
        res_size = res_size / np.max(res_size) if type != Aggregator else 0

        # Invert the priority, so the bigger the resource, the smaller the priority
        res_size = 1 - res_size
        individual_priority = res_size * 0.9

        return individual_priority

    def calculate_priority_df(self):
        """
        Calculate the priority list
        """

        # Create a DataFrame to use matrix operations
        df = pd.DataFrame({})
        df['priority'] = self.calculate_group_priority()
        df['type'] = [x.get_type() for x in self.data]
        df['name'] = [x.name for x in self.data]

        # Calculate the priority of the resources
        for type in df['type'].unique():
            priority = self.calculate_resource_size_priority(type)
            df.loc[df['type'] == type, 'priority'] += np.array(priority)

        return df

    def calculate_priority(self, ascending: bool | None = False):
        """
        Return the priority list
        """

        priority_df = self.calculate_priority_df()
        if ascending is not None:
            return priority_df.sort_values(by='priority', ascending=ascending)
        else:
            return priority_df
