# Extends the BaseResource class to provide a storage resource

from pyecom.resources.base_resource import BaseResource
import numpy as np


class Storage(BaseResource):
    def __init__(self,
                 name: str,
                 value: np.array,
                 lb: np.array,
                 ub: np.array,
                 cost: np.array,
                 max_capacity: np.array,
                 initial_charge: np.array,
                 state_of_charge: np.array,
                 charge_efficiency: np.array,
                 discharge_efficiency: np.array,
                 capital_cost: np.array,
                 ):
        super().__init__(name, value, lb, ub, cost)

        self.max_capacity = max_capacity
        self.initial_charge = initial_charge
        self.state_of_charge = state_of_charge
        self.charge_efficiency = charge_efficiency
        self.discharge_efficiency = discharge_efficiency
        self.capital_cost = capital_cost
