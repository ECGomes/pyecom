# Extends the BaseResource class to provide a storage resource

import numpy as np
from src.resources.base_resource import BaseResource

from typing import Union


class Storage(BaseResource):
    def __init__(self,
                 name: str,
                 value: Union[np.array, float],
                 lower_bound: np.array(float),
                 upper_bound: np.array(float),
                 cost: np.array(float),
                 cost_discharge: np.array(float),
                 cost_charge: np.array(float),
                 capacity_max: np.array(float),
                 capacity_min: np.array(float),
                 initial_charge: np.array(float),
                 discharge_efficiency: np.array(float),
                 discharge_max: np.array(float),
                 charge_efficiency: np.array(float),
                 charge_max: np.array(float),
                 capital_cost: np.array(float),
                 ):
        super().__init__(name, value, lower_bound, upper_bound, cost)

        self.capacity_max = capacity_max
        self.capacity_min = capacity_min
        self.initial_charge = initial_charge
        self.charge_efficiency = charge_efficiency
        self.discharge_efficiency = discharge_efficiency
        self.capital_cost = capital_cost

        self.discharge = np.zeros(self.value.shape) \
            if isinstance(value, np.ndarray) else 0.0
        self.discharge_max = discharge_max
        self.cost_discharge = cost_discharge

        self.charge = np.zeros(self.value.shape) \
            if isinstance(value, np.ndarray) else 0.0
        self.charge_max = charge_max
        self.cost_charge = cost_charge

        self.emin_relax = np.zeros(self.value.shape) \
            if isinstance(value, np.ndarray) else 0.0
