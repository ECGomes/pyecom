# Extends the BaseResource class to provide a vehicle resource

import numpy as np
from src.resources.base_resource import BaseResource


class Vehicle(BaseResource):

    def __init__(self,
                 name: str,
                 value: np.array,
                 lower_bound: np.array,
                 upper_bound: np.array,
                 cost: np.array,
                 cost_discharge: np.array,
                 cost_charge: np.array,
                 capacity_max: np.array,
                 initial_charge: np.array,
                 min_charge: np.array,
                 discharge_efficiency: np.array,
                 charge_efficiency: np.array,
                 capital_cost: np.array,
                 schedule_discharge: np.array,
                 schedule_charge: np.array
                 ):
        super().__init__(name, value, lower_bound, upper_bound, cost)

        self.capacity_max = capacity_max
        self.initial_charge = initial_charge
        self.capital_cost = capital_cost
        self.min_charge = min_charge

        self.discharge_efficiency = discharge_efficiency
        self.discharge = np.zeros(self.value.shape)
        self.cost_discharge = cost_discharge
        self.is_discharging = np.zeros(self.value.shape)
        self.schedule_discharge = schedule_discharge

        self.charge_efficiency = charge_efficiency
        self.charge = np.zeros(self.value.shape)
        self.cost_charge = cost_charge
        self.is_charging = np.zeros(self.value.shape)
        self.schedule_charge = schedule_charge
