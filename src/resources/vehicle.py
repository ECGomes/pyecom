# Extends the BaseResource class to provide a vehicle resource

import numpy as np
from typing import Union
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
                 capacity_max: Union[np.array, float],
                 initial_charge: Union[np.array, float],
                 min_charge: Union[np.array, float],
                 discharge_efficiency: Union[np.array, float],
                 charge_efficiency: Union[np.array, float],
                 schedule_discharge: np.array = None,
                 schedule_charge: np.array = None,
                 schedule_requirement_soc: np.array = None,
                 schedule_arrival_soc: np.array = None,
                 capital_cost: Union[np.array, float] = None,
                 ):
        super().__init__(name, value, lower_bound, upper_bound, cost)

        self.capacity_max = capacity_max
        self.initial_charge = initial_charge
        self.capital_cost = capital_cost
        self.min_charge = min_charge

        # Discharging variables
        self.discharge_efficiency = discharge_efficiency
        self.discharge = np.zeros(self.value.shape)
        self.cost_discharge = cost_discharge
        self.is_discharging = np.zeros(self.value.shape)
        self.schedule_discharge = schedule_discharge

        # Charging variables
        self.charge_efficiency = charge_efficiency
        self.charge = np.zeros(self.value.shape)
        self.cost_charge = cost_charge
        self.is_charging = np.zeros(self.value.shape)
        self.schedule_charge = schedule_charge

        # Schedule variables
        self.schedule_requirement_soc = schedule_requirement_soc
        self.schedule_arrival_soc = schedule_arrival_soc
