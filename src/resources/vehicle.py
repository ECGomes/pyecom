# Extends the BaseResource class to provide a vehicle resource

import numpy as np
from typing import Union
from src.resources.base_resource import BaseResource


class Vehicle(BaseResource):

    def __init__(self,
                 name: str,
                 value: Union[np.ndarray, float],
                 lower_bound: np.ndarray,
                 upper_bound: np.ndarray,
                 cost: np.ndarray,
                 cost_discharge: np.ndarray,
                 cost_charge: np.ndarray,
                 capacity_max: Union[np.ndarray, float],
                 initial_charge: Union[np.ndarray, float],
                 min_charge: Union[np.ndarray, float],
                 discharge_efficiency: Union[np.ndarray, float],
                 charge_efficiency: Union[np.ndarray, float],
                 schedule_connected: np.ndarray = None,
                 schedule_discharge: np.ndarray = None,
                 schedule_charge: np.ndarray = None,
                 schedule_requirement_soc: np.ndarray = None,
                 schedule_arrival_soc: np.ndarray = None,
                 capital_cost: Union[np.ndarray, float] = None,
                 ):
        super().__init__(name, value, lower_bound, upper_bound, cost)

        self.capacity_max = capacity_max
        self.initial_charge = initial_charge
        self.capital_cost = capital_cost
        self.min_charge = min_charge

        # Discharging variables
        self.discharge_efficiency = discharge_efficiency
        self.discharge = np.zeros(self.value.shape) if isinstance(self.value, np.ndarray) else 0.0
        self.cost_discharge = cost_discharge
        self.is_discharging = np.zeros(self.value.shape) if isinstance(self.value, np.ndarray) else 0.0
        self.schedule_discharge = schedule_discharge

        # Charging variables
        self.charge_efficiency = charge_efficiency
        self.charge = np.zeros(self.value.shape) if isinstance(self.value, np.ndarray) else 0.0
        self.cost_charge = cost_charge
        self.is_charging = np.zeros(self.value.shape) if isinstance(self.value, np.ndarray) else 0.0
        self.schedule_charge = schedule_charge

        # Schedule variables
        self.schedule_connected = schedule_connected
        self.schedule_requirement_soc = schedule_requirement_soc
        self.schedule_arrival_soc = schedule_arrival_soc
