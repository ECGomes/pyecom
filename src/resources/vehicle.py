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

    def sample(self,
               avg_daily_trips: int = 2, stddev_daily_trips: int = 1,
               avg_soc_arrival_percentage: float = 0.2, stddev_soc_arrival_percentage: float = 0.1,
               avg_soc_departure_percentage: float = 0.3, stddev_soc_departure_percentage: float = 0.1,
               n_timeslots: int = 24):
        """
        Creates new samples for the vehicle
        :param avg_daily_trips: Average daily trips to be made by the vehicle
        :param stddev_daily_trips: Standard deviation of the daily trips to be made by the vehicle
        :param avg_soc_arrival_percentage: Average percentage of the battery the EV will arrive with
        :param stddev_soc_arrival_percentage: Standard deviation of the percentage of the battery on arrival
        :param avg_soc_departure_percentage: Average percentage of the battery that will be used per trip
        :param stddev_soc_departure_percentage: Standard deviation of the percentage of the battery on departure
        :param n_timeslots: Number of timeslots in the schedule
        :return:
        """

        # Sample the number of trips using a normal distribution with center at avg_daily_trips
        trips = int(np.random.normal(avg_daily_trips, stddev_daily_trips, 1)[0])

        # Sample arrival and departure soc for scheduling
        arrival_soc = np.random.normal(avg_soc_arrival_percentage,
                                       stddev_soc_arrival_percentage, trips).round(2)
        departure_soc = np.random.normal(avg_soc_departure_percentage,
                                         stddev_soc_departure_percentage, trips).round(2)

        # Create timeslots for the arrivals and departures
        timeslots = np.random.choice(n_timeslots, trips * 2, replace=False)
        timeslots = np.sort(timeslots)

        # Create the schedules
        schedule_connected = np.zeros(n_timeslots)
        schedule_requirement_soc = np.zeros(n_timeslots)
        schedule_arrival_soc = np.zeros(n_timeslots)
        for i in range(trips):
            # Connected
            schedule_connected[timeslots[i * 2]:timeslots[i * 2 + 1]] = 1

            # Requirement SOC
            schedule_requirement_soc[timeslots[i * 2 + 1]] = departure_soc[i]

            # Arrival SOC
            schedule_arrival_soc[timeslots[i * 2]] = arrival_soc[i]

        schedule_discharge = schedule_connected.copy() * np.max(self.schedule_discharge)
        schedule_charge = schedule_connected.copy() * np.max(self.schedule_charge)

        return trips, schedule_connected, \
            schedule_discharge, schedule_charge, \
            schedule_requirement_soc, schedule_arrival_soc

