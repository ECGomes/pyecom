from .base_repair import BaseRepair
from ..parsers.hm_parser import HMParser
import numpy as np


class HMRepair(BaseRepair):

    def __init__(self, data, parsed_data: HMParser, set_vehicles):
        self.components = data
        self.parsed_data = parsed_data

        # Set the initial variables to work with
        self.__initial_variables__ = {}
        self.__var_idx__ = []
        self.__var_names__ = []

        self.n_steps = self.components['gen'].value.shape[1]
        self.n_gen = self.components['gen'].value.shape[0]
        self.n_load = self.components['loads'].value.shape[0]
        self.n_stor = self.components['stor'].value.shape[0]
        self.n_v2g = self.components['evs'].value.shape[0]

        self._initialize_values()

        # Set the vehicles to a deterministic state
        self.set_vehicles = set_vehicles
        if self.set_vehicles:
            self.set_v2g(self.__initial_variables__)

        self.storCapCost = self.components['stor'].capital_cost
        self.v2gCapCost = self.components['evs'].capital_cost

        super().__init__()

    def _initialize_values(self):
        """
        Initialize the values.
        Only necessary if values are not initialized in the algorithm.
        :return:
        """

        temp_vars = {'genActPower': np.zeros((self.n_gen, self.n_steps)),
                     'genExcActPower': np.zeros((self.n_gen, self.n_steps)),
                     'pImp': np.zeros(self.n_steps),
                     'pExp': np.zeros(self.n_steps),
                     'loadRedActPower': np.zeros((self.n_load, self.n_steps)),
                     'loadCutActPower': np.zeros((self.n_load, self.n_steps)),
                     'loadENS': np.zeros((self.n_load, self.n_steps)),
                     'storDchActPower': np.zeros((self.n_stor, self.n_steps)),
                     'storChActPower': np.zeros((self.n_stor, self.n_steps)),
                     'EminRelaxStor': np.zeros((self.n_stor, self.n_steps)),
                     'storEnerState': np.zeros((self.n_stor, self.n_steps)),
                     'v2gDchActPower': np.zeros((self.n_v2g, self.n_steps)),
                     'v2gChActPower': np.zeros((self.n_v2g, self.n_steps)),
                     'EminRelaxEV': np.zeros((self.n_v2g, self.n_steps)),
                     'v2gEnerState': np.zeros((self.n_v2g, self.n_steps)),
                     'genXo': np.zeros((self.n_gen, self.n_steps)),
                     'loadXo': np.zeros((self.n_load, self.n_steps)),
                     'storDchXo': np.zeros((self.n_stor, self.n_steps)),
                     'storChXo': np.zeros((self.n_stor, self.n_steps)),
                     'v2gDchXo': np.zeros((self.n_v2g, self.n_steps)),
                     'v2gChXo': np.zeros((self.n_v2g, self.n_steps))}

        self.__var_idx__ = [temp_vars[v].ravel().shape[0] for v in temp_vars.keys()]
        self.__var_names__ = list(temp_vars.keys())

        self.__initial_variables__ = temp_vars
        return

    def check_imports_exports(self, x):
        # Clip the values
        x['pImp'] = np.clip(x['pImp'], 0, self.components['pimp'].upper_bound)
        x['pExp'] = np.clip(x['pExp'], 0, self.components['pexp'].upper_bound)

        # Set exclusivity
        x['pExp'] = x['pExp'] * (1 - x['pImp'] > 0.0)
        return

    def check_generators(self, x):
        # Binary variable
        x['genXo'] = (x['genXo'] > 0.5).astype(int)

        # Clip the values
        x['genActPower'] = np.clip(x['genActPower'], np.zeros(x['genActPower'].shape),
                                   self.components['gen'].upper_bound)

        # Set the excess power to 0 and fix from there
        x['genExcActPower'] = np.zeros(x['genExcActPower'].shape)

        # Safeguard for 1 generator
        if self.n_gen == 1:
            # Check type 1 generators
            if self.components['gen'].is_renewable == 1:
                x['genActPower'] = self.components['gen'].upper_bound * x['genXo']
            else:
                x['genExcActPower'] = (self.components['gen'].upper_bound - x['genActPower'])

            return

        # Generator types
        # Type 1 (non-renewable)
        mask = self.components['gen'].is_renewable == 1
        x['genActPower'][mask] = (self.components['gen'].upper_bound * x['genXo'])[mask]

        # Type 2 (renewable)
        mask = self.components['gen'].is_renewable == 2
        x['genExcActPower'][mask] = (self.components['gen'].upper_bound - x['genActPower'])[mask]

        # Clip again to make sure
        x['genActPower'] = np.clip(x['genActPower'], np.zeros(x['genActPower'].shape),
                                   self.components['gen'].upper_bound)
        x['genExcActPower'] = np.clip(x['genExcActPower'], np.zeros(x['genExcActPower'].shape),
                                      self.components['gen'].upper_bound)
        return

    def check_loads(self, x):
        # Assign the binary variable
        x['loadXo'] = (x['loadXo'] > 0.5).astype(int)

        # Load reduction
        x['loadRedActPower'] = np.clip(x['loadRedActPower'], 0, self.components['loads'].upper_bound)

        # Load curtailing
        x['loadCutActPower'] = self.components['loads'].upper_bound * x['loadXo']

        # Load ENS
        temp_vals = self.components['loads'].upper_bound - x['loadRedActPower'] - x['loadCutActPower']
        x['loadENS'] = np.clip(x['loadENS'], 0, temp_vals)
        x['loadENS'][x['loadENS'] < 0] = 0.0
        return

    def check_storage(self, x):
        # Assign the binary variables
        x['storDchXo'] = (x['storDchXo'] > 0.5).astype(int)
        x['storChXo'] = (x['storChXo'] > 0.5).astype(int)

        # Mutual exclusion
        x['storDchXo'] = x['storDchXo'] * (1 - x['storChXo'])

        # Value clipping
        x['storDchActPower'] = np.clip(x['storDchActPower'], np.zeros(x['storDchActPower'].shape),
                                       self.components['stor'].discharge_max)
        x['storChActPower'] = np.clip(x['storChActPower'], np.zeros(x['storChActPower'].shape),
                                      self.components['stor'].charge_max)

        # Clip the energy state to the maximum allowed
        x['storEnerState'] = np.clip(x['storEnerState'], np.zeros(x['storEnerState'].shape),
                                     (np.ones(x['storEnerState'].shape).transpose() *
                                      self.components['stor'].capacity_max).transpose())

        x['EminRelaxStor'] = np.clip(x['storEnerState'], np.zeros(x['storEnerState'].shape),
                                     (np.ones(x['storEnerState'].shape).transpose() *
                                      self.components['stor'].capacity_max).transpose())

        # Initial state of charge
        x['storEnerState'][:, 0] = self.components['stor'].capacity_max * self.components['stor'].initial_charge + \
                                   x['storChActPower'][:, 0] * self.components['stor'].charge_efficiency - \
                                   x['storDchActPower'][:, 0] / self.components['stor'].discharge_efficiency

        # Initialize the iterator and range (we already did the initial timestep!)
        t: int = 1
        t_range = range(1, self.n_steps)

        # Fix the timestep dependencies
        for t in t_range:
            # Check if charging
            charged = x['storChXo'][:, t] * x['storChActPower'][:, t] * \
                      (1 - self.components['stor'].charge_efficiency)

            # Prevent over charging
            secondary_mask = (x['storEnerState'][:, t - 1] + charged) > self.components['stor'].capacity_max
            x['storChActPower'][:, t][secondary_mask] = \
                ((self.components['stor'].capacity_max - x['storEnerState'][:, t - 1]) /
                 self.components['stor'].charge_efficiency)[secondary_mask]

            # Check if discharging
            discharged = x['storDchXo'][:, t] * x['storDchActPower'][:, t] \
                         / self.components['stor'].discharge_efficiency
            secondary_mask = (x['storEnerState'][:, t - 1] - discharged) < 0
            x['storDchActPower'][:, t][secondary_mask] = (x['storEnerState'][:, t - 1] *
                                                          self.components['stor'].discharge_efficiency)[secondary_mask]

            # Update the energy state
            x['storEnerState'][:, t] = x['storEnerState'][:, t - 1] + x['storChActPower'][:, t] * \
                                       self.components['stor'].charge_efficiency - \
                                       x['storDchActPower'][:, t] / self.components['stor'].discharge_efficiency

            # Check minimum energy state
            mask = x['storEnerState'][:, t] < self.components['stor'].capacity_max * \
                   self.components['stor'].capacity_min - x['EminRelaxStor'][:, t]
            x['storEnerState'][:, t][mask] = self.components['stor'].capacity_max[mask] * \
                                             self.components['stor'].capacity_min[mask] - \
                                             x['EminRelaxStor'][:, t][mask]

        # Clip the values of discharging and charging to the maximum allowed
        x['storDchActPower'] = np.clip(x['storDchActPower'], np.zeros(x['storDchActPower'].shape),
                                       self.components['stor'].discharge_max)
        x['storChActPower'] = np.clip(x['storChActPower'], np.zeros(x['storChActPower'].shape),
                                      self.components['stor'].charge_max)

        # Clip the energy state to the maximum allowed
        x['storEnerState'] = np.clip(x['storEnerState'], np.zeros(x['storEnerState'].shape),
                                     (np.ones(x['storEnerState'].shape).transpose() *
                                      self.components['stor'].capacity_max).transpose())

        return

    def check_v2g(self, x):

        # Placeholders for the energy state and relaxation variable
        x['v2gEnerState'] = np.zeros(x['v2gEnerState'].shape)
        x['EminRelaxEV'] = np.zeros(x['EminRelaxEV'].shape)

        # Bound binaries
        x['v2gDchXo'] = (x['v2gDchXo'] > 0.5).astype(int)
        x['v2gChXo'] = (x['v2gChXo'] > 0.5).astype(int)

        # Mutually exclusive
        x['v2gDchXo'] = x['v2gDchXo'] * (1 - x['v2gChXo'])

        # Preallocate range
        t_range = range(1, self.n_steps)

        # Clip the values of discharging and charging to the maximum allowed
        x['v2gDchActPower'] = np.clip(x['v2gDchActPower'], 0, self.components['evs'].schedule_discharge)
        x['v2gChActPower'] = np.clip(x['v2gChActPower'], 0, self.components['evs'].schedule_charge)

        # Set initial EV state
        x['v2gEnerState'][:, 0] = self.components['evs'].capacity_max * 0.8

        # Fix the timestep dependencies
        for t in t_range:
            # Check if charging
            # mask = x['v2gChXo'][:, t] > np.zeros(x['v2gChXo'][:, t].shape)
            charged = x['v2gChActPower'][:, t] * (1 - self.components['evs'].charge_efficiency)

            # Prevent over charging
            secondary_mask = (x['v2gEnerState'][:, t - 1] + charged) > self.components['evs'].capacity_max
            x['v2gChActPower'][:, t][secondary_mask] = ((self.components['evs'].capacity_max -
                                                         x['v2gEnerState'][:, t - 1]) /
                                                        self.components['evs'].charge_efficiency)[secondary_mask]

            # Check if discharging
            # mask = x['v2gDchXo'][:, t] > np.zeros(x['v2gDchXo'][:, t].shape)
            discharged = x['v2gDchActPower'][:, t] / self.components['evs'].discharge_efficiency
            secondary_mask = (x['v2gEnerState'][:, t - 1] - discharged) < 0
            x['v2gDchActPower'][:, t][secondary_mask] = (x['v2gEnerState'][:, t - 1] *
                                                         self.components['evs'].discharge_efficiency)[
                secondary_mask]

            # Update the energy state
            x['v2gChActPower'][:, t] *= x['v2gChXo'][:, t]
            x['v2gDchActPower'][:, t] *= x['v2gDchXo'][:, t]

            x['v2gEnerState'][:, t] = x['v2gEnerState'][:, t - 1] + x['v2gChActPower'][:, t] * \
                                      self.components['evs'].charge_efficiency - \
                                      x['v2gDchActPower'][:, t] / self.components['evs'].discharge_efficiency

            # Check minimum energy state
            mask = x['v2gEnerState'][:, t] < self.components['evs'].capacity_max * self.components['evs'].min_charge - \
                   x['EminRelaxEV'][:, t]
            x['v2gEnerState'][:, t][mask] = self.components['evs'].capacity_max[mask] * \
                                            self.components['evs'].min_charge[mask] - \
                                            x['EminRelaxEV'][:, t][mask]

        # Clip the values of discharging and charging to the maximum allowed
        x['v2gDchActPower'] = np.clip(x['v2gDchActPower'], np.zeros(x['v2gDchActPower'].shape),
                                      self.components['evs'].schedule_discharge)
        x['v2gChActPower'] = np.clip(x['v2gChActPower'], np.zeros(x['v2gChActPower'].shape),
                                     self.components['evs'].schedule_charge)

    def set_v2g(self, x):

        x['v2gDchActPower'] = np.zeros(x['v2gDchActPower'].shape)
        x['v2gChActPower'] = np.zeros(x['v2gChActPower'].shape)
        x['v2gEnerState'] = np.zeros(x['v2gEnerState'].shape)
        x['EminRelaxEV'] = np.zeros(x['EminRelaxEV'].shape)

        # Bound binaries
        x['v2gDchXo'] = (x['v2gDchXo'] > 0.5).astype(int)
        x['v2gChXo'] = (x['v2gChXo'] > 0.5).astype(int)

        x['v2gDchActPower'] = (self.parsed_data.vehicle['p_charge_max'] * self.parsed_data.vehicle['schedule'].T).T
        x['v2gChActPower'] = (self.parsed_data.vehicle['p_discharge_max'] * self.parsed_data.vehicle['schedule'].T).T

        for v in np.arange(self.n_v2g):
            isConnected = False
            connectedTo = 0

            for c in np.arange(self.parsed_data.vehicle['schedule_cs_usage'].shape[0]):
                isConnected = True
                if self.parsed_data.vehicle['schedule_cs_usage'][c, v, 0] > 0:
                    x['v2gDchActPower'][v, 0] = min(x['v2gDchActPower'][v, 0],
                                                    self.parsed_data.charging_station['p_charge_limit'][c, 0])
                    x['v2gChActPower'][v, 0] = min(x['v2gChActPower'][v, 0],
                                                   self.parsed_data.charging_station['p_discharge_limit'][c, 0])

                    connectedTo = c

                else:
                    x['v2gDchActPower'][v, 0] = 0
                    x['v2gChActPower'][v, 0] = 0

            mask = self.parsed_data.vehicle['schedule_departure_soc'][v, :] > 0
            temp_val = self.parsed_data.vehicle['schedule_departure_soc'][v, :][mask]

            if isConnected & (len(temp_val) > 0):
                if x['v2gEnerState'][v, 0] < temp_val[0]:
                    next_index = next((idx for idx, val in np.ndenumerate(
                        self.parsed_data.vehicle['schedule_departure_soc'][v, :])
                                       if val == temp_val[0]))[0]
                    min_tsteps = np.ceil((temp_val[0] - x['v2gEnerState'][v, 0]) / \
                                         self.parsed_data.charging_station['p_charge_max'][connectedTo]) - 1

                    if min_tsteps >= next_index:
                        x['v2gChXo'][v, 0] = 1
                        x['v2gDchXo'][v, 0] = 0

            x['v2gChActPower'][v, 0] *= x['v2gChXo'][v, 0]
            x['v2gDchActPower'][v, 0] *= x['v2gDchXo'][v, 0]

            if self.parsed_data.vehicle['schedule'][v, 0] == 0:
                x['v2gEnerState'][v, 0] = 0
            elif self.parsed_data.vehicle['schedule'][v, 0] == 1:
                x['v2gEnerState'][v, 0] = self.parsed_data.vehicle['schedule_arrival_soc'][v, 0] + \
                                            x['v2gChActPower'][v, 0] * self.parsed_data.vehicle['charge_efficiency'][v] - \
                                            x['v2gDchActPower'][v, 0] / self.parsed_data.vehicle['discharge_efficiency'][v]

            # Fix timestep dependencies
            for t in np.arange(1, x['v2gDchActPower'].shape[1]):

                isConnected = False
                connectedTo = 0

                # Check Charging Stations
                for c in np.arange(self.parsed_data.vehicle['schedule_cs_usage'].shape[0]):
                    if self.parsed_data.vehicle['schedule_cs_usage'][c, v, t] > 0:
                        x['v2gDchActPower'][v, t] = min(x['v2gDchActPower'][v, t],
                                                        self.parsed_data.charging_station['p_charge_limit'][c, t])
                        x['v2gChActPower'][v, t] = min(x['v2gChActPower'][v, t],
                                                       self.parsed_data.charging_station['p_discharge_limit'][c, t])

                        connectedTo = c
                        isConnected = True

                    else:
                        x['v2gDchXo'][v, t] = 0
                        x['v2gChXo'][v, t] = 0

                # Disable charge and discharge in the same timestep
                if x['v2gChXo'][v, t] + x['v2gDchXo'][v, t] > 1:
                    x['v2gDchXo'][v, t] = 1 - x['v2gChXo'][v, t]

                # Charge to minimum limits
                mask = self.parsed_data.vehicle['schedule_departure_soc'][v, t:] > 0
                temp_val = self.parsed_data.vehicle['schedule_departure_soc'][v, t:][mask]

                # Check if there are any requirements for EVs
                if isConnected & (len(temp_val) > 0):
                    if x['v2gEnerState'][v, t-1] < temp_val[0]:
                        next_index = next((idx for idx, val in np.ndenumerate(
                            self.parsed_data.vehicle['schedule_departure_soc'][v, t:])
                                           if val == temp_val[0]))[0]
                        min_tsteps = np.ceil((temp_val[0] - x['v2gEnerState'][v, t-1]) / \
                                             self.parsed_data.charging_station['p_charge_max'][connectedTo]) - 1

                        if min_tsteps >= next_index:
                            x['v2gChXo'][v, t] = 1
                            x['v2gDchXo'][v, t] = 0

                            if (x['v2gEnerState'][v, t-1] + x['v2gChActPower'][v, t] * \
                                self.parsed_data.vehicle['charge_efficiency'][v]) > self.parsed_data.vehicle['e_capacity_max'][v]:
                                x['v2gChActPower'][v, t] = (self.parsed_data.vehicle['e_capacity_max'][v] - x['v2gEnerState'][v, t-1]) / \
                                                           self.parsed_data.vehicle['charge_efficiency'][v]

                # Prevent overcharge
                if x['v2gEnerState'][v, t-1] == self.parsed_data.vehicle['e_capacity_max'][v]:
                    x['v2gChXo'][v, t] = 0

                # Prevent discharge below minimum
                if x['v2gEnerState'][v, t-1] == 0:
                    x['v2gDchXo'][v, t] = 0

                # Update the energy state
                x['v2gChActPower'][v, t] *= x['v2gChXo'][v, t]
                x['v2gDchActPower'][v, t] *= x['v2gDchXo'][v, t]

                # Update battery state
                if (self.parsed_data.vehicle['schedule'][v, t-1] == 1) & (self.parsed_data.vehicle['schedule'][v, t] == 1):
                    x['v2gEnerState'][v, t] = x['v2gEnerState'][v, t-1] + x['v2gChActPower'][v, t] * \
                                              self.parsed_data.vehicle['charge_efficiency'][v] - \
                                              x['v2gDchActPower'][v, t] / self.parsed_data.vehicle['discharge_efficiency'][v]

                elif (self.parsed_data.vehicle['schedule'][v, t-1] == 0) & (self.parsed_data.vehicle['schedule'][v, t] == 1):
                    x['v2gEnerState'][v, t] = self.parsed_data.vehicle['schedule_arrival_soc'][v, t] + \
                                              x['v2gChActPower'][v, t] * self.parsed_data.vehicle['charge_efficiency'][v] - \
                                              x['v2gDchActPower'][v, t] / self.parsed_data.vehicle['discharge_efficiency'][v]

                elif (self.parsed_data.vehicle['schedule'][v, t-1] == 1) & (self.parsed_data.vehicle['schedule'][v, t] == 0):
                    x['v2gEnerState'][v, t] = 0

        return

    def check_balance(self, x):

        # Calculate the values over time
        balance_gens = np.sum(x['genActPower'] - x['genExcActPower'], axis=0)
        balance_loads = np.sum(x['loadRedActPower'] + x['loadCutActPower'] + x['loadENS'] -
                               self.components['loads'].upper_bound, axis=0)
        balance_stor = np.sum(x['storDchActPower'] - x['storChActPower'], axis=0)
        balance_cs = np.sum(x['v2gDchActPower'] - x['v2gChActPower'], axis=0)

        balance_rest = balance_gens + balance_loads + balance_stor + balance_cs

        # Attribute penalties to import and exports to compensate the imbalance
        mask = balance_rest > 0
        x['pImp'][mask] = 0.0
        x['pExp'][mask] = balance_rest[mask]

        mask = balance_rest < 0
        x['pExp'][mask] = 0.0
        x['pImp'][mask] = abs(balance_rest)[mask]

        return

    def repair(self, x: dict) -> dict:
        """
        Repair a single element
        :param x: Member to repair
        :return: Repaired solution
        """
        # Check the imports and exports
        self.check_imports_exports(x)

        # Check the generators
        self.check_generators(x)

        # Check the loads
        self.check_loads(x)

        # Check the storage
        self.check_storage(x)

        # Check the EVs
        if self.set_vehicles:
            x['v2gDchActPower'] = self.__initial_variables__['v2gDchActPower']
            x['v2gChActPower'] = self.__initial_variables__['v2gChActPower']
            x['v2gEnerState'] = self.__initial_variables__['v2gEnerState']
            x['EminRelaxEV'] = self.__initial_variables__['EminRelaxEV']
            x['v2gDchXo'] = self.__initial_variables__['v2gDchXo']
            x['v2gChXo'] = self.__initial_variables__['v2gChXo']
        else:
            self.check_v2g(x)

        # Check the balance
        self.check_balance(x)

        return x
