# Pymoo repair class for the HM problem

import numpy as np
from pymoo.core.repair import Repair
from ..parsers import HMParser


class HMRepairPymoo(Repair):

    def __init__(self, data: HMParser):
        self.components = data

        # Set the initial variables to work with
        self.__initial_variables__ = {}
        self.__var_idx__ = []
        self.__var_names__ = []

        self.n_steps = data.generator['p_forecast'].shape[1]
        self.n_gen = data.generator['p_forecast'].shape[0]
        self.n_load = data.load['p_forecast'].shape[0]
        self.n_stor = data.storage['p_charge_limit'].shape[0]
        self.n_v2g = data.vehicle['p_charge_max'].shape[0]

        self._initialize_values()

        self.storCapCost = [0.05250, 0.10500, 0.01575]
        self.v2gCapCost = [0.042, 0.063, 0.042, 0.042, 0.063]

        super().__init__()

    def _initialize_values(self):

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
        x['pImp'] = np.clip(x['pImp'], 0, self.components.peers['import_contracted_p_max'][0, :])
        x['pExp'] = np.clip(x['pExp'], 0, self.components.peers['export_contracted_p_max'][0, :])
        return

    def check_generators(self, x):
        # Clip the values
        x['genActPower'] = np.clip(x['genActPower'], np.zeros(x['genActPower'].shape),
                                   (np.ones(x['genActPower'].shape).transpose() * self.components.generator[
                                       'p_max']).transpose())

        # Set the excess power to 0 and fix from there
        x['genExcActPower'] = np.zeros(x['genExcActPower'].shape)

        # Generator types
        # Type 1 (non-renewable)
        mask = self.components.generator['type_generator'] == np.ones(self.components.generator['type_generator'].shape)
        x['genActPower'][mask] = (self.components.generator['p_forecast'] * x['genXo'])[mask]

        # Type 2 (renewable)
        mask = self.components.generator['type_generator'] == 2 * \
               np.ones(self.components.generator['type_generator'].shape)
        x['genExcActPower'][mask] = (self.components.generator['p_forecast'] - x['genActPower'])[mask]

        x['genExcActPower'] = np.clip(x['genExcActPower'], np.zeros(x['genActPower'].shape),
                                      (np.ones(x['genActPower'].shape).transpose() * self.components.generator[
                                          'p_max']).transpose())
        return

    def check_loads(self, x):
        # Assign the binary variable
        x['loadXo'] = (x['loadXo'] > 0.5).astype(int)

        # Load reduction
        x['loadRedActPower'] = np.clip(x['loadRedActPower'], 0, self.components.load['p_reduce'])

        # Load curtailing
        x['loadCutActPower'] = self.components.load['p_forecast'] * x['loadXo']

        # Load ENS
        temp_vals = self.components.load['p_forecast'] - x['loadRedActPower'] - x['loadCutActPower']
        x['loadENS'] = np.clip(temp_vals, 0, self.components.load['p_forecast'])
        return

    def check_storage(self, x):
        # Assign the binary variables
        x['storDchXo'] = (x['storDchXo'] > 0.5).astype(int)
        x['storChXo'] = (x['storChXo'] > 0.5).astype(int)

        x['storDchXo'] = x['storDchXo'] * (1 - x['storChXo'])

        # Value clipping
        x['storDchActPower'] = np.clip(x['storDchActPower'], np.zeros(x['storDchActPower'].shape),
                                       (np.ones(x['storDchActPower'].shape).transpose() * \
                                        self.components.storage['p_discharge_max']).transpose())
        x['storChActPower'] = np.clip(x['storChActPower'], np.zeros(x['storChActPower'].shape),
                                      (np.ones(x['storChActPower'].shape).transpose() * self.components.storage[
                                          'p_charge_max']).transpose())

        # Initial state of charge
        x['storEnerState'][:, 0] = self.components.storage['energy_capacity'] * (self.components.storage['initial_state']) + \
                                   x['storChActPower'][:, 0] * self.components.storage['charge_efficiency'] - \
                                   x['storDchActPower'][:, 0] / self.components.storage['discharge_efficiency']

        # Initialize the iterator and range (we already did the initial timestep!)
        t: int = 1
        t_range = range(1, self.n_steps)

        # Fix the timestep dependencies
        for t in t_range:
            # Check if charging
            mask = x['storChXo'][:, t] > np.zeros(x['storChXo'][:, t].shape)
            charged = x['storChActPower'][:, t] * (1 - self.components.storage['charge_efficiency'])

            # Prevent over charging
            secondary_mask = (x['storEnerState'][:, t - 1] + charged) > self.components.storage['energy_capacity']
            x['storChActPower'][:, t][secondary_mask] = \
            ((self.components.storage['energy_capacity'] - x['storEnerState'][:, t - 1]) / \
             (self.components.storage['charge_efficiency']))[secondary_mask]

            # Check if discharging
            mask = x['storDchXo'][:, t] > np.zeros(x['storDchXo'][:, t].shape)
            discharged = x['storDchActPower'][:, t] / self.components.storage['discharge_efficiency']
            secondary_mask = (x['storEnerState'][:, t - 1] - discharged) < 0
            x['storDchActPower'][:, t][secondary_mask] = (x['storEnerState'][:, t - 1] * \
                                                          self.components.storage['discharge_efficiency'])[secondary_mask]

            # Update the energy state
            x['storChActPower'][:, t] *= x['storChXo'][:, t]
            x['storDchActPower'][:, t] *= x['storDchXo'][:, t]

            x['storEnerState'][:, t] = x['storEnerState'][:, t - 1] + x['storChActPower'][:, t] * self.components.storage[
                'charge_efficiency'] - \
                                       x['storDchActPower'][:, t] / self.components.storage['discharge_efficiency']

            # Check minimum energy state
            mask = x['storEnerState'][:, t] < self.components.storage['energy_capacity'] * self.components.storage[
                'energy_min_percentage'] - \
                   x['EminRelaxStor'][:, t]
            x['storEnerState'][:, t][mask] = self.components.storage['energy_capacity'][mask] * \
                                             self.components.storage['energy_min_percentage'][mask] - \
                                             x['EminRelaxStor'][:, t][mask]

        x['storDchActPower'] = np.clip(x['storDchActPower'], np.zeros(x['storDchActPower'].shape),
                                       (np.ones(x['storDchActPower'].shape).transpose() * \
                                        self.components.storage['p_discharge_max']).transpose())
        x['storChActPower'] = np.clip(x['storChActPower'], np.zeros(x['storChActPower'].shape),
                                      (np.ones(x['storChActPower'].shape).transpose() * self.components.storage[
                                          'p_charge_max']).transpose())

        return

    def check_v2g(self, x):

        # Placeholders for the energy state and relaxation variable
        x['v2gEnerState'] = np.zeros(x['v2gEnerState'].shape)
        x['EminRelaxEV'] = np.zeros(x['EminRelaxEV'].shape)

        # Bound binaries
        x['v2gDchXo'] = (x['v2gDchXo'] > 0.5).astype(int)
        x['v2gChXo'] = (x['v2gChXo'] > 0.5).astype(int)

        x['v2gDchXo'] = x['v2gDchXo'] * (1 - x['v2gChXo'])

        # Preallocate range
        t_range = range(1, self.n_steps)

        # Clip the values of discharging and charging to the maximum allowed
        x['v2gDchActPower'] = np.clip(x['v2gDchActPower'], 0, self.components.vehicle['schedule_discharge'])
        x['v2gChActPower'] = np.clip(x['v2gChActPower'], 0, self.components.vehicle['schedule_charge'])

        # Set initial EV state
        x['v2gEnerState'][:, 0] = self.components.vehicle['e_capacity_max'] * 0.8

        # Fix the timestep dependencies
        for t in t_range:
            # Check if charging
            mask = x['v2gChXo'][:, t] > np.zeros(x['v2gChXo'][:, t].shape)
            charged = x['v2gChActPower'][:, t] * (1 - self.components.vehicle['charge_efficiency'])

            # Prevent over charging
            secondary_mask = (x['v2gEnerState'][:, t - 1] + charged) > self.components.vehicle['e_capacity_max']
            x['v2gChActPower'][:, t][secondary_mask] = ((self.components.vehicle['e_capacity_max'] - \
                                                         x['v2gEnerState'][:, t - 1]) / \
                                                        (self.components.vehicle['charge_efficiency']))[secondary_mask]

            # Check if discharging
            mask = x['v2gDchXo'][:, t] > np.zeros(x['v2gDchXo'][:, t].shape)
            discharged = x['v2gDchActPower'][:, t] / self.components.vehicle['discharge_efficiency']
            secondary_mask = (x['v2gEnerState'][:, t - 1] - discharged) < 0
            x['v2gDchActPower'][:, t][secondary_mask] = (x['v2gEnerState'][:, t - 1] * \
                                                         self.components.vehicle['discharge_efficiency'])[secondary_mask]

            # Update the energy state
            x['v2gChActPower'][:, t] *= x['v2gChXo'][:, t]
            x['v2gDchActPower'][:, t] *= x['v2gDchXo'][:, t]

            x['v2gEnerState'][:, t] = x['v2gEnerState'][:, t - 1] + x['v2gChActPower'][:, t] * \
                                      self.components.vehicle['charge_efficiency'] - \
                                      x['v2gDchActPower'][:, t] / self.components.vehicle['discharge_efficiency']

            # Check minimum energy state
            mask = x['v2gEnerState'][:, t] < self.components.vehicle['e_capacity_max'] * \
                   self.components.vehicle['min_technical_soc'] - \
                   x['EminRelaxEV'][:, t]
            x['v2gEnerState'][:, t][mask] = self.components.vehicle['e_capacity_max'][mask] * \
                                            self.components.vehicle['min_technical_soc'][mask] - \
                                            x['EminRelaxEV'][:, t][mask]

        # Clip the values of discharging and charging to the maximum allowed
        x['v2gDchActPower'] = np.clip(x['v2gDchActPower'], 0, self.components.vehicle['schedule_discharge'])
        x['v2gChActPower'] = np.clip(x['v2gChActPower'], 0, self.components.vehicle['schedule_charge'])

    def check_balance(self, x):

        # Create the iterators and ranges
        t_range = range(self.n_steps)

        g_range = range(self.n_gen)
        balance_gens = np.zeros(self.n_steps)

        l_range = range(self.n_load)
        balance_loads = np.zeros(self.n_steps)

        s_range = range(self.n_stor)
        balance_stor = np.zeros(self.n_steps)

        v_range = range(self.n_v2g)
        balance_cs = np.zeros(self.n_steps)  # note: balance of the EVs is made through the charging station

        # Calculate the values over time
        for t in t_range:
            balance_gens[t] = np.sum([x['genActPower'][g, t] - x['genExcActPower'][g, t] for g in g_range])

            balance_loads[t] = np.sum([x['loadRedActPower'][l, t] + \
                                       x['loadCutActPower'][l, t] + \
                                       x['loadENS'][l, t] + \
                                       -self.components.load['p_forecast'][l, t]
                                       for l in l_range])

            balance_stor[t] = np.sum([x['storDchActPower'][s, t] - x['storChActPower'][s, t] for s in s_range])

            balance_cs[t] = np.sum([x['v2gDchActPower'][v, t] - x['v2gChActPower'][v, t] for v in v_range])

        balance_rest = balance_gens + balance_loads + balance_stor + balance_cs

        # Attribute penalties to import and exports to compensate the imbalance
        mask = balance_rest > 0
        x['pImp'][mask] *= 0.0
        x['pExp'][mask] = balance_rest[mask]

        mask = balance_rest < 0
        x['pExp'][mask] *= 0.0
        x['pImp'][mask] = abs(balance_rest)[mask]

        return

    def _do(self, problem, Z, **kwargs):
        for z in np.arange(len(Z)):
            temp_z = self.decode(Z[z])
            self.check_imports_exports(temp_z)
            self.check_generators(temp_z)
            self.check_loads(temp_z)
            self.check_storage(temp_z)
            self.check_v2g(temp_z)

            self.check_balance(temp_z)

            temp_z = self.encode(temp_z)
            Z[z] = temp_z

        return Z

    @staticmethod
    def encode(x):
        result_encoded = np.concatenate([x[idx].ravel() for idx in x.keys()])
        return result_encoded

    def decode(self, x):
        result_decoded = {}
        current_index = 0

        for idx in range(len(self.__var_names__)):
            result_index = current_index + self.__var_idx__[idx]
            result_decoded[self.__var_names__[idx]] = np.reshape(x[current_index:result_index],
                                                                 self.__initial_variables__[
                                                                     self.__var_names__[idx]].shape)

            current_index = result_index

        return result_decoded
