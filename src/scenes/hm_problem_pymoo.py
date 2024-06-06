# Problem definition for the optimization problem

import numpy as np
from pymoo.core.problem import Problem
from ..parsers import HMParser
from ..resources import BaseResource


class HMProblemPymoo(Problem):

    def __init__(self, data):

        # Set the components
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

        # Set the lower and upper bounds
        self.xl_dict = self._initialize_xl()
        self.xu_dict = self._initialize_xu(data)
        self.xl = self.encode(self.xl_dict)
        self.xu = self.encode(self.xu_dict)

        # Storage and V2G battery parameters
        # To calculate these parameters, we use the following formula
        # Provided in utils/battery/parameter_calculation.py
        # Each element of the list is for a different battery
        self.storCapCost = [0.05250, 0.10500, 0.01575]
        self.storCapCost = np.tile(self.storCapCost, self.n_stor)
        self.v2gCapCost = [0.042, 0.063, 0.042, 0.042, 0.063]
        self.v2gCapCost = np.tile(self.v2gCapCost, self.n_v2g)

        # Placeholder for the objective function
        self.objFn = 0.0

        # Call the super class
        super().__init__(n_var=len(self.xl), n_obj=1, n_constr=0,
                         xl=self.xl, xu=self.xu, vtype=float)

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

    def _initialize_xl(self):

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

        return temp_vars

    def _initialize_xu(self, x):

        temp_vars = {'genActPower': x.generator['p_forecast'],
                     'genExcActPower': x.generator['p_forecast'],
                     'pImp': x.peers['import_contracted_p_max'][0, :],
                     'pExp': x.peers['export_contracted_p_max'][0, :],
                     'loadRedActPower': x.load['p_reduce'],
                     'loadCutActPower': x.load['p_forecast'],
                     'loadENS': x.load['p_forecast'],
                     'storDchActPower': x.storage['p_discharge_limit'],
                     'storChActPower': x.storage['p_charge_limit'],
                     'EminRelaxStor': x.storage['p_charge_limit'],
                     'storEnerState': (x.storage['energy_capacity'] * np.ones((self.n_steps,
                                                                               x.storage['energy_capacity'].shape[
                                                                                   0]))).transpose(),
                     'v2gDchActPower': x.vehicle['schedule_discharge'],
                     'v2gChActPower': x.vehicle['schedule_charge'],
                     'EminRelaxEV': (x.vehicle['e_capacity_max'] * np.ones((self.n_steps,
                                                                            x.vehicle['e_capacity_max'].shape[
                                                                                0]))).transpose(),
                     'v2gEnerState': (x.vehicle['e_capacity_max'] * np.ones((self.n_steps,
                                                                             x.vehicle['e_capacity_max'].shape[
                                                                                 0]))).transpose(),
                     'genXo': np.ones(x.generator['p_forecast'].shape),
                     'loadXo': np.ones(x.load['p_forecast'].shape),
                     'storDchXo': np.ones((self.n_stor, self.n_steps)),
                     'storChXo': np.ones((self.n_stor, self.n_steps)),
                     'v2gDchXo': np.ones((self.n_v2g, self.n_steps)),
                     'v2gChXo': np.ones((self.n_v2g, self.n_steps))}

        return temp_vars

    def objective_function(self, x):
        # Set the iterators and ranges

        i: int = 0
        t: int = 0
        g: int = 0
        l: int = 0
        s: int = 0
        v: int = 0

        # Set the ranges
        t_range = range(self.n_steps)
        gen_range = range(self.n_gen)
        load_range = range(self.n_load)
        stor_range = range(self.n_stor)
        v2g_range = range(self.n_v2g)

        # Assign penalties for import/export
        balance_penalty = 0.0
        for t in t_range:
            if x['pImp'][t] > self.components.peers['import_contracted_p_max'][0, t]:
                balance_penalty += 100000

            if x['pExp'][t] > self.components.peers['export_contracted_p_max'][0, t]:
                balance_penalty += 100000

        # Calculate the individual component costs
        temp_gens = np.sum(x['genActPower'] * self.components.generator['cost_parameter_b'] +
                           x['genExcActPower'] * self.components.generator['cost_nde'])

        temp_loads = np.sum(x['loadRedActPower'] * self.components.load['cost_reduce'] +
                            x['loadCutActPower'] * self.components.load['cost_cut'] +
                            x['loadENS'] * self.components.load['cost_ens'])

        temp_stor: float = sum([self.storCapCost[s] * (x['storEnerState'][s, t] /
                                                       self.components.storage['energy_capacity'][s] - 0.63) ** 2 + \
                                x['storDchActPower'][s, t] * self.components.storage['discharge_price'][s, t] + \
                                x['storChActPower'][s, t] * self.components.storage['charge_price'][s, t] + \
                                (6.5e-3) / self.components.storage['energy_capacity'][s] * x['storChActPower'][
                                    s, t] ** 2
                                for t in t_range for s in stor_range])

        temp_v2g: float = sum([self.v2gCapCost[v] * (x['v2gEnerState'][v, t] /
                                                     self.components.vehicle['e_capacity_max'][v] - 0.63) ** 2 + \
                               x['v2gDchActPower'][v, t] * self.components.vehicle['discharge_price'][v, 0] + \
                               x['v2gChActPower'][v, t] * self.components.vehicle['charge_price'][v, 0] + \
                               (6.5e-3) / self.components.vehicle['e_capacity_max'][v] * x['v2gChActPower'][v, t] ** 2
                               for t in t_range for v in v2g_range])

        temp_rest: float = sum([x['pImp'][t] * self.components.peers['buy_price'][0, t] + \
                                x['pExp'][t] * self.components.peers['sell_price'][0, t]
                                for t in t_range])

        self.objFn = temp_gens + temp_loads + temp_stor + temp_v2g + temp_rest + balance_penalty

        return

    def _decode(self, x):
        result_decoded = {}
        current_index = 0

        for idx in range(len(self.__var_names__)):
            result_index = current_index + self.__var_idx__[idx]
            result_decoded[self.__var_names__[idx]] = np.reshape(x[current_index:result_index],
                                                                 self.__initial_variables__[
                                                                     self.__var_names__[idx]].shape)

            current_index = result_index

        return result_decoded

    @staticmethod
    def encode(x):
        result_encoded = np.concatenate([x[idx].ravel() for idx in x.keys()])
        return result_encoded

    def _evaluate(self, x, out, *args, **kwargs):
        objective_function = []
        for temp in range(x.shape[0]):
            temp_solution = self._decode(x[temp])
            self.objective_function(temp_solution)
            objective_function.append(self.objFn)

        out['F'] = objective_function

    def decode(self, x: np.ndarray) -> dict:
        """
        Decodes the solution to the resources
        :param x: Encoded solution
        :return: Dictionary of resources
        """

        # Decode the solution
        decoded = self._decode(x)

        # Decode the generators
        genActPower = [BaseResource(name='genActPower_{:02d}'.format(g),
                                    value=decoded['genActPower'][g, :],
                                    lower_bound=self.xl_dict['genActPower'][g, :],
                                    upper_bound=self.xu_dict['genActPower'][g, :],
                                    cost=self.components.generator['cost_parameter_b'][g])
                       for g in range(self.n_gen)]

        genExcActPower = [BaseResource(name='genExcActPower_{:02d}'.format(g),
                                       value=decoded['genExcActPower'][g, :],
                                       lower_bound=self.xl_dict['genExcActPower'][g, :],
                                       upper_bound=self.xu_dict['genExcActPower'][g, :],
                                       cost=self.components.generator['cost_nde'][g])
                          for g in range(self.n_gen)]

        # Decode the import/export
        pImp = BaseResource(name='pImp',
                            value=decoded['pImp'],
                            lower_bound=self.xl_dict['pImp'],
                            upper_bound=self.xu_dict['pImp'],
                            cost=self.components.peers['buy_price'][0])

        pExp = BaseResource(name='pExp',
                            value=decoded['pExp'],
                            lower_bound=self.xl_dict['pExp'],
                            upper_bound=self.xu_dict['pExp'],
                            cost=self.components.peers['sell_price'][0])

        # Decode the loads
        loadRedActPower = [BaseResource(name='loadRedActPower_{:02d}'.format(l),
                                        value=decoded['loadRedActPower'][l, :],
                                        lower_bound=self.xl_dict['loadRedActPower'][l, :],
                                        upper_bound=self.xu_dict['loadRedActPower'][l, :],
                                        cost=self.components.load['cost_reduce'][l])
                           for l in range(self.n_load)]

        loadCutActPower = [BaseResource(name='loadCutActPower_{:02d}'.format(l),
                                        value=decoded['loadCutActPower'][l, :],
                                        lower_bound=self.xl_dict['loadCutActPower'][l, :],
                                        upper_bound=self.xu_dict['loadCutActPower'][l, :],
                                        cost=self.components.load['cost_cut'][l])
                           for l in range(self.n_load)]

        loadENS = [BaseResource(name='loadENS_{:02d}'.format(l),
                                value=decoded['loadENS'][l, :],
                                lower_bound=self.xl_dict['loadENS'][l, :],
                                upper_bound=self.xu_dict['loadENS'][l, :],
                                cost=self.components.load['cost_ens'][l])
                   for l in range(self.n_load)]

        loadActPower = [BaseResource(name='loadActPower_{:02d}'.format(l),
                                     value=self.components.load['p_forecast'][l, :],
                                     lower_bound=self.xl_dict['loadCutActPower'][l, :],
                                     upper_bound=self.xu_dict['loadCutActPower'][l, :],
                                     cost=self.components.load['cost_cut'][l])
                        for l in range(self.n_load)]

        # Decode the storage
        storDchActPower = [BaseResource(name='storDchActPower_{:02d}'.format(s),
                                        value=decoded['storDchActPower'][s, :],
                                        lower_bound=self.xl_dict['storDchActPower'][s, :],
                                        upper_bound=self.xu_dict['storDchActPower'][s, :],
                                        cost=self.components.storage['discharge_price'][s])
                           for s in range(self.n_stor)]

        storChActPower = [BaseResource(name='storChActPower_{:02d}'.format(s),
                                       value=decoded['storChActPower'][s, :],
                                       lower_bound=self.xl_dict['storChActPower'][s, :],
                                       upper_bound=self.xu_dict['storChActPower'][s, :],
                                       cost=self.components.storage['charge_price'][s])
                          for s in range(self.n_stor)]

        storEnerState = [BaseResource(name='storEnerState_{:02d}'.format(s),
                                      value=decoded['storEnerState'][s, :],
                                      lower_bound=self.xl_dict['storEnerState'][s, :],
                                      upper_bound=self.xu_dict['storEnerState'][s, :],
                                      cost=np.array([0.0 for _ in range(self.n_steps)]))
                         for s in range(self.n_stor)]

        v2gDchActPower = [BaseResource(name='v2gDchActPower_{:02d}'.format(v),
                                       value=decoded['v2gDchActPower'][v, :],
                                       lower_bound=self.xl_dict['v2gDchActPower'][v, :],
                                       upper_bound=self.xu_dict['v2gDchActPower'][v, :],
                                       cost=self.components.vehicle['discharge_price'][v, 0])
                          for v in range(self.n_v2g)]

        v2gChActPower = [BaseResource(name='v2gChActPower_{:02d}'.format(v),
                                      value=decoded['v2gChActPower'][v, :],
                                      lower_bound=self.xl_dict['v2gChActPower'][v, :],
                                      upper_bound=self.xu_dict['v2gChActPower'][v, :],
                                      cost=self.components.vehicle['charge_price'][v, 0])
                         for v in range(self.n_v2g)]

        v2gEnerState = [BaseResource(name='v2gEnerState_{:02d}'.format(v),
                                     value=decoded['v2gEnerState'][v, :],
                                     lower_bound=self.xl_dict['v2gEnerState'][v, :],
                                     upper_bound=self.xu_dict['v2gEnerState'][v, :],
                                     cost=np.array([0.0 for _ in range(self.n_steps)]))
                        for v in range(self.n_v2g)]

        # Add everything to the dictionary
        result = {'genActPower': genActPower,
                  'genExcActPower': genExcActPower,
                  'pImp': pImp,
                  'pExp': pExp,
                  'loadRedActPower': loadRedActPower,
                  'loadCutActPower': loadCutActPower,
                  'loadENS': loadENS,
                  'loadActPower': loadActPower,
                  'storDchActPower': storDchActPower,
                  'storChActPower': storChActPower,
                  'storEnerState': storEnerState,
                  'v2gDchActPower': v2gDchActPower,
                  'v2gChActPower': v2gChActPower,
                  'v2gEnerState': v2gEnerState}

        return result
