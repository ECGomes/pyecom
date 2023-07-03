# Contains the EnergyCommunity class, which is a custom heuristic data class

import numpy as np
import cython


class HeuristicData(object):
    def __init__(self):
        return

    def objectiveFunction(self):
        return

    def doIteration(self):
        return

    def newIteration(self, encoded_vals):
        return


class EnergyCommunity(HeuristicData):

    def __init__(self, parameters, variables):
        # Set initial values
        self.__initial_variables__ = variables
        self.__var_idx__ = [variables[v].ravel().shape[0] for v in variables.keys()]
        self.__var_names__ = list(variables.keys())

        self.parameterAssign(parameters)
        self.variableAssign(variables)

        self.encoded = None
        self.decoded = None

        self.objFn = 0

        self.checkV2G()
        self.checkCS()
        return

    def parameterAssign(self, parameters):
        self.genLimit = parameters['genLimit']
        self.genInfo = parameters['genInfo']
        self.pImpMax = parameters['pImpMax']
        self.pExpMax = parameters['pExpMax']
        self.loadLimit = parameters['loadLimit']
        self.loadActPower = parameters['loadActPower']
        self.storLimit = parameters['storLimit']
        self.storInfo = parameters['storInfo']
        self.v2gLimit = parameters['v2gLimit']
        self.v2gInfo = parameters['v2gInfo']
        self.csLimit = parameters['csLimit']
        self.csInfo = parameters['csInfo']
        self.EV_CS_Info = parameters['EV_CS_Info']
        self.buyPrice = parameters['buyPrice']
        self.sellPrice = parameters['sellPrice']
        self.t = parameters['t']
        self.gen = parameters['gen']
        self.load = parameters['load']
        self.stor = parameters['stor']
        self.v2g = parameters['v2g']
        self.cs = parameters['cs']
        # self.storPieceSegments = parameters['storPieceSegments']
        self.storCapCost = parameters['storCapCost']
        self.storCapCost = np.reshape(self.storCapCost, (self.storCapCost.shape[0], 1))
        # self.storCyclesMult = parameters['storCyclesMult']
        self.v2gCapCost = parameters['v2gCapCost']
        self.v2gCapCost = np.reshape(self.v2gCapCost, (self.v2gCapCost.shape[0], 1))
        # self.v2gCyclesMult = parameters['v2gCyclesMult']

        # Stor and V2G aux variables
        self.aux_storInfo = np.reshape(self.storInfo[:, 5], (self.storInfo[:, 5].shape[0], 1))
        self.aux_v2gInfo = np.reshape(self.v2gInfo[:, 4], (self.v2gInfo[:, 4].shape[0], 1))

        return

    def variableAssign(self, variables):
        self.genActPower = variables['genActPower']
        self.genExcActPower = variables['genExcActPower']
        self.pImp = variables['pImp']
        self.pExp = variables['pExp']
        self.loadRedActPower = variables['loadRedActPower']
        self.loadENS = variables['loadENS']
        self.storDchActPower = variables['storDchActPower']
        self.storChActPower = variables['storChActPower']
        self.EminRelaxStor = variables['EminRelaxStor']
        self.genXo = variables['genXo']
        self.loadXo = variables['loadXo']
        self.storDchXo = variables['storDchXo']
        self.storChXo = variables['storChXo']
        self.v2gDchXo = variables['v2gDchXo']
        self.v2gChXo = variables['v2gChXo']

        self.loadCutActPower = np.zeros((len(self.load),
                                         len(self.t)))

        self.storEnerState = np.zeros((len(self.stor),
                                       len(self.t)))

        return

    def objectiveFunction(self):

        # Define t and t_range as cython int and int array
        t: cython.int
        t_range: cython.int[self.t.shape[0]] = range(self.t.shape[0])

        # Get the balance of the system
        balance_gens = np.sum(self.genActPower - self.genExcActPower, axis=0)
        balance_loads = np.sum(self.loadActPower - self.loadRedActPower - self.loadCutActPower - self.loadENS, axis=0)
        balance_stor = np.sum(self.storChActPower - self.storDchActPower, axis=0)
        balance_v2g = np.sum(self.v2gChActPower - self.v2gDchActPower, axis=0)
        balance_rest = (balance_gens - balance_loads - balance_stor - balance_v2g).ravel()

        # Balances and import/export penalties
        mask = balance_rest > 0
        self.pImp[mask] *= 0
        self.pExp[mask] = balance_rest[mask]

        mask = balance_rest < 0
        self.pExp[mask] *= 0
        self.pImp[mask] = abs(balance_rest)[mask]

        # Attribute penalties for import/export
        balance_penalty = np.sum(
            np.array([100000 for t in t_range if self.pImp[t] > self.pImpMax[t] or self.pExp[t] > self.pExpMax[t]]))

        # Get the cost values for each component of the system
        temp_gens = np.sum(self.genActPower * self.genLimit[:, :, 2] + self.genExcActPower * self.genLimit[:, :, 4])

        temp_loads = np.sum(self.loadRedActPower * self.loadLimit[:, :, 6] + self.loadCutActPower * self.loadLimit[:, :,
                                                                                                    7] + self.loadENS * self.loadLimit[
                                                                                                                        :,
                                                                                                                        :,
                                                                                                                        9])

        temp_rest = np.sum(self.pImp * self.buyPrice - self.pExp * self.sellPrice)

        temp_storDeg = np.sum(self.storCapCost * (self.storEnerState / self.aux_storInfo - 0.63) ** 2 +
                              self.storDchActPower * self.storLimit[:, :, 3] +
                              self.storChActPower * self.storLimit[:, :, 2] +
                              (6.5e-3) / self.aux_storInfo * self.storChActPower ** 2)

        temp_v2gDeg = np.sum(self.v2gCapCost * (self.v2gEnerState / self.aux_v2gInfo - 0.63) ** 2 +
                             self.v2gDchActPower * self.v2gLimit[:, :, 6] +
                             self.v2gChActPower * self.v2gLimit[:, :, 5] +
                             (6.5e-3) / self.aux_v2gInfo * self.v2gChActPower ** 2)

        # Cost oriented objective function
        self.objFn = temp_gens + temp_loads + temp_rest + balance_penalty + temp_storDeg + temp_v2gDeg  # temp_stor + temp_v2g

        # Renewable Energy

        if self.objFn < 0:
            print('Negative objective function value: ', self.objFn)
            print('Generators: ', temp_gens)
            print('Loads: ', temp_loads)
            print('Rest: ', temp_rest)
            print('Storages: ', temp_storDeg)
            print('V2G: ', temp_v2gDeg)
            print('Balance penalty: ', balance_penalty)
            print('Balance of the system: ', balance_rest)
            print('Balance of the generators: ', balance_gens)
            print('Balance of the loads: ', balance_loads)
            print('Balance of the storages: ', balance_stor)
            print('Balance of the V2G: ', balance_v2g)
            print('Import: ', self.pImp)
            print('Export: ', self.pExp)

            raise ValueError('Negative objective function value')

        return

    def _objectiveFunctionRenewables(self):

        t_range: cython.int[self.t.shape[0]] = range(self.t.shape[0])

        # Get the balance of the system
        balance_gens = np.sum(self.genActPower - self.genExcActPower, axis=0)
        balance_loads = np.sum(self.loadActPower - self.loadRedActPower - self.loadCutActPower - self.loadENS, axis=0)
        balance_stor = np.sum(self.storChActPower - self.storDchActPower, axis=0)
        balance_v2g = np.sum(self.v2gChActPower - self.v2gDchActPower, axis=0)
        balance_rest = (balance_gens - balance_loads - balance_stor - balance_v2g).ravel()

        # Balances and import/export penalties
        mask = balance_rest > 0
        self.pImp[mask] *= 0
        self.pExp[mask] = balance_rest[mask]

        mask = balance_rest < 0
        self.pExp[mask] *= 0
        self.pImp[mask] = abs(balance_rest)[mask]

        # Attribute penalties for import/export
        balance_penalty = 0
        balance_penalty = sum(
            100000 for t in t_range if self.pImp[t] > self.pImpMax[t] or self.pExp[t] > self.pExpMax[t])

        cons = np.sum(self.loadActPower) + np.sum(self.csActPower) + np.sum(storChActPower) - np.sum(
            storDchActPower) - np.sum(self.v2gDchActPower)

        # Generator types - type 2
        mask = self.genInfo[:, 1] == 2 * np.ones(self.genInfo[:, 4].shape)
        gen_res = np.sum(self.genActPower[mask])

        self.objFn = balance_penalty + np.sum(self.loadENS) * 1000 + (cons - gen_res) ** 2

    def doIteration(self):
        self.checkImpExp()
        self.checkGens()
        self.checkLoads()
        self.checkStor()

        self._objectiveFunctionRenewables()

        self.encoded = self.encode()
        return

    def newIteration(self, encoded_vals):
        self.decode(encoded_vals)
        self.doIteration()
        return

    def checkImpExp(self):
        # Imports and exports
        self.pImp = np.clip(self.pImp, 0, self.pImpMax)
        self.pExp = np.clip(self.pExp, 0, self.pExpMax)

        return

    def checkGens(self):
        # Generator binary variable
        self.genXo = (self.genXo > 0.5).astype(int)

        # Generation bounds
        self.genActPower = np.clip(self.genActPower, 0, self.genLimit[:, :, 0])
        self.genExcActPower = np.clip(self.genExcActPower, 0, self.genLimit[:, :, 0])

        # Generator types - type 1
        mask = self.genInfo[:, 1] == np.ones(self.genInfo[:, 4].shape)
        self.genActPower[mask] = (self.genLimit[:, :, 0] * self.genXo)[mask]

        # Generator types - type 2
        mask = self.genInfo[:, 1] == 2 * np.ones(self.genInfo[:, 4].shape)
        self.genExcActPower[mask] = self.genLimit[:, :, 0][mask] - self.genActPower[mask]

        return

    def checkLoads(self):

        # Bound the values to either 0 or 1
        self.loadXo = (self.loadXo > 0.5).astype(int)

        # RedAct value checks
        self.loadRedActPower = np.clip(self.loadRedActPower, 0, self.loadLimit[:, :, 2])

        # loadCut value checks
        self.loadCutActPower = np.multiply(self.loadLimit[:, :, 3], self.loadXo)

        # loadENS
        temp_ens = self.loadActPower - self.loadRedActPower - self.loadCutActPower
        self.loadENS = np.clip(self.loadENS, 0, self.loadActPower)

        return

    def checkStor(self):

        # Binary variables bound
        self.storChXo = (self.storChXo > 0.5).astype(int)
        self.storDchXo = (self.storDchXo > 0.5).astype(int)

        # Discharge and charge value checks
        self.storDchActPower = np.clip(self.storDchActPower, 0, self.storLimit[:, :, 1])
        self.storChActPower = np.clip(self.storChActPower, 0, self.storLimit[:, :, 0])

        # Charge and discharge efficiencies
        charge_eff = self.storInfo[:, 7] * 0.01
        discharge_eff = self.storInfo[:, 8] * 0.01

        # Initial stor SoC
        self.storEnerState[:, 0] = self.storInfo[:, 5] * (self.storInfo[:, 9] / 100) + self.storChActPower[:,
                                                                                       0] * charge_eff - self.storDchActPower[
                                                                                                         :,
                                                                                                         0] / discharge_eff

        # Cython range
        t_range: cython.int[len(self.t) - 1] = range(1, len(self.t))

        # Expand the efficiencies to an array of the same size as the time array
        charge_eff = np.tile(charge_eff, (len(self.t), 1)).T
        discharge_eff = np.tile(discharge_eff, (len(self.t), 1)).T

        # Precalculate the charge and discharge values
        charged = np.multiply(self.storChActPower, charge_eff)
        discharged = np.multiply(self.storDchActPower, discharge_eff)

        # Fix the timestep dependencies
        for t in t_range:
            # Prevent charging beyond limit
            mask = (self.storEnerState[:, t - 1] + charged[:, t]) > self.storInfo[:, 5]
            self.storChActPower[:, t][mask] = ((self.storInfo[:, 5] - self.storEnerState[:, t - 1]) / charge_eff[:, t])[
                mask]
            self.storChActPower[:, t] = np.clip(self.storChActPower[:, t], 0, self.storInfo[:, 5])

            # Check if discharging is allowed
            mask = (self.storEnerState[:, t - 1] - discharged[:, t]) < 0
            self.storDchActPower[:, t][mask] = (self.storEnerState[:, t - 1] * discharge_eff[:, t])[mask]
            self.storDchActPower[:, t] = np.clip(self.storDchActPower[:, t], 0, self.storInfo[:, 5])

            # Update charge, discharge and SoC
            self.storChActPower[:, t] = np.multiply(self.storChActPower[:, t], self.storChXo[:, t])
            self.storDchActPower[:, t] = np.multiply(self.storDchActPower[:, t], self.storDchXo[:, t])
            self.storEnerState[:, t] = self.storEnerState[:, t - 1] + (self.storChActPower[:, t] * charge_eff[:, t]) - (
                        self.storDchActPower[:, t] / discharge_eff[:, t])

            mask = self.storEnerState[:, t] < self.storInfo[:, 6] * (
                        self.storInfo[:, 5] * 0.01)  # - self.EminRelaxStor[:, t]
            self.storEnerState[:, t][mask] = (self.storInfo[:, 6] * (self.storInfo[:, 5] * 0.01))[
                mask]  # - self.EminRelaxStor[:, t])[mask]

        return

    def checkV2G(self):
        # Binary variables bounding
        self.v2gDchActPower = np.zeros((len(self.v2g),
                                        len(self.t)))

        self.v2gChActPower = np.zeros((len(self.v2g),
                                       len(self.t)))

        self.v2gEnerState = np.zeros((len(self.v2g),
                                      len(self.t)))

        self.EminRelaxEv = np.zeros((len(self.v2g),
                                     len(self.t)))

        # Bound binary variables
        self.v2gChXo = (self.v2gChXo > 0.5).astype(int)
        self.v2gDchXo = (self.v2gDchXo > 0.5).astype(int)

        mask = None

        v: cython.int
        c: cython.int

        self.v2gDchActPower = self.v2gLimit[:, :, 4] * self.v2gLimit[:, :, 0]
        self.v2gChActPower = self.v2gLimit[:, :, 3] * self.v2gLimit[:, :, 0]

        # V2G constraints
        v_range: cython.int[len(self.v2g)] = range(len(self.v2g))
        c_range: cython.int[len(self.cs)] = range(len(self.cs))
        for v in v_range:
            # Check connection to charging stations
            isConnected = False
            connectedTo = 0

            # Check the charging stations
            for c in c_range:
                isConnected = True
                if self.EV_CS_Info[v, c, 0] > 0:
                    self.v2gChActPower[v, 0] = min(self.v2gChActPower[v, 0], self.csInfo[c, 4])
                    self.v2gDchActPower[v, 0] = min(self.v2gDchActPower[v, 0], self.csInfo[c, 5])

                    connectedTo = c
                else:
                    self.v2gDchXo[v, 0] = 0
                    self.v2gChXo[v, 0] = 0

            if self.v2gChXo[v, 0] + self.v2gDchXo[v, 0] > 1:
                self.v2gDchXo[v, 0] = 1 - self.v2gChXo[v, 0]

            mask = self.v2gLimit[v, :, 2] > 0
            temp_val = self.v2gLimit[v, :, 2][mask]

            if isConnected & (len(temp_val) > 0):
                if self.v2gEnerState[v, 0] < temp_val[0]:
                    next_index = next((idx for idx, val in np.ndenumerate(self.v2gLimit[v, :, 2])
                                       if val == temp_val[0]))[0]
                    min_tsteps = np.ceil((temp_val[0] - self.v2gEnerState[v, 0]) / self.csInfo[connectedTo, 4]) - 1

                    if min_tsteps >= next_index:
                        self.v2gChXo[v, 0] = 1
                        self.v2gDchXo[v, 0] = 0

            self.v2gChActPower[v, 0] *= self.v2gChXo[v, 0]
            self.v2gDchActPower[v, 0] *= self.v2gDchXo[v, 0]

            if self.v2gLimit[v, 0, 0] == 0:
                self.v2gEnerState[v, 0] = 0
            elif self.v2gLimit[v, 0, 0] == 1:
                self.v2gEnerState[v, 0] = self.v2gLimit[v, 0, 1] + self.v2gChActPower[v, 0] * self.v2gInfo[v, 7] - \
                                          self.v2gDchActPower[v, 0] / self.v2gInfo[v, 8]

            # Timestep
            for t in range(1, len(self.t)):

                isConnected = False
                connectedTo = 0

                # Check the charging stations
                for c in c_range:
                    if self.EV_CS_Info[v, c, t] > 0:
                        isConnected = True
                        self.v2gChActPower[v, t] = min(self.v2gChActPower[v, t], self.csInfo[c, 4])
                        self.v2gDchActPower[v, t] = min(self.v2gDchActPower[v, t], self.csInfo[c, 5])

                        connectedTo = c
                    else:
                        self.v2gDchXo[v, t] = 0
                        self.v2gChXo[v, t] = 0

                # Disable charge and discharge in the same period
                if self.v2gChXo[v, t] + self.v2gDchXo[v, t] > 1:
                    self.v2gDchXo[v, t] = 1 - self.v2gChXo[v, t]

                # Incentivise charge to meet minimum limits
                mask = self.v2gLimit[v, t:, 2] > 0
                temp_val = self.v2gLimit[v, t:, 2][mask]

                # Check if there are any requirements for EVs
                if isConnected & (len(temp_val) > 0):
                    if self.v2gEnerState[v, t - 1] < temp_val[0]:
                        next_index = next((idx for idx, val in np.ndenumerate(self.v2gLimit[v, t:, 2])
                                           if val == temp_val[0]))[0]
                        min_tsteps = np.ceil(
                            (temp_val[0] - self.v2gEnerState[v, t - 1]) / self.csInfo[connectedTo, 4]) - 1
                        if min_tsteps <= next_index:
                            self.v2gChXo[v, t] = 1
                            self.v2gDchXo[v, t] = 0

                            if (self.v2gEnerState[v, t - 1] + (self.v2gChActPower[v, t] * float(self.v2gInfo[v, 7]))) >= \
                                    self.v2gInfo[v, 4]:
                                self.v2gChActPower[v, t] = (self.v2gInfo[v, 4] - self.v2gEnerState[v, t - 1]) / float(
                                    self.v2gInfo[v, 7])

                # Prevent charging when battery is full
                if self.v2gEnerState[v, t - 1] == self.v2gInfo[v, 4]:
                    # print('HERE')
                    self.v2gChXo[v, t] = 0

                # Prevent discharge when battery is empty
                elif self.v2gEnerState[v, t - 1] == 0:
                    self.v2gDchXo[v, t] = 0

                self.v2gChActPower[v, t] *= self.v2gChXo[v, t]
                self.v2gDchActPower[v, t] *= self.v2gDchXo[v, t]

                # Update battery capacity
                if (self.v2gLimit[v, t - 1, 0] == 1) & (self.v2gLimit[v, t, 0] == 1):
                    self.v2gEnerState[v, t] = self.v2gEnerState[v, t - 1] + self.v2gLimit[v, t, 1] + (
                                self.v2gChActPower[v, t] * float(self.v2gInfo[v, 7])) - (
                                                          self.v2gDchActPower[v, t] / float(self.v2gInfo[v, 8]))
                elif (self.v2gLimit[v, t - 1, 0] == 0) & (self.v2gLimit[v, t, 0] == 1):
                    self.v2gEnerState[v, t] = self.v2gLimit[v, t, 1] + (
                                self.v2gChActPower[v, t] * float(self.v2gInfo[v, 7])) + (
                                                          self.v2gDchActPower[v, t] / float(self.v2gInfo[v, 8]))

        return

    def checkCS(self):

        self.csActPower = np.zeros((len(self.cs),
                                    len(self.t)))

        self.csActPowerNet = np.zeros((len(self.cs),
                                       len(self.t)))

        c: cython.int
        t: cython.int
        v: cython.int

        t_range: cython.int[len(self.t)] = range(len(self.t))
        c_range: cython.int[len(self.cs)] = range(len(self.cs))
        v_range: cython.int[len(self.v2g)] = range(len(self.v2g))

        # Timesteps
        for t in t_range:

            # Charging station constraints
            for c in c_range:

                temp_val = 0
                temp_val2 = 0
                for v in v_range:
                    if self.EV_CS_Info[v, c, t] > 0:
                        temp_val += (self.v2gChActPower[v, t] - self.v2gDchActPower[v, t])
                        temp_val2 += (self.v2gChActPower[v, t] / (self.csInfo[c, 6] / 100) - (
                                    self.v2gDchActPower[v, t] * self.csInfo[c, 7] / 100))

                if temp_val > self.csInfo[c, 4]:
                    temp_val = self.csInfo[c, 4]
                if temp_val < -self.csInfo[c, 5]:
                    temp_val = -self.csInfo[c, 5]

                self.csActPower[c, t] = temp_val
                self.csActPowerNet[c, t] = temp_val2
        return

    def encode(self):
        return np.concatenate([self.__dict__[current_var].ravel()
                               for current_var in self.__var_names__])

    def decode(self, new_variables):
        splits = np.cumsum(self.__var_idx__)
        variables_split = np.split(new_variables, splits[:-1])

        for name, variable in zip(self.__var_names__, variables_split):
            self.__dict__[name] = np.reshape(variable, self.__initial_variables__[name].shape)

        self.loadCutActPower = np.zeros((len(self.load),
                                         len(self.t)))

        self.storEnerState = np.zeros((len(self.stor),
                                       len(self.t)))

        return
