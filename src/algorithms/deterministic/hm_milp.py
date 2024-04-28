import numpy as np
import pandas as pd
import pyomo.environ as pe

from src.resources import Generator, Load, Storage, Vehicle, Aggregator
from src.algorithms.deterministic.utils import convert_to_dictionary


class HMMilpPyomo:
    """
    Class to create the Pyomo model for the HM MILP problem.
    Considers Generators, Loads, Storage, Vehicles and Imports/Exports.
    """

    def __init__(self, resources, default_behaviour=pe.Constraint.Skip):

        self.resources = resources
        self.default_behaviour = default_behaviour

        # Resource types
        self.generators = self.separate_resources(Generator)
        self.loads = self.separate_resources(Load)
        self.storages = self.separate_resources(Storage)
        self.vehicles = self.separate_resources(Vehicle)
        self.aggregator = self.separate_resources(Aggregator)[0]

        # Model variables
        self.model = None

    def set_resources(self, resources):
        """
        Set the resources for the Pyomo model.
        To be used if resources need to be updated after class is instantiated.
        :param resources: list of resources
        """
        self.resources = resources
        return

    def separate_resources(self, type):
        """
        Separate resources by type.
        :param type: type of resource to separate
        """
        temp = [res for res in self.resources if isinstance(res, type)]

        return temp

    def create_model(self):
        """
        Create the Pyomo model.
        """
        # Create the model
        self.model = pe.ConcreteModel()
        return

    def solve_model(self,
                    write_path: str = None,
                    solver_path: str = None,
                    solver_io_options=None):

        if solver_io_options is None:
            solver_io_options = {'symbolic_solver_labels': True}

        # Create the model file
        self.model.write(write_path + '_model.lp', io_options=solver_io_options)

        # Create the solver
        solver = pe.SolverFactory('scip', executable=solver_path)
        solver.options['LogFile'] = write_path + '_log.log'

        # Solve the model
        results = solver.solve(self.model)
        results.write()

        return

    @staticmethod
    def extract_results(vals):
        # make a pd.Series from each
        s = pd.Series(vals.extract_values(),
                      index=vals.extract_values().keys())

        # if the series is multi-indexed we need to unstack it...
        if type(s.index[0]) == tuple:  # it is multi-indexed
            s = s.unstack(level=1)
        else:
            # force transition from Series -> df
            s = pd.DataFrame(s)

        return s

    def add_model_sets(self):
        """
        Add the sets to the Pyomo model (time dimension and resource numbers).
        """
        self.model.t = pe.Set(initialize=np.arange(1, self.generators[0].upper_bound.shape[0] + 1))
        self.model.gen = pe.Set(initialize=np.arange(1, len(self.generators) + 1))
        self.model.loads = pe.Set(initialize=np.arange(1, len(self.loads) + 1))
        self.model.stor = pe.Set(initialize=np.arange(1, len(self.storages) + 1))
        self.model.v2g = pe.Set(initialize=np.arange(1, len(self.vehicles) + 1))

        return

    def add_model_aggregator(self):
        # Parameters (information to pass to the model)
        self.model.impMax = pe.Param(self.model.t,
                                     initialize=convert_to_dictionary(self.aggregator.import_max),
                                     doc='Maximum import power')
        self.model.expMax = pe.Param(self.model.t,
                                     initialize=convert_to_dictionary(self.aggregator.export_max),
                                     doc='Maximum export power')
        self.model.buyPrice = pe.Param(self.model.t,
                                       initialize=convert_to_dictionary(self.aggregator.import_cost),
                                       doc='Buy price')
        self.model.sellPrice = pe.Param(self.model.t,
                                        initialize=convert_to_dictionary(self.aggregator.export_cost),
                                        doc='Sell price')

        # Variables (values to be optimized)
        self.model.imports = pe.Var(self.model.t, within=pe.NonNegativeReals, initialize=0,
                                    doc='Import power')
        self.model.exports = pe.Var(self.model.t, within=pe.NonNegativeReals, initialize=0,
                                    doc='Export power')
        self.model.import_binary = pe.Var(self.model.t, within=pe.Binary, initialize=0,
                                          doc='Binary variable for import')
        self.model.export_binary = pe.Var(self.model.t, within=pe.Binary, initialize=0,
                                          doc='Binary variable for export')

        # Constraints
        def impMax_rule(model, t):
            return model.imports[t] <= model.impMax[t] * model.import_binary[t]

        self.model.impMax_rule = pe.Constraint(self.model.t, rule=impMax_rule)

        def expMax_rule(model, t):
            return model.exports[t] <= model.expMax[t] * model.export_binary[t]

        self.model.expMax_rule = pe.Constraint(self.model.t, rule=expMax_rule)

        def impExp_rule(model, t):
            return model.import_binary[t] + model.export_binary[t] <= 1

        self.model.impExp_rule = pe.Constraint(self.model.t, rule=impExp_rule)

        return

    def add_model_generators(self):
        # Parameters (information to pass to the model)
        self.model.genMin = pe.Param(self.model.gen, self.model.t,
                                     initialize=convert_to_dictionary(
                                         np.concatenate([res.lower_bound.reshape(1, -1) for res in self.generators],
                                                        axis=0)),
                                     doc='Minimum generation power')
        self.model.genMax = pe.Param(self.model.gen, self.model.t,
                                     initialize=convert_to_dictionary(
                                         np.concatenate([res.upper_bound.reshape(1, -1) for res in self.generators],
                                                        axis=0)),
                                     doc='Maximum generation power')
        self.model.genCost = pe.Param(self.model.gen, self.model.t,
                                      initialize=convert_to_dictionary(
                                          np.concatenate([res.cost.reshape(1, -1) for res in self.generators], axis=0)),
                                      doc='Generation cost')
        self.model.genCostNDE = pe.Param(self.model.gen, self.model.t,
                                         initialize=convert_to_dictionary(
                                             np.concatenate([res.cost_nde.reshape(1, -1) for res in self.generators],
                                                            axis=0)),
                                         doc='Generation cost NDE')
        self.model.genType = pe.Param(self.model.gen,
                                      initialize=convert_to_dictionary(
                                          np.array([res.is_renewable for res in self.generators])),
                                      doc='Generation type')

        # Variables (values to be optimized)
        self.model.genPower = pe.Var(self.model.gen, self.model.t, within=pe.NonNegativeReals, initialize=0,
                                     doc='Generation power')
        self.model.genExcessPower = pe.Var(self.model.gen, self.model.t, within=pe.NonNegativeReals, initialize=0,
                                           doc='Excess generation power')
        self.model.gen_binary = pe.Var(self.model.gen, self.model.t, within=pe.Binary, initialize=0,
                                       doc='Binary variable for generation')

        # Constraints
        def genMin_rule(model, g, t):
            if not model.genType[g]:
                return model.genPower[g, t] >= model.genMin[g, t] * model.gen_binary[g, t]
            return pe.Constraint.Skip

        self.model.genMin_rule = pe.Constraint(self.model.gen, self.model.t, rule=genMin_rule)

        def genMax_rule(model, g, t):
            if not model.genType[g]:
                return model.genPower[g, t] <= model.genMax[g, t]
            else:
                return model.genPower[g, t] + model.genExcessPower[g, t] == model.genMax[g, t]

        self.model.genMax_rule = pe.Constraint(self.model.gen, self.model.t, rule=genMax_rule)

        return

    def add_model_loads(self):
        # Parameters (information to pass to the model)
        self.model.loadMax = pe.Param(self.model.loads, self.model.t,
                                      initialize=convert_to_dictionary(
                                          np.concatenate([res.upper_bound.reshape(1, -1) for res in self.loads],
                                                         axis=0)),
                                      doc='Maximum load power')
        self.model.loadCut = pe.Param(self.model.loads, self.model.t,
                                      initialize=convert_to_dictionary(
                                          np.concatenate([res.upper_bound.reshape(1, -1) for res in self.loads],
                                                         axis=0)),
                                      doc='Load cut power')
        self.model.loadReduce = pe.Param(self.model.loads, self.model.t,
                                         initialize=convert_to_dictionary(
                                             np.concatenate([res.upper_bound.reshape(1, -1) for res in self.loads],
                                                            axis=0)),
                                         doc='Load reduce power')
        self.model.loadEns = pe.Param(self.model.loads, self.model.t,
                                      initialize=convert_to_dictionary(
                                          np.concatenate([res.upper_bound.reshape(1, -1) for res in self.loads],
                                                         axis=0)),
                                      doc='Load ens power')
        self.model.loadCutCost = pe.Param(self.model.loads, self.model.t,
                                          initialize=convert_to_dictionary(
                                              np.array([res.cost_cut for res in self.loads])),
                                          doc='Load cut cost')
        self.model.loadReduceCost = pe.Param(self.model.loads, self.model.t,
                                             initialize=convert_to_dictionary(
                                                 np.array([res.cost_reduce for res in self.loads])),
                                             doc='Load reduce cost')
        self.model.loadEnsCost = pe.Param(self.model.loads, self.model.t,
                                          initialize=convert_to_dictionary(
                                              np.array([res.cost_ens for res in self.loads])),
                                          doc='Load ENS cost')

        # Variables (values to be optimized)
        self.model.loadCutPower = pe.Var(self.model.loads, self.model.t, within=pe.NonNegativeReals, initialize=0,
                                         doc='Load cut power')
        self.model.loadReducePower = pe.Var(self.model.loads, self.model.t, within=pe.NonNegativeReals, initialize=0,
                                            doc='Load reduce power')
        self.model.loadEnsPower = pe.Var(self.model.loads, self.model.t, within=pe.NonNegativeReals, initialize=0,
                                         doc='Load ENS power')
        self.model.load_binary = pe.Var(self.model.loads, self.model.t, within=pe.Binary, initialize=0,
                                        doc='Binary variable for load')

        # Constraints
        def loadReduce_rule(model, l, t):
            return model.loadReducePower[l, t] <= model.loadReduce[l, t]

        self.model.loadReduce_rule = pe.Constraint(self.model.loads, self.model.t, rule=loadReduce_rule)

        def loadCut_rule(model, l, t):
            return model.loadCutPower[l, t] <= model.loadCut[l, t] * model.load_binary[l, t]

        self.model.loadCut_rule = pe.Constraint(self.model.loads, self.model.t, rule=loadCut_rule)

        def loadEns_rule(model, l, t):
            return model.loadEnsPower[l, t] + model.loadCutPower[l, t] + model.loadReducePower[l, t] <= model.loadMax[
                l, t]

        self.model.loadEns_rule = pe.Constraint(self.model.loads, self.model.t, rule=loadEns_rule)

        return

    def add_model_storages(self):
        # Parameters (information to pass to the model)
        self.model.storageDischargeMax = pe.Param(self.model.stor, self.model.t,
                                                  initialize=convert_to_dictionary(np.concatenate(
                                                      [res.discharge_max.reshape(1, -1) for res in self.storages],
                                                      axis=0)),
                                                  doc='Maximum discharge power')
        self.model.storageChargeMax = pe.Param(self.model.stor, self.model.t,
                                               initialize=convert_to_dictionary(np.concatenate(
                                                   [res.charge_max.reshape(1, -1) for res in self.storages], axis=0)),
                                               doc='Maximum charge power')
        self.model.storageCapacityMax = pe.Param(self.model.stor,
                                                 initialize=convert_to_dictionary(
                                                     np.concatenate([[res.capacity_max] for res in self.storages])),
                                                 doc='Maximum storage capacity')
        self.model.storageCapacityMin = pe.Param(self.model.stor,
                                                 initialize=convert_to_dictionary(
                                                     np.concatenate([[res.capacity_min] for res in self.storages],
                                                                    axis=0)),
                                                 doc='Minimum storage capacity')
        self.model.storageInitialCharge = pe.Param(self.model.stor,
                                                   initialize=convert_to_dictionary(
                                                       np.concatenate([[res.initial_charge] for res in self.storages])),
                                                   doc='Initial storage charge')
        self.model.storageDischargeEfficiency = pe.Param(self.model.stor,
                                                         initialize=convert_to_dictionary(np.concatenate(
                                                             [[res.discharge_efficiency] for res in self.storages])),
                                                         doc='Discharge efficiency')
        self.model.storageChargeEfficiency = pe.Param(self.model.stor,
                                                      initialize=convert_to_dictionary(np.concatenate(
                                                          [[res.charge_efficiency] for res in self.storages])),
                                                      doc='Charge efficiency')
        self.model.storageDischargeCost = pe.Param(self.model.stor, self.model.t,
                                                   initialize=convert_to_dictionary(np.concatenate(
                                                       [res.cost_discharge.reshape(1, -1) for res in self.storages],
                                                       axis=0)),
                                                   doc='Discharge cost')
        self.model.storageChargeCost = pe.Param(self.model.stor, self.model.t,
                                                initialize=convert_to_dictionary(np.concatenate(
                                                    [res.cost_charge.reshape(1, -1) for res in self.storages], axis=0)),
                                                doc='Charge cost')

        # Variables (values to be optimized)
        self.model.storageState = pe.Var(self.model.stor, self.model.t, within=pe.NonNegativeReals, initialize=0,
                                         doc='Storage state of charge')
        self.model.storageDischargePower = pe.Var(self.model.stor, self.model.t, within=pe.NonNegativeReals,
                                                  initialize=0,
                                                  doc='Storage discharge power')
        self.model.storageChargePower = pe.Var(self.model.stor, self.model.t, within=pe.NonNegativeReals, initialize=0,
                                               doc='Storage charge power')
        self.model.storageRelaxPower = pe.Var(self.model.stor, self.model.t, within=pe.NonNegativeReals, initialize=0,
                                              doc='Storage relaxation power')
        self.model.storageDischarge_binary = pe.Var(self.model.stor, self.model.t, within=pe.Binary, initialize=0,
                                                    doc='Binary variable for storage discharge')
        self.model.storageCharge_binary = pe.Var(self.model.stor, self.model.t, within=pe.Binary, initialize=0,
                                                 doc='Binary variable for storage charge')

        # Constraints
        def storageDischarge_rule(model, s, t):
            return model.storageDischargePower[s, t] <= model.storageDischargeMax[s, t] * model.storageDischarge_binary[
                s, t]

        self.model.storageDischarge_rule = pe.Constraint(self.model.stor, self.model.t, rule=storageDischarge_rule)

        def storageCharge_rule(model, s, t):
            return model.storageChargePower[s, t] <= model.storageChargeMax[s, t] * model.storageCharge_binary[s, t]

        self.model.storageCharge_rule = pe.Constraint(self.model.stor, self.model.t, rule=storageCharge_rule)

        def storageState_rule(model, s, t):
            return model.storageState[s, t] <= model.storageCapacityMax[s]

        self.model.storageState_rule = pe.Constraint(self.model.stor, self.model.t, rule=storageState_rule)

        def storageStateMin_rule(model, s, t):
            return model.storageState[s, t] >= model.storageCapacityMin[s] * model.storageCapacityMax[s] - \
                model.storageRelaxPower[s, t]

        self.model.storageStateMin_rule = pe.Constraint(self.model.stor, self.model.t, rule=storageStateMin_rule)

        def storageBalance_rule(model, s, t):
            if t == 1:
                return model.storageState[s, t] == model.storageCapacityMax[s] * model.storageInitialCharge[s] \
                    + model.storageChargePower[s, t] * model.storageChargeEfficiency[s] \
                    - model.storageDischargePower[s, t] / model.storageDischargeEfficiency[s]
            elif t > 1:
                return model.storageState[s, t] == model.storageState[s, t - 1] \
                    + model.storageChargePower[s, t] * model.storageChargeEfficiency[s] \
                    - model.storageDischargePower[s, t] / model.storageDischargeEfficiency[s]

        self.model.storageBalance_rule = pe.Constraint(self.model.stor, self.model.t, rule=storageBalance_rule)

        def storageChargeDischarge_rule(model, s, t):
            return model.storageCharge_binary[s, t] + model.storageDischarge_binary[s, t] <= 1

        self.model.storageChargeDischarge_rule = pe.Constraint(self.model.stor, self.model.t,
                                                               rule=storageChargeDischarge_rule)

        return

    def add_model_vehicles(self):
        # Parameters (information to pass to the model)
        self.model.v2gDischargeMax = pe.Param(self.model.v2g, self.model.t,
                                              initialize=convert_to_dictionary(np.concatenate(
                                                  [res.schedule_discharge.reshape(1, -1) for res in self.vehicles],
                                                  axis=0)),
                                              doc='Maximum discharge power')
        self.model.v2gChargeMax = pe.Param(self.model.v2g, self.model.t,
                                           initialize=convert_to_dictionary(np.concatenate(
                                               [res.schedule_charge.reshape(1, -1) for res in self.vehicles], axis=0)),
                                           doc='Maximum charge power')
        self.model.v2gDischargeEfficiency = pe.Param(self.model.v2g,
                                                     initialize=convert_to_dictionary(np.concatenate(
                                                         [[res.discharge_efficiency] for res in self.vehicles])),
                                                     doc='Discharge efficiency')
        self.model.v2gChargeEfficiency = pe.Param(self.model.v2g,
                                                  initialize=convert_to_dictionary(np.concatenate(
                                                      [[res.charge_efficiency] for res in self.vehicles])),
                                                  doc='Charge efficiency')
        self.model.v2gCapacityMax = pe.Param(self.model.v2g,
                                             initialize=convert_to_dictionary(
                                                 np.concatenate([[res.capacity_max] for res in self.vehicles])),
                                             doc='Maximum vehicle capacity')
        self.model.v2gCapacityMin = pe.Param(self.model.v2g,
                                             initialize=convert_to_dictionary(
                                                 np.concatenate([[res.min_charge] for res in self.vehicles])),
                                             doc='Minimum vehicle capacity')
        self.model.v2gScheduleConnected = pe.Param(self.model.v2g, self.model.t,
                                                   initialize=convert_to_dictionary(np.concatenate(
                                                       [res.schedule_connected.reshape(1, -1) for res in self.vehicles],
                                                       axis=0)),
                                                   doc='Vehicle connected schedule')
        self.model.v2gScheduleArrivalSOC = pe.Param(self.model.v2g, self.model.t,
                                                    initialize=convert_to_dictionary(np.concatenate(
                                                        [res.schedule_arrival_soc.reshape(1, -1) for res in
                                                         self.vehicles], axis=0)),
                                                    doc='Vehicle disconnected schedule')
        self.model.v2gScheduleDepartureSOC = pe.Param(self.model.v2g, self.model.t,
                                                      initialize=convert_to_dictionary(np.concatenate(
                                                          [res.schedule_requirement_soc.reshape(1, -1) for res in
                                                           self.vehicles], axis=0)),
                                                      doc='Vehicle departure SOC')
        self.model.v2gDischargeCost = pe.Param(self.model.v2g, self.model.t,
                                               initialize=convert_to_dictionary(np.concatenate(
                                                   [res.cost_discharge.reshape(1, -1) for res in self.vehicles],
                                                   axis=0)),
                                               doc='Discharge cost')
        self.model.v2gChargeCost = pe.Param(self.model.v2g, self.model.t,
                                            initialize=convert_to_dictionary(np.concatenate(
                                                [res.cost_charge.reshape(1, -1) for res in self.vehicles], axis=0)),
                                            doc='Charge cost')

        # Variables (values to be optimized)
        self.model.v2gDischargePower = pe.Var(self.model.v2g, self.model.t, within=pe.NonNegativeReals, initialize=0,
                                              doc='Vehicle discharge power')
        self.model.v2gChargePower = pe.Var(self.model.v2g, self.model.t, within=pe.NonNegativeReals, initialize=0,
                                           doc='Vehicle charge power')
        self.model.v2gRelaxPower = pe.Var(self.model.v2g, self.model.t, within=pe.NonNegativeReals, initialize=0,
                                          doc='Vehicle relaxation power')
        self.model.v2gState = pe.Var(self.model.v2g, self.model.t, within=pe.NonNegativeReals, initialize=0,
                                     doc='Vehicle state of charge')
        self.model.v2gDischarge_binary = pe.Var(self.model.v2g, self.model.t, within=pe.Binary, initialize=0,
                                                doc='Binary variable for vehicle discharge')
        self.model.v2gCharge_binary = pe.Var(self.model.v2g, self.model.t, within=pe.Binary, initialize=0,
                                             doc='Binary variable for vehicle charge')

        # Constraints
        def v2gDischarge_rule(model, v, t):
            return model.v2gDischargePower[v, t] <= model.v2gDischargeMax[v, t] * model.v2gDischarge_binary[v, t]

        self.model.v2gDischarge_rule = pe.Constraint(self.model.v2g, self.model.t, rule=v2gDischarge_rule)

        def v2gCharge_rule(model, v, t):
            return model.v2gChargePower[v, t] <= model.v2gChargeMax[v, t] * model.v2gCharge_binary[v, t]

        self.model.v2gCharge_rule = pe.Constraint(self.model.v2g, self.model.t, rule=v2gCharge_rule)

        def v2gCapacity_rule(model, v, t):
            return model.v2gState[v, t] <= model.v2gCapacityMax[v]

        self.model.v2gCapacity_rule = pe.Constraint(self.model.v2g, self.model.t, rule=v2gCapacity_rule)

        def v2gRelax_rule(model, v, t):
            if model.v2gScheduleConnected[v, t] == 1:
                return model.v2gState[v, t] >= model.v2gCapacityMin[v] - model.v2gRelaxPower[v, t]
            elif t < model.t.last():
                if (model.v2gScheduleConnected[v, t] == 1) & (model.v2gScheduleConnected[v, t + 1] == 0) & (
                        model.v2gScheduleDepartureSOC[v, t] == 0):
                    return model.v2gState[v, t] >= model.v2gCapacityMax[v] - model.v2gRelaxPower[v, t]
            elif (model.v2gScheduleConnected[v, t] == 1) & (model.v2gScheduleDepartureSOC[v, t] == 0) & (
                    t == model.t.last()):
                return model.v2gState[v, t] >= model.v2gCapacityMax[v] - model.v2gRelaxPower[v, t]
            return self.default_behaviour

        self.model.v2gRelax_rule = pe.Constraint(self.model.v2g, self.model.t, rule=v2gRelax_rule)

        def v2gBalance_rule(model, v, t):
            if model.v2gScheduleConnected[v, t] == 0:
                return model.v2gState[v, t] == 0
            elif (model.v2gScheduleConnected[v, t] == 1) & (t == 1):
                return model.v2gState[v, t] == model.v2gState[v, t] \
                    + model.v2gChargePower[v, t] * model.v2gChargeEfficiency[v] \
                    - model.v2gDischargePower[v, t] / model.v2gDischargeEfficiency[v]
            elif t > 1:
                if (model.v2gScheduleConnected[v, t - 1] == 1) & (model.v2gScheduleConnected[v, t] == 1):
                    return model.v2gState[v, t] == model.v2gState[v, t - 1] \
                        + model.v2gChargePower[v, t] * model.v2gChargeEfficiency[v] \
                        - model.v2gDischargePower[v, t] / model.v2gDischargeEfficiency[v]
                elif (model.v2gScheduleConnected[v, t - 1] == 0) & (model.v2gScheduleConnected[v, t] == 1):
                    return model.v2gState[v, t] == model.v2gScheduleArrivalSOC[v, t] \
                        + model.v2gChargePower[v, t] * model.v2gChargeEfficiency[v] \
                        - model.v2gDischargePower[v, t] / model.v2gDischargeEfficiency[v]
            return self.default_behaviour

        self.model.v2gBalance_rule = pe.Constraint(self.model.v2g, self.model.t, rule=v2gBalance_rule)

        def v2gChargeDischarge_rule(model, v, t):
            return model.v2gCharge_binary[v, t] + model.v2gDischarge_binary[v, t] <= 1

        self.model.v2gChargeDischarge_rule = pe.Constraint(self.model.v2g, self.model.t, rule=v2gChargeDischarge_rule)

        return

    def add_model_balance(self):
        # Calculate the sums

        def balance_rule(model, t):
            sum_gens = sum([model.genPower[g, t] - model.genExcessPower[g, t]
                            for g in np.arange(1, model.gen.last() + 1)])
            sum_loads = sum(
                [model.loadMax[l, t] - model.loadReducePower[l, t] - model.loadCutPower[l, t] - model.loadEnsPower[l, t]
                 for l in np.arange(1, model.loads.last() + 1)])
            sum_storages = sum([model.storageChargePower[s, t] - model.storageDischargePower[s, t]
                                for s in np.arange(1, model.stor.last() + 1)])
            sum_vehicles = sum([model.v2gChargePower[v, t] - model.v2gDischargePower[v, t]
                                for v in np.arange(1, model.v2g.last() + 1)])

            return sum_gens - sum_loads - sum_storages - sum_vehicles + model.imports[t] - model.exports[t] == 0

        self.model.balance_rule = pe.Constraint(self.model.t, rule=balance_rule)

        return

    def add_model_objective(self):

        def objectiveFunction(model):
            cost_gens = sum(
                [model.genPower[g, t] * model.genCost[g, t] + model.genExcessPower[g, t] * model.genCostNDE[g, t]
                 for t in np.arange(1, model.t.last() + 1) for g in np.arange(1, model.gen.last() + 1)])

            cost_loads = sum([model.loadCutPower[l, t] * model.loadCutCost[l, t] \
                              + model.loadReducePower[l, t] * model.loadReduceCost[l, t] \
                              + model.loadEnsPower[l, t] * model.loadEnsCost[l, t]
                              for t in np.arange(1, model.t.last() + 1) for l in np.arange(1, model.loads.last() + 1)])

            cost_storages = sum([model.storageDischargePower[s, t] * model.storageDischargeCost[s, t] \
                                 - model.storageChargePower[s, t] * model.storageChargeCost[s, t]
                                 for t in np.arange(1, model.t.last() + 1) for s in
                                 np.arange(1, model.stor.last() + 1)])

            cost_vehicles = sum([model.v2gDischargePower[v, t] * model.v2gDischargeCost[v, t] \
                                 - model.v2gChargePower[v, t] * model.v2gChargeCost[v, t] \
                                 + model.v2gRelaxPower[v, t] * 200
                                 for t in np.arange(1, model.t.last() + 1) for v in np.arange(1, model.v2g.last() + 1)])

            cost_importExport = sum([model.imports[t] * model.buyPrice[t] - model.exports[t] * model.sellPrice[t]
                                     for t in np.arange(1, model.t.last() + 1)])

            return cost_gens + cost_loads + cost_storages + cost_vehicles + cost_importExport

        self.model.objective = pe.Objective(rule=objectiveFunction, sense=pe.minimize)

        return

    def run_model(self):
        """
        Run the Pyomo model.
        """
        self.create_model()
        self.add_model_sets()
        self.add_model_aggregator()
        self.add_model_generators()
        self.add_model_loads()
        self.add_model_storages()
        self.add_model_vehicles()
        self.add_model_balance()
        self.add_model_objective()

        return

    def extract_all(self):

        results = {'genPower': self.extract_results(self.model.genPower),
                   'genExcessPower': self.extract_results(self.model.genExcessPower),
                   'imports': self.extract_results(self.model.imports),
                   'exports': self.extract_results(self.model.exports),
                   'loadReducePower': self.extract_results(self.model.loadReducePower),
                   'loadCutPower': self.extract_results(self.model.loadCutPower),
                   'loadEnsPower': self.extract_results(self.model.loadEnsPower),
                   'storageDischargePower': self.extract_results(self.model.storageDischargePower),
                   'storageChargePower': self.extract_results(self.model.storageChargePower),
                   'storageState': self.extract_results(self.model.storageState),
                   'v2gDischargePower': self.extract_results(self.model.v2gDischargePower),
                   'v2gChargePower': self.extract_results(self.model.v2gChargePower),
                   'v2gState': self.extract_results(self.model.v2gState)}

        return results
