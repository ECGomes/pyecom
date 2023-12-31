# Scenario based on Hugo Morais' community scene
import numpy as np
import tqdm as tqdm

from src.scenes import BaseScene
from src.algorithms import HydeDF
from ..repairs import HMRepair
from ..parsers import HMParser
from ..resources import BaseResource


class HMProblemScene(BaseScene):

    def __init__(self, name: str, data, hm_parser: HMParser,
                 n_iter=200,
                 iter_tolerance=10,
                 epsilon_tolerance=1e-6,
                 pop_size=10):

        # Parsed data
        self.parsed_data = hm_parser

        # Initialize the components
        super().__init__(name, data)

        # Problem specific parameters
        self.n_steps = self.components['gen'].value.shape[1]
        self.n_gen = self.components['gen'].value.shape[0]
        self.n_load = self.components['loads'].value.shape[0]
        self.n_stor = self.components['stor'].value.shape[0]
        self.n_v2g = self.components['evs'].value.shape[0]

        # Create the variables for the optimization process
        self.decoded_lower_bounds, self.decoded_upper_bounds = self._create_variables()

        # Best instance placeholder
        self.current_best_fitness = None
        self.current_best_idx = None
        self.current_best = None

        # Objective function placeholders
        self.objective_function_val = []
        self.objective_function_val_history = []

        # Placeholder for the number of iterations
        self.n_iter = 0

        # Component size split placeholder
        self.component_size_split = [self.decoded_lower_bounds[component].ravel().shape[0]
                                     for component in self.decoded_lower_bounds.keys()]

        # Lower and upper bounds
        self.lower_bounds = None
        self.upper_bounds = None

        # Repair instance
        self.hm_repair = HMRepair(self.components)

        # Reference for the algorithm instance
        self.algo = None
        self.algo_n_iter = n_iter
        self.algo_iter_tolerance = iter_tolerance
        self.algo_epsilon_tolerance = epsilon_tolerance
        self.algo_pop_size = pop_size

        # Solution placeholder
        self.solution = None

    # Encoding process
    @staticmethod
    def encode(x: dict):
        return np.concatenate([x[component].ravel()
                               for component in x.keys()])

    # Decoding process
    def decode(self, x: np.ndarray):
        splits = np.cumsum(self.component_size_split)
        decoded_splits = np.split(x, splits[:-1])

        temp_dict = {}

        for name, component in zip(self.decoded_lower_bounds.keys(), decoded_splits):
            temp_dict[name] = component.reshape(self.decoded_lower_bounds[name].shape)

        return temp_dict

    def _create_variables(self):

        temp_xl = {'genActPower': self.components['gen'].lower_bound,
                   'genExcActPower': self.components['gen'].lower_bound,
                   'pImp': self.components['pimp'].lower_bound,
                   'pExp': self.components['pexp'].lower_bound,
                   'loadRedActPower': self.components['loads'].lower_bound,
                   'loadCutActPower': self.components['loads'].lower_bound,
                   'loadENS': self.components['loads'].lower_bound,
                   'storDchActPower': self.components['stor'].lower_bound,
                   'storChActPower': self.components['stor'].lower_bound,
                   'EminRelaxStor': self.components['stor'].lower_bound,
                   'storEnerState': self.components['stor'].lower_bound,
                   'v2gDchActPower': self.components['evs'].lower_bound,
                   'v2gChActPower': self.components['evs'].lower_bound,
                   'EminRelaxEV': self.components['evs'].lower_bound,
                   'v2gEnerState': self.components['evs'].lower_bound,
                   'genXo': self.components['gen'].lower_bound,
                   'loadXo': self.components['loads'].lower_bound,
                   'storDchXo': self.components['stor'].lower_bound,
                   'storChXo': self.components['stor'].lower_bound,
                   'v2gDchXo': self.components['evs'].lower_bound,
                   'v2gChXo': self.components['evs'].lower_bound}

        temp_xu = {'genActPower': self.components['gen'].upper_bound,
                   'genExcActPower': self.components['gen'].upper_bound,
                   'pImp': self.components['pimp'].upper_bound,
                   'pExp': self.components['pexp'].upper_bound,
                   'loadRedActPower': self.components['loads'].upper_bound,
                   'loadCutActPower': self.components['loads'].upper_bound,
                   'loadENS': self.components['loads'].upper_bound,
                   'storDchActPower': self.components['stor'].discharge_max,
                   'storChActPower': self.components['stor'].charge_max,
                   'EminRelaxStor': self.components['stor'].upper_bound,
                   'storEnerState': self.components['stor'].upper_bound,
                   'v2gDchActPower': self.components['evs'].schedule_discharge,
                   'v2gChActPower': self.components['evs'].schedule_charge,
                   'EminRelaxEV': self.components['evs'].upper_bound,
                   'v2gEnerState': self.components['evs'].upper_bound,
                   'genXo': np.ones(self.components['gen'].value.shape),
                   'loadXo': np.ones(self.components['loads'].value.shape),
                   'storDchXo': np.ones(self.components['stor'].value.shape),
                   'storChXo': np.ones(self.components['stor'].value.shape),
                   'v2gDchXo': np.ones(self.components['evs'].value.shape),
                   'v2gChXo': np.ones(self.components['evs'].value.shape)}

        return temp_xl, temp_xu

    def initialize(self):

        # Set the number of iterations to 0
        self.n_iter = 0

        # Set the objective function value to an empty list
        self.objective_function_val = []

        # Set the objective function value history to an empty list
        self.objective_function_val_history = []

        # Set the component size split to the number of components
        self.component_size_split = [self.decoded_lower_bounds[component].ravel().shape[0]
                                     for component in self.decoded_lower_bounds.keys()]

        # Set the current best to None
        self.current_best = None

        # Set the current best index to None
        self.current_best_idx = None

        # Set the current best fitness to None
        self.current_best_fitness = None

        # Set the lower and upper bounds
        self.lower_bounds = np.concatenate([self.decoded_lower_bounds[component].ravel()
                                            for component in self.decoded_lower_bounds.keys()])
        self.upper_bounds = np.concatenate([self.decoded_upper_bounds[component].ravel()
                                            for component in self.decoded_upper_bounds.keys()])

        return

    def repair(self, x):

        # Repair the member
        repaired_member = self.hm_repair.repair(x)

        return repaired_member

    def evaluate(self, x):
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
            if x['pImp'][t] > self.components['pimp'].upper_bound[t]:
                balance_penalty += 100000

            if x['pExp'][t] > self.components['pexp'].upper_bound[t]:
                balance_penalty += 100000

        # Calculate the individual component costs
        temp_gens = np.sum([x['genActPower'] * self.components['gen'].cost +
                            x['genExcActPower'] * self.components['gen'].cost_nde])

        temp_loads = np.sum([x['loadRedActPower'] * self.components['loads'].cost_reduce +
                             x['loadCutActPower'] * self.components['loads'].cost_cut +
                             x['loadENS'] * self.components['loads'].cost_ens])

        temp_stor: float = sum([self.components['stor'].capital_cost[s] *
                                (x['storEnerState'][s, t] / self.components['stor'].capacity_max[s] - 0.63) ** 2 +
                                x['storDchActPower'][s, t] * self.components['stor'].cost_discharge[s, t] +
                                x['storChActPower'][s, t] * self.components['stor'].cost_charge[s, t] +
                                6.5e-3 / self.components['stor'].capacity_max[s] * x['storChActPower'][
                                    s, t] ** 2
                                for t in t_range for s in stor_range])

        temp_v2g: float = sum([self.components['evs'].capital_cost[v] *
                               (x['v2gEnerState'][v, t] / self.components['evs'].capacity_max[v] - 0.63) ** 2 +
                               x['v2gDchActPower'][v, t] * self.components['evs'].cost_discharge[v] +
                               x['v2gChActPower'][v, t] * self.components['evs'].cost_charge[v] +
                               6.5e-3 / self.components['evs'].capacity_max[v] * x['v2gChActPower'][v, t] ** 2
                               for t in t_range for v in v2g_range])

        temp_rest: float = sum([x['pImp'][t] * self.components['pimp'].cost[t] +
                                x['pExp'][t] * self.components['pexp'].cost[t]
                                for t in t_range])

        # print the components of the objective function for debugging
        #print('temp_gens: ', temp_gens)
        #print('temp_loads: ', temp_loads)
        #print('temp_stor: ', temp_stor)
        #print('temp_v2g: ', temp_v2g)
        #print('temp_rest: ', temp_rest)
        #print('balance_penalty: ', balance_penalty)
        #print('\n')

        obj_fn = temp_gens + temp_loads + temp_stor + temp_v2g + temp_rest + balance_penalty

        return obj_fn

    def run(self):

        # Initialize the algorithm
        self.algo = HydeDF(n_iter=self.algo_n_iter, iter_tolerance=self.algo_iter_tolerance,
                           epsilon_tolerance=self.algo_epsilon_tolerance,
                           pop_size=self.algo_pop_size,
                           pop_dim=self.lower_bounds.shape[0],
                           lower_bound=self.lower_bounds, upper_bound=self.upper_bounds,
                           f_weight=0.5, f_cr=0.9)
        self.algo.initialize()  # Generates the initial population

        # Evaluate the initial population
        # Requires a decoding and initial fix
        current_pop_fitness = []
        for member_idx in np.arange(self.algo.population.shape[0]):
            member = self.decode(self.algo.population[member_idx])
            member = self.repair(member)
            member_fitness = self.evaluate(member)

            # Update the population fitness
            current_pop_fitness.append(member_fitness)
            self.algo.population[member_idx] = self.encode(member)
            self.algo.population_fitness[member_idx] = member_fitness

            # Since it's the first iteration, the old population is the same as the new population
            self.algo.population_old = self.algo.population
            self.algo.population_old_fitness = self.algo.population_fitness
            self.algo.population_history.append(self.algo.population)
        self.objective_function_val.append(current_pop_fitness)

        # Update the best fitness
        self.current_best_fitness = np.min(self.algo.population_fitness)
        self.current_best_idx = np.argmin(self.algo.population_fitness)
        self.current_best = self.decode(self.algo.population[self.current_best_idx])

        self.algo.current_best_fitness = self.current_best_fitness
        self.algo.current_best_idx = self.current_best_idx
        self.algo.current_best = self.encode(self.current_best)

        # Main loop
        for i in tqdm.tqdm(np.arange(self.algo.n_iter)):

            # Update algorithm iteration count
            self.algo.current_iteration = i

            # Apply the operator and adaptive parameters
            self.algo.update_population()

            # Repair the new population
            current_pop_fitness = []
            for member_idx in np.arange(self.algo.population.shape[0]):
                member = self.decode(self.algo.population[member_idx])
                member = self.repair(member)
                member_fitness = self.evaluate(member)

                # Update the population member and its fitness
                current_pop_fitness.append(member_fitness)
                self.algo.population[member_idx] = self.encode(member)
                self.algo.population_fitness[member_idx] = member_fitness
            self.objective_function_val.append(current_pop_fitness)

            # Update the best fitness
            self.current_best_fitness = np.min(self.algo.population_fitness)
            self.current_best_idx = np.argmin(self.algo.population_fitness)
            self.current_best = self.decode(self.algo.population[self.current_best_idx])

            self.algo.current_best_fitness = self.current_best_fitness
            self.algo.current_best_idx = self.current_best_idx
            self.algo.current_best = self.encode(self.current_best)

            # Update remaining parameters and history
            self.algo.post_update_cleanup()

            # Check for stopping criteria
            if self.algo.check_stopping_criteria():
                break

        return
