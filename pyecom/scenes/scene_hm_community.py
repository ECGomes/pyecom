# Scenario based on Hugo Morais' community scene
import numpy as np

from pyecom.scenes import BaseScene
from pyecom.algorithms import HydeDF


class HMCommunityScene(BaseScene):

    def __init__(self, name: str, components: dict):
        super().__init__(name, components)

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
        self.component_size_split = [components[component].ravel().shape[0]
                                     for component in components.keys()]

        # Lower and upper bounds
        self.lower_bounds = None
        self.upper_bounds = None

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

        for name, component in zip(self.components.keys(), decoded_splits):
            temp_dict[name] = component.reshape(self.components[name].shape)

        return temp_dict

    def initialize(self):

        # Set the number of iterations to 0
        self.n_iter = 0

        # Set the objective function value to an empty list
        self.objective_function_val = []

        # Set the objective function value history to an empty list
        self.objective_function_val_history = []

        # Set the component size split to the number of components
        self.component_size_split = [self.components[component].ravel().shape[0]
                                        for component in self.components.keys()]

        # Set the current best to None
        self.current_best = None

        # Set the current best index to None
        self.current_best_idx = None

        # Set the current best fitness to None
        self.current_best_fitness = None

        # Set the lower and upper bounds
        self.lower_bounds = np.concatenate([self.components[component].lower_bound.ravel()
                                            for component in self.components.keys()])
        self.upper_bounds = np.concatenate([self.components[component].upper_bound.ravel()
                                            for component in self.components.keys()])

        return

    def repair(self):
        return

    def repair_member(self, x: dict) -> dict:
        return x

    def evaluate(self):
        return

    def evaluate_member(self, x: dict) -> float:
        return 0.0

    def run(self):

        # Initialize the algorithm
        algo = HydeDF(n_iter=200, iter_tolerance=10, epsilon_tolerance=1e-6,
                      pop_size=10, pop_dim=self.lower_bounds.shape[0],
                      lower_bound=self.lower_bounds, upper_bound=self.upper_bounds,
                      f_weight=0.5, f_cr=0.9)
        algo.initialize()  # Generates the initial population

        # Evaluate the initial population
        # Requires a decoding and initial fix

        for i in np.arange(self.n_iter):

            for member_idx in algo.population.shape[0]:
                member = self.decode(algo.population[member_idx])
                member = self.repair_member(member)
                member_fitness = self.evaluate_member(member)

                # Update the population fitness
                self.objective_function_val.append(member_fitness)
                algo.population_fitness[member_idx] = member_fitness

            # Update the best fitness
            self.current_best_fitness = np.min(algo.population_fitness)
            self.current_best_idx = np.argmin(algo.population_fitness)
            self.current_best = self.decode(algo.population[self.current_best_idx])

            algo.current_best_fitness = self.current_best_fitness
            algo.current_best_idx = self.current_best_idx
            algo.current_best = self.current_best

            # Update algorithm iteration count
            algo.current_iteration = i

            # Apply the operator and adaptive parameters
            algo.update_population()

            # Elite selection
            algo.selection_mechanism()

            # Update remaining parameters and history
            algo.post_update_cleanup()

            # Check for stopping criteria
            if algo.check_stopping_criteria():
                break

        return
