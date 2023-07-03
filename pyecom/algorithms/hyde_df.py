# HyDE-DF implementation that extends the MetaheuristicsBase class
import copy

from metaheuristic_base import MetaheuristicBase

import numpy as np


class HydeDF(MetaheuristicBase):

    def __init__(self, n_iter: int, pop_size: int, pop_dim: int,
                 lb: np.ndarray, ub: np.ndarray, f_weight: float, f_cr: float):
        super().__init__(n_iter=n_iter,
                         pop_size=pop_size,
                         pop_dim=pop_dim)

        self.lb = lb
        self.ub = ub

        # HyDE-DF adaptive parameters
        self.f_weight = self.f_weight_old = np.tile(f_weight, (self.pop_size, 3))
        self.f_cr = self.f_cr_old = np.tile(f_cr, (self.pop_size, 1))

        # Operator placeholder values
        self.fvr_rotation = np.arange(self.pop_size)

        # Placeholder values for population and population history
        self.population = None
        self.population_old = None
        self.population_history = None

        # Placeholder values for best population member and it's index
        self.current_best = None
        self.current_best_idx = None

        # Linear decrease placeholder value
        self.linear_decrease: float = 1.0

        # Current iteration value
        self.current_iteration: int = 0

    def _generate_population(self):
        """
        Method to generate an initial population
        :return: Generated initial population
        """
        return np.random.uniform(low=self.lb, high=self.ub, size=(self.pop_size,
                                                                  self.pop_dim))

    def _update(self):
        """
        Population clip by lower and upper bounds
        :param pop: Population received
        :param lb: Population lower bound
        :param ub: Population upper bound
        :return: Clipped population
        """
        return np.clip(self.population, self.lb, self.ub)

    def _initial_check(self):

        if self.pop_size < 5:
            self.pop_size = 5
            print('Population size increase to minimal value of 5\n')

        if (self.f_cr < 0.0) | (self.f_cr > 1.0):
            self.f_cr = 0.5
            print('Crossover rate must be between 0.0 and 1.0 (inclusive). Set to 0.5\n')

        if self.n_iter < 0:
            self.n_iter = 200
            print('Negative iterations encountered. Set value to 200\n')

    def _calculate_linear_decrease(self) -> float:
        return (self.n_iter - self.current_iteration) / self.n_iter

    def _operator(self):
        """
        Operator for the HyDE-DF crossover
        :param population:
        :param best_member:
        :return:
        """

        # Random permutations
        permutation_idx = np.random.permutation(5)

        # Shuffle the vectors
        fvr_idx = np.random.permutation(self.pop_size)
        fvr_rt = (self.fvr_rotation + permutation_idx[0]) % self.pop_size
        fvr_idx2 = fvr_idx[fvr_rt]

        # Shuffled population
        pop_rot01 = self.population_old[fvr_idx, :]
        pop_rot02 = self.population_old[fvr_idx2, :]

        # Mutated population
        pop_mutated_inverse = (np.random.uniform(size=(self.pop_size, self.pop_dim)) < self.f_cr).astype(int)
        pop_mutated = np.logical_not(pop_mutated_inverse).astype(int)

        # Best member
        population_best_member = np.tile(self.current_best, (self.pop_size, 1))

        # Exponential decrease
        self.linear_decrease = self._calculate_linear_decrease()
        exp_decrease = np.exp(1 - (1 / self.linear_decrease ** 2))

        # Differential variation
        pop_00 = np.reshape(np.tile(self.f_weight[:, 2],
                                    (1, self.pop_dim)),
                            (self.f_weight.shape[0], self.pop_dim))
        pop_01 = np.reshape(np.tile(self.f_weight[:, 0],
                                    (1, self.pop_dim)),
                            (self.f_weight.shape[0], self.pop_dim))
        pop_02 = np.reshape(np.tile(self.f_weight[:, 1],
                                    (1, self.pop_dim)),
                            (self.f_weight.shape[0], self.pop_dim))

        diff_var = (pop_01 * (population_best_member *
                              (pop_02 +
                               np.random.uniform(size=(self.pop_size,
                                                       self.pop_dim) - self.population_old)))) * exp_decrease

        # Population update
        new_population = self.population_old + pop_00 * (pop_rot01 - pop_rot02) + diff_var
        new_population = self.population_old * pop_mutated + new_population * pop_mutated_inverse

        return new_population, population_best_member

    # Search loop
    def _iterate(self):
        """
        Method to run the search loop
        :return: None
        """

        # TODO: Separate all the calculations into separate methods
        # TODO: Objective is to have a single method with a single responsibility

        # Generate initial population
        self.population = self._generate_pop()

        # Initial best member
        self.current_best = self.population[0, :]
        self.current_best_idx = 0

        # Initial population history
        self.population_history = self.population

        # Initial check
        self._initial_check()

        # Iteration tolerance
        self.iteration_tolerance = 0

        # Search loop
        for i in range(self.n_iter):
            # Update current iteration
            self.current_iteration = i

            # Update HyDE-DF values
            idx01 = np.random.uniform(size=(self.pop_size, 3)) < 0.1
            idx02 = np.random.uniform(size=(self.pop_size, 1)) < 0.1

            self.f_weight[idx01] = (0.1 + np.random.uniform(size=(self.pop_size, 3)) * 0.9)[idx01]
            self.f_weight[~idx01] = self.f_weight_old[~idx01]

            self.f_cr[idx02] = np.random.uniform(size=(self.pop_size, 1))[idx02]
            self.f_cr[~idx02] = self.f_cr_old[~idx02]

            # Update population
            population, current_best = self._operator()

            # Boundary check
            population = self._update()
            self.population = population

            # Update population history
            self.population_history = np.vstack((self.population_history,
                                                 self.population))

            # Update best member

    # Execute full training loop of the algorithm
    def _execute(self):

        # TODO: have only this method as public and call the other methods from here
        # TODO: this will be the method that will be called from the outside and overriden by the user
        self._initial_check()
        self._generate_population()
        self._iterate()