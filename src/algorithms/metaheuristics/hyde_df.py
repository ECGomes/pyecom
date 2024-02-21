# HyDE-DF implementation that extends the MetaheuristicsBase class
import copy

from .base_metaheuristic import BaseMetaheuristic

import numpy as np
import copy


class HydeDF(BaseMetaheuristic):

    def __init__(self, n_iter: int, iter_tolerance: int, epsilon_tolerance: float,
                 pop_size: int, pop_dim: int,
                 lower_bound: np.ndarray, upper_bound: np.ndarray,
                 f_weight: float, f_cr: float, keep_population: bool = False):
        super().__init__(n_iter=n_iter,
                         iter_tolerance=iter_tolerance,
                         epsilon_tolerance=epsilon_tolerance,
                         pop_size=pop_size,
                         pop_dim=pop_dim)

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        # HyDE-DF adaptive parameters
        self.initial_f_weight = self.initial_f_weight_old = f_weight
        self.initial_f_cr = self.initial_f_cr = f_cr

        self.f_weight = self.f_weight_old = np.tile(f_weight, (self.pop_size, 3))
        self.f_cr = self.f_cr_old = np.tile(f_cr, (self.pop_size, 1))

        # Operator placeholder values
        self.fvr_rotation = np.arange(self.pop_size)

        # Placeholder values for population and population history
        self.population = []
        self.population_fitness = [np.inf for _ in range(self.pop_size)]

        self.population_old = []
        self.population_old_fitness = [np.inf for _ in range(self.pop_size)]

        self.population_history = []
        self.population_history_fitness = []

        # Flag to keep population history
        self.keep_population = keep_population

        # Placeholder values for best population member and it's index
        self.current_best = None
        self.current_best_idx = None
        self.current_best_fitness = None

        # Linear decrease placeholder value
        self.linear_decrease: float = 1.0

        # Current iteration value
        self.current_iteration: int = 0

        # Iteration tolerance
        self.current_tolerance: int = 0

    def _generate_population(self):
        """
        Method to generate an initial population
        :return: Generated initial population
        """

        return np.random.uniform(low=np.array(self.lower_bound,
                                              dtype=np.float64),
                                 high=np.array(self.upper_bound,
                                               dtype=np.float64),
                                 size=(self.pop_size,
                                       self.pop_dim))

    def _update(self):
        return np.clip(self.population, self.lower_bound, self.upper_bound)

    def _initial_check(self):

        if self.pop_size < 5:
            self.pop_size = 5
            print('Population size increase to minimal value of 5\n')

        if (self.initial_f_cr < 0.0) | (self.initial_f_cr > 1.0):
            self.f_cr = np.tile(0.5, (self.pop_size, 1))
            print('Crossover rate must be between 0.0 and 1.0 (inclusive). Set to 0.5\n')

        if self.n_iter < 0:
            self.n_iter = 200
            print('Negative iterations encountered. Set value to 200\n')

        # Set the current number of iterations and tolerance to 0
        self.current_iteration = 0
        self.iteration_tolerance = 0

        # Set the linear decrease value
        self.linear_decrease = self._calculate_linear_decrease()

    def _calculate_linear_decrease(self) -> float:
        return (self.n_iter - self.current_iteration) / self.n_iter

    def _operator(self):
        """
        Operator for the HyDE-DF crossover
        :return: None
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

        diff_var = population_best_member * (pop_02 + np.random.uniform(size=(self.pop_size, self.pop_dim)))
        diff_var = (diff_var - self.population_old) * pop_01 * exp_decrease

        # Prevent overflow
        diff_var = np.clip(diff_var, -1e+10, 1e+10)

        # Population update
        new_population = self.population_old + pop_00 * (np.array(pop_rot01) - np.array(pop_rot02)) + diff_var
        new_population = self.population_old * pop_mutated + new_population * pop_mutated_inverse

        self.population = new_population
        # self.population_history.append(self.population)

        return

    # Search loop
    def _update_hyde_params(self):
        """
        Method to run the search loop
        :return: None
        """

        idx01 = np.random.uniform(size=(self.pop_size, 3)) < 0.1
        idx02 = np.random.uniform(size=(self.pop_size, 1)) < 0.1

        self.f_weight[idx01] = (0.1 + np.random.uniform(size=(self.pop_size, 3)) * 0.9)[idx01]
        self.f_weight[~idx01] = self.f_weight_old[~idx01]

        self.f_cr[idx02] = np.random.uniform(size=(self.pop_size, 1))[idx02]
        self.f_cr[~idx02] = self.f_cr_old[~idx02]

        return

    # Initialization method to call before training
    def initialize(self):
        """
        Method to initialize the algorithm
        :return:
        """

        self._initial_check()

        # Set the placeholder variables to empty lists
        self.population = []
        self.population_fitness = [np.inf for _ in range(self.pop_size)]

        self.population_old = []
        self.population_old_fitness = [np.inf for _ in range(self.pop_size)]

        self.population_history = []
        self.population_history_fitness = []

        # Generate the initial population
        self.population = self.population_old = self._generate_population()

        return

    # Get the best population member
    def get_best(self):
        # Set best member
        self.current_best_idx = np.argmin(self.population_fitness)
        self.current_best = self.population[self.current_best_idx, :]
        self.current_best_fitness = self.population_fitness[self.current_best_idx]

        # Save to history
        self.population_history_fitness.append(self.current_best_fitness)

        return

    # Execute full training loop of the algorithm
    def update_population(self):
        """
        Method to run an iteration of the algorithm
        :return: None
        """

        # Update HyDE-DF values
        self._update_hyde_params()

        # Pre-update cleanup
        self.pre_update_cleanup()

        # Operator
        self._operator()

        return

    # Elitism selection
    def selection_mechanism(self):

        # Keep the best members
        for i in range(self.pop_size):
            if self.population_old_fitness[i] < self.population_fitness[i]:
                self.population[i, :] = self.population_old[i, :]
                self.population_fitness[i] = self.population_old_fitness[i]

        return

    def pre_update_cleanup(self):

        # Handle the history
        self.population_old = copy.deepcopy(self.population)
        self.population_old_fitness = copy.deepcopy(self.population_fitness)

        # Save to history
        if self.keep_population:
            self.population_history.append(self.population)

    def post_update_cleanup(self):

        # Elitism selection
        self.selection_mechanism()

        # Update best member
        self.get_best()

        # Save best parameters
        self.f_weight_old[self.current_best_idx, :] = self.f_weight[self.current_best_idx, :]
        self.f_cr_old[self.current_best_idx] = self.f_cr[self.current_best_idx]

    # Post update cleanup
    def check_stopping_criteria(self) -> bool:
        """
        Method to check the stopping criteria
        :return: True if the stopping criteria is met, False otherwise
        """

        if abs(np.sum([-self.current_best_fitness,
                       self.population_fitness[self.current_best_idx]])) < self.epsilon_tolerance:
            self.current_best = self.population[self.current_best_idx, :]
            self.current_best_fitness = self.population_fitness[self.current_best_idx]
            self.current_tolerance = 0
        else:
            self.current_tolerance += 1

        if self.current_tolerance >= self.iter_tolerance:
            return True

        return False
