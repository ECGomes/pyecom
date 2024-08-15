# MGO implementation that extends the MetaheuristicsBase class
# Attempt at reproducing the code available in:
# https://www.mathworks.com/matlabcentral/fileexchange/118680-mountain-gazelle-optimizer
# Publication link:
# https://www.sciencedirect.com/science/article/abs/pii/S0965997822001831

from .base_metaheuristic import BaseMetaheuristic

import numpy as np
import copy


class MGO(BaseMetaheuristic):

    def __init__(self, n_iter: int, iter_tolerance: int, epsilon_tolerance: float,
                 pop_size: int, pop_dim: int,
                 lower_bound: np.ndarray, upper_bound: np.ndarray,
                 keep_population: bool = False):
        super().__init__(n_iter=n_iter,
                         iter_tolerance=iter_tolerance,
                         epsilon_tolerance=epsilon_tolerance,
                         pop_size=pop_size,
                         pop_dim=pop_dim)

        # Initialize bounds
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        # Flag to keep population history
        self.keep_population = keep_population
        self.population_history = []
        self.population_history_fitness = []

        # Placeholder values for population and population history
        self.population = []
        self.population_fitness = [np.inf for _ in range(self.pop_size)]
        self.population_old = []
        self.population_old_fitness = [np.inf for _ in range(self.pop_size)]

        # Placeholder values for best population member and its index
        self.current_best = None
        self.current_best_idx = None
        self.current_best_fitness = None

        # Current iteration value
        self.current_iteration: int = 0

        # Iteration tolerance
        self.current_tolerance: int = 0

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

    def _initial_check(self):

        if self.pop_size < 5:
            self.pop_size = 5
            print('Population size increase to minimal value of 5\n')

        if self.n_iter < 0:
            self.n_iter = 200
            print('Negative iterations encountered. Set value to 200\n')

        # Set the current number of iterations and tolerance to 0
        self.current_iteration = 0
        self.iteration_tolerance = 0

        return

    # Population initialization method
    def _generate_population(self):
        """
        Generate the initial population for the algorithm
        """

        # Initialize a population with shape (pop_size, pop_dim)
        temp_pop = self.rng.uniform(size=(self.pop_size, self.pop_dim))

        # Scale the population to the bounds of the problem
        return temp_pop * (self.upper_bound - self.lower_bound) + self.lower_bound

    # Update population sorting method
    def _update(self):
        """
        Update the population and sort it based on fitness
        """

        # Sort the population based on fitness
        sorted_idx = np.argsort(self.population_fitness)
        self.population = self.population[sorted_idx, :][:self.pop_size]
        self.population_fitness = np.array(self.population_fitness)[sorted_idx][:self.pop_size].tolist()

        return

    def update_population(self):
        """
        Method to run an iteration of the algorithm
        :return: None
        """

        # Pre-update cleanup
        self.pre_update_cleanup()

        # Operator
        self._operator()

        return

    def get_best(self):
        # Set best member
        self.current_best_idx = np.argmin(self.population_fitness)
        self.current_best = self.population[self.current_best_idx, :]
        self.current_best_fitness = self.population_fitness[self.current_best_idx]

        # Save to history
        self.population_history_fitness.append(self.current_best_fitness)

        return

    # Original boundary checking method
    def _boundary_check(self, population: np.ndarray):
        """
        Check if the population is within the bounds of the problem
        """

        flag_ub = population > self.upper_bound
        flag_lb = population < self.lower_bound

        temp_pop = population * ~(flag_ub | flag_lb) + self.upper_bound * flag_ub + self.lower_bound * flag_lb

        return temp_pop

    def _coefficient_vector(self, current_iter: int):
        """
        Generate the coefficient vector for the algorithm
        """

        a2 = current_iter * ((-1) / self.n_iter) - 1
        u_vector = self.rng.normal(size=self.pop_dim)
        v_vector = self.rng.normal(size=self.pop_dim)

        coefficient = np.zeros(shape=(4, self.pop_dim))
        coefficient[0, :] = self.rng.uniform(low=0, high=1, size=self.pop_dim)
        coefficient[1, :] = (a2 + 1) * self.rng.uniform(size=1)
        coefficient[2, :] = a2 * self.rng.normal(size=(1, self.pop_dim))
        coefficient[3, :] = u_vector * np.power(v_vector, 2) *\
                            np.cos(2 * self.rng.uniform(size=1) * u_vector)

        return coefficient

    def _operator(self):
        """
        Operator method for the algorithm
        """

        for member in np.arange(self.pop_size):
            random_candidates = self.rng.permutation(np.arange(self.pop_size))[:int(self.pop_size / 3)]

            # Get the new candidates
            pop_idx = self.rng.integers(low=int(self.pop_size / 3), high=self.pop_size, size=1)

            # MATLAB code does floor(rand) which ALWAYS returns 0, and ceil(rand) which ALWAYS returns 1.
            # Modified to normal distribution instead of uniform
            m = self.population[pop_idx] * np.floor(self.rng.normal()) + \
                np.mean(self.population[random_candidates], axis=0) * np.ceil(self.rng.normal())

            # Calculate coefficients
            coefficients = self._coefficient_vector(self.current_iteration + 1)

            # Calculate A and D vectors
            a_vector = self.rng.normal(size=self.pop_dim) * np.exp(2 - self.current_iteration + 1 * (2 / self.n_iter))
            d_vector = (np.abs(self.population[member]) + \
                        np.abs(self.current_best)) * (2 * self.rng.uniform() - 1)

            # Update location
            gazelles = np.zeros(shape=(4, self.pop_dim))
            gazelles[1, :] = self.current_best - \
                             np.abs((self.rng.integers(1, 3) * m - self.rng.integers(1, 3) * \
                                     self.population[member]) * a_vector) * \
                             coefficients[self.rng.integers(0, 4), :]
            gazelles[2, :] = (m + coefficients[self.rng.integers(0, 4), :]) + \
                             (self.rng.integers(1, 3) * self.current_best - self.rng.integers(1, 3) * \
                              self.population[self.rng.integers(0, self.pop_size)]) * coefficients[self.rng.integers(0, 4), :]
            gazelles[3, :] = self.population[member] - d_vector + \
                             (self.rng.integers(1, 3) * self.current_best - self.rng.integers(1, 3) * m) * \
                             coefficients[self.rng.integers(0, 4), :]

            # Boundary check
            new_gazelles = self._boundary_check(gazelles)

            self.population = np.vstack((self.population, new_gazelles))
            for _ in np.arange(4):
                self.population_fitness.append(np.inf)

        return

    def pre_update_cleanup(self):

        # Handle the history
        self.population_old = copy.deepcopy(self.population)
        self.population_old_fitness = copy.deepcopy(self.population_fitness)

        # Save to history
        if self.keep_population:
            self.population_history.append(self.population)

    def post_update_cleanup(self):
        """
        Post-update cleanup method for the algorithm
        """

        # Update the population
        self._update()

        # Update the best member
        self.get_best()

        return

    def check_termination(self) -> bool:
        """
        Method to check the stopping criteria
        :return: True if the stopping criteria is met, False otherwise
        """

        if self.current_best_fitness < self.population_old_fitness[np.argmin(self.population_old_fitness)]:
            if self.current_best_fitness \
                    + self.epsilon_tolerance < self.population_old_fitness[np.argmin(self.population_old_fitness)]:
                self.current_tolerance += 1

            else:
                self.current_tolerance = 0

        else:
            self.current_tolerance = 0

        if self.current_tolerance >= self.iter_tolerance:
            return True

        return False
