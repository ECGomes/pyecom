# DO implementation that extends the MetaheuristicsBase class
# Attempt at reproducing the code available in:
# https://www.mathworks.com/matlabcentral/fileexchange/114680-dandelion-optimizer
# Publication link:
# https://www.sciencedirect.com/science/article/abs/pii/S0952197622002305

from .base_metaheuristic import BaseMetaheuristic

import numpy as np
import math
import copy


class DO(BaseMetaheuristic):

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

    def _generate_population(self) -> np.ndarray:
        """
        Method to generate an initial population
        :return: Generated initial population
        """

        # Positions(:,i)=rand(Popsize,1).*(ub_i-lb_i)+lb_i;  % Generation of random population

        return self.rng.uniform(size=(self.pop_size, self.pop_dim)) \
            * (self.upper_bound - self.lower_bound) + self.lower_bound

    def _boundary_check(self, population: np.ndarray) -> np.ndarray:
        """
        Method to check if the population is within the bounds
        :param population: Population to check
        :return: Population within bounds
        """

        return np.clip(population, self.lower_bound, self.upper_bound)

    def _update(self):

        # Sort the population based on fitness
        sorted_idx = np.argsort(self.population_fitness)
        self.population = self.population[sorted_idx, :][:self.pop_size]
        self.population_fitness = np.array(self.population_fitness)[sorted_idx][:self.pop_size].tolist()

        return

    def pre_update_cleanup(self):

        # Handle the history
        self.population_old = copy.deepcopy(self.population)
        self.population_old_fitness = copy.deepcopy(self.population_fitness)

        # Save to history
        if self.keep_population:
            self.population_history.append(self.population)

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

    def post_update_cleanup(self):
        """
        Post-update cleanup method for the algorithm
        """

        # Update the population
        self._update()

        # Update the best member
        self.get_best()

        return

    def _operator(self):
        """
        Operator for the DO algorithm
        :return:
        """

        # Rising stage
        beta = self.rng.normal(size=(self.pop_size, self.pop_dim))
        alpha = self.rng.uniform() * ((1 / np.power(self.n_iter, 2)) * \
                                      np.power(self.current_iteration, 2) - \
                                      2 / self.n_iter * self.current_iteration + 1)

        a = 1 / (np.power(self.n_iter, 2) - 2 * self.n_iter + 1)
        b = -2 * a
        c = 1 - a - b
        k = 1 - self.rng.uniform() * (c + a * np.power(self.current_iteration, 2) + b * self.current_iteration)

        # Update the population (rising stage)
        if self.rng.normal() < 1.5:
            lambda_val = np.abs(self.rng.normal(size=(self.pop_size, self.pop_dim)))
            theta = (2 * self.rng.uniform(size=(self.pop_size, 1)) - 1) * np.pi
            row = 1 / np.exp(theta)
            vx = row * np.cos(theta)
            vy = row * np.sin(theta)
            new = self.rng.uniform(size=(self.pop_size, self.pop_dim)) * \
                  (self.upper_bound - self.lower_bound) + self.lower_bound

            self.population = self.population + np.multiply(alpha * vx * vy,
                                                            np.log(lambda_val) * (new - self.population))

        else:
            self.population = np.multiply(self.population, k)

        # Boundary check
        self.population = self._boundary_check(self.population)

        # Decline stage
        # Calculate the Lévy flight steps
        step_length = self._levy_flight(self.pop_size, self.pop_dim, 1.5)
        elite = np.tile(self.current_best, (self.pop_size, 1))

        new_pop = elite + step_length * alpha * (elite - self.population) * \
                  (2 * (self.current_iteration + 1) / self.n_iter)

        self.population = new_pop

        # Boundary check
        self.population = self._boundary_check(self.population)

        return

    # Lévy flight method
    def _levy_flight(self, steps: int, dims: int, power: float):
        """
        Lévy flight method
        :param steps: Number of steps
        :param dims: Number of dimensions
        :param power: Power law index (1 < beta < 2)
        :return: Lévy flight steps in dims dimensions
        """

        # Check if beta is within the range
        power = np.clip(power, 1.01, 1.99)

        # Calculate numerator and denominator
        num = math.gamma(1 + power) * np.sin(np.pi * power / 2)
        den = math.gamma((1 + power) / 2) * power * np.power(2, (power - 1) / 2)

        # Calculate standard deviation
        sigma_u = np.power(num / den, 1 / power)
        u = self.rng.normal(0, np.power(sigma_u, 2), (steps, dims))

        v = self.rng.normal(0, 1, (steps, dims))
        z = u / np.power(np.abs(v), 1 / power)

        return z

    def check_termination(self) -> bool:
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