# Base class for metaheuristics
import numpy as np

from ..base_algorithm import BaseAlgorithm


class BaseMetaheuristic(BaseAlgorithm):

    def __init__(self, n_iter: int, iter_tolerance: int, epsilon_tolerance: float,
                 pop_size: int, pop_dim: int, seed=None):
        super().__init__()
        self.n_iter = n_iter
        self.iter_tolerance = iter_tolerance
        self.epsilon_tolerance = epsilon_tolerance
        self.pop_size = pop_size
        self.pop_dim = pop_dim
        self.rng = np.random.default_rng(seed)

    def initialize(self):
        raise NotImplementedError

    def pre_update_cleanup(self):
        raise NotImplementedError

    def post_update_cleanup(self):
        raise NotImplementedError

    def update_population(self):
        raise NotImplementedError

    def check_termination(self):
        raise NotImplementedError
