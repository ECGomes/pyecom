# Base class for metaheuristics

from ..base_algorithm import BaseAlgorithm


class BaseMetaheuristic(BaseAlgorithm):

    def __init__(self, n_iter: int, iter_tolerance: int, epsilon_tolerance: float,
                 pop_size: int, pop_dim: int):
        self.n_iter = n_iter
        self.iter_tolerance = iter_tolerance
        self.epsilon_tolerance = epsilon_tolerance
        self.pop_size = pop_size
        self.pop_dim = pop_dim
