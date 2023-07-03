# Base class for metaheuristics

class MetaheuristicBase(object):

    def __init__(self, n_iter: int, n_iter_tolerance: int,
                 pop_size: int, pop_dim: int):
        self.n_iter = n_iter
        self.n_iter_tolerance = n_iter_tolerance
        self.pop_size = pop_size
        self.pop_dim = pop_dim

    def execute(self):
        raise NotImplementedError

    def obj_fn(self):
        raise NotImplementedError

    def pop_fix(self):
        raise NotImplementedError
