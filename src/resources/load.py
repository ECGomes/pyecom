# Extends the BaseResource class to provide a load resource

import numpy as np

from src.resources.base_resource import BaseResource


class Load(BaseResource):

    def __init__(self,
                 name: str,
                 value: np.array,
                 lower_bound: np.array,
                 upper_bound: np.array,
                 cost: np.array,
                 cost_reduce: np.array,
                 cost_cut: np.array,
                 cost_ens: np.array):
        super().__init__(name, value, lower_bound, upper_bound, cost)

        self.cost_reduce = cost_reduce
        self.cost_cut = cost_cut
        self.cost_ens = cost_ens

        self.is_active = np.zeros(self.value.shape)

        self.load_cut = np.zeros(self.value.shape)
        self.load_reduce = np.zeros(self.value.shape)
        self.load_ens = np.zeros(self.value.shape)
