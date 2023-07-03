# Extends the BaseResource class to provide a generator resource

from pyecom.resources.base_resource import BaseResource
import numpy as np


class Generator(BaseResource):
    def __init__(self,
                 name: str,
                 value: np.array,
                 lb: np.array,
                 ub: np.array,
                 cost: np.array,
                 renewable: np.array):
        super().__init__(name, value, lb, ub, cost)

        self.renewable = renewable
