# Create a resource that serves for both import and export

import numpy as np

from src.resources.base_resource import BaseResource


class BinaryResource(BaseResource):

    def __init__(self,
                 name: str,
                 value: np.array,
                 lower_bound: np.array,
                 upper_bound: np.array,
                 cost: np.array,
                 is_active: np.array):
        super().__init__(name, value, lower_bound, upper_bound, cost)

        self.is_active = is_active
