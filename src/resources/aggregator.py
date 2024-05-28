# Extends the BaseResource class to provide an aggregator resource

import numpy as np
from src.resources.base_resource import BaseResource

from typing import Union


class Aggregator(BaseResource):

    def __init__(self,
                 name: str,
                 value: Union[np.array, float],
                 lower_bound: np.array,
                 upper_bound: np.array,
                 cost: np.array,
                 imports: np.array,
                 exports: np.array,
                 import_cost: np.array,
                 export_cost: np.array,
                 import_max: np.array,
                 export_max: np.array):
        super().__init__(name, value, lower_bound, upper_bound, cost)

        self.imports = imports
        self.exports = exports

        self.import_max = import_max
        self.export_max = export_max

        self.import_cost = import_cost
        self.export_cost = export_cost
