# Defines the base resource class for all resources
# Every resource should inherit from this class
# Has the following properties:
#   - name: str
#   - value: np.array
#   - lb: np.array
#   - ub: np.array
#   - cost: np.array

import numpy as np


class BaseResource:
    """
    Base class for all resources.
    Name, value, lb, ub, and cost are required.
    Name: str
    Value: NumPy array
    Lower bound: NumPy array
    Upper bound: NumPy array
    Cost: NumPy array
    """

    def __init__(self,
                 name: str,
                 value: np.array,
                 lower_bound: np.array,
                 upper_bound: np.array,
                 cost: np.array):
        self.name = name
        self.value = value
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.cost = cost

    def __repr__(self):
        return f'{self.name}'

    def __str__(self):
        return f'{self.name}'

    def __eq__(self, other):
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __len__(self):
        return len(self.value)

    def __contains__(self, item):
        return item in self.value

    def __add__(self, other):
        return self.value + other.value

    def __sub__(self, other):
        return self.value - other.value

    def __mul__(self, other):
        return self.value * other.value

    def __truediv__(self, other):
        return self.value / other.value

    def __floordiv__(self, other):
        return self.value // other.value

    def __mod__(self, other):
        return self.value % other.value

    def __divmod__(self, other):
        return divmod(self.value, other.value)

    def __pow__(self, other):
        return self.value ** other.value

    def ravel(self):
        return self.value.ravel()

    def shape(self):
        return self.value.shape
