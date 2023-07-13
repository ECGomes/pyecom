# Base definition of neural networks

from ..base_algorithm import BaseAlgorithm


class BaseNetwork(BaseAlgorithm):

    def __init__(self):
        return

    def preprocess(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError
