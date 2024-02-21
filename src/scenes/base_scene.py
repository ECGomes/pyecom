# Base scene to be extended

class BaseScene:

    def __init__(self, name: str, components: dict):
        self.name = name
        self.components = components

    def initialize(self):
        raise NotImplementedError

    def evaluate(self, x):
        raise NotImplementedError

    def repair(self, x):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError
