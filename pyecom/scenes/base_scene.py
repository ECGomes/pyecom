# Base scene to be extended

class BaseScene(object):

    def __init__(self, name: str, components: dict):
        self.name = name
        self.components = components

    def initialize(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError
