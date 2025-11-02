# Base class for calculating priority

class Priority(object):

    def __init__(self, data):
        self.data = data
        self.priority = None

    def calculate_priority(self):
        raise NotImplementedError('Method calculate_priority() must be implemented.')
