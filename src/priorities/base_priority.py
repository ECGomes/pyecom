# Base class for calculating priority
import numpy as np


class Priority(object):

    def __init__(self, data):
        self.data = data
        self.priority = None

    def calculate_priority(self):
        raise NotImplementedError('Method calculate_priority() must be implemented.')

    # Calculate the priority of the
    # data using the maximum value of each resource
    def calculate_max_priority(self):
        raise NotImplementedError('Method calculate_max_priority() must be implemented.')

    # Calculate the priority of the resource group
    def calculate_group_priority(self):
        raise NotImplementedError('Method calculate_group_priority() must be implemented.')

    # Calculate the priority of individual resources within a group
    def calculate_resource_size_priority(self):
        raise NotImplementedError('Method calculate_resource_size_priority() must be implemented.')