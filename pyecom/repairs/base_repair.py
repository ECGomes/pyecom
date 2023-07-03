# Base repair class to be extended

class BaseRepair(object):
    def __init__(self):
        pass

    def repair(self):
        raise NotImplementedError
