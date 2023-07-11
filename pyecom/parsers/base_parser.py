# Base Parser to be used by the package
#

class BaseParser:

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.data = None

    def parse(self):
        raise NotImplementedError('Method parse() must be implemented.')
