# PROCSIM parser

from .base_parser import BaseParser

import pandas as pd


class PROCSIMParser(BaseParser):

    def __init__(self,
                 file_path: str):
        super().__init__(file_path)

        self.generator = None
        self.load = None

        return

    def parse(self):

        # Open the file
        data = pd.read_csv(self.file_path, sep=';')
        data.index = pd.to_datetime(data['Date'])
        data.drop('Date', axis=1, inplace=True)

        self.generator = data['Production']

        self.load = data['Demand']

        return
