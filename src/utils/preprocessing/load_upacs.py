# Auxiliar function to load UPAC data.
# UPAC data has two columns: 'pv' and 'load'.

import glob
import pandas as pd


def load_upac_pv(filepath: str, resample: str = 'H') -> pd.DataFrame:
    """
    Load UPAC PV and Load data from a single CSV file.
    """

    temp = pd.read_csv(filepath, index_col=0, parse_dates=True)
    temp = temp.resample(resample).mean()

    # Convert W to kW
    temp = temp / 1000

    # Set any negative values to 0
    temp[temp < 0] = 0

    # Get only the full years (2019 and 2020)
    temp = temp.loc['2019':'2020']

    # Fill any missing values with zeros
    temp = temp.fillna(0)

    return temp


def load_multiple_upacs_pv(directory: str, resample: str = 'H') -> dict:
    """
    Load UPAC data from multiple CSV files in a directory.
    """

    temp = {}

    for file in glob.glob(directory):
        filename = file.split('/')[-1].split('_')[0].split('upac')[1]
        temp[filename] = load_upac_pv(file, resample)

    return temp
