# Auxiliary function receives a time series, upper and lower bounds and returns a probabilistic timeseries
from typing import Union, Tuple, Any

import numpy as np
from numpy import ndarray


def to_probabilistic(data: np.ndarray,
                     lcb: float = 0.1,
                     ucb: float = 0.9,
                     method: str = 'gaussian',
                     force_positive: bool = True) -> Union[Union[tuple[Any, Any], ndarray, int, float, complex], Any]:
    """
    to_probabilistic auxiliary function receives a timeseries, upper and lower bounds and
    returns a probabilistic range of values
    :param data: time series
    :param lcb: lower confidence bound
        - If method is 'simple', then lcb is the data - data * lcb
        - If the method is gaussian, it is unused
    :param ucb: upper confidence bound
        - If method is 'simple', then ucb is the data + data * (1.0 - ucb)
        - If the method is gaussian, it is unused
    :param method: method to use for probabilistic timeseries. Default is gaussian. Available methods are:
        - tube
        - gaussian
        - uniform
    :param force_positive: if True, then the probabilistic timeseries will be forced to be positive
    :return: probabilistic timeseries
    """

    # If the ucb is larger than 1, force it to be 1
    if ucb > 1.0:
        ucb = 1.0

    if lcb < 0.0:
        lcb = 0.0

    if method == 'tube':
        # Simple method to create a "tube" around the data
        temp_lcb = data - data * lcb
        temp_ucb = data + data * ucb

        if force_positive:
            temp_lcb[temp_lcb < 0] = 0
            temp_ucb[temp_ucb < 0] = 0

        return temp_lcb, temp_ucb

    elif method == 'random':
        # Create a random number between data and bounds for each data point
        rand_lower = np.random.random_sample(len(data))
        rand_upper = np.random.random_sample(len(data))
        temp_lcb = data - data * (1.0 - lcb) * rand_lower
        temp_ucb = data + data * ucb * rand_upper

        if force_positive:
            temp_lcb[temp_lcb < 0] = 0
            temp_ucb[temp_ucb < 0] = 0

        return temp_lcb, temp_ucb

    elif method == 'gaussian':
        sample_mean = np.mean(data)
        sample_std = np.std(data)

        # If the standard deviation is 0, then the data is constant
        if sample_std == 0:
            temp_sample = np.ones(len(data)) * sample_mean
            if force_positive:
                temp_sample[temp_sample < 0] = 0
            return temp_sample

        # If the standard deviation is not 0, then the data is not constant
        # We can use the normal distribution with center in the mean and standard deviation
        else:
            temp_sample = np.random.normal(sample_mean, sample_std, len(data))
            if force_positive:
                temp_sample[temp_sample < 0] = 0
            return temp_sample

    elif method == 'uniform':
        return np.random.uniform(lcb, ucb, len(data))
    else:
        raise ValueError('Method not supported')
