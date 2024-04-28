import numpy as np


def convert_to_dictionary(a):
    temp_dictionary = {}

    if len(a.shape) == 3:
        for dim0 in np.arange(a.shape[0]):
            for dim1 in np.arange(a.shape[1]):
                for dim2 in np.arange(a.shape[2]):
                    temp_dictionary[(dim0 + 1, dim1 + 1, dim2 + 1)] = a[dim0, dim1, dim2]
    elif len(a.shape) == 2:
        for dim0 in np.arange(a.shape[0]):
            for dim1 in np.arange(a.shape[1]):
                temp_dictionary[(dim0 + 1, dim1 + 1)] = a[dim0, dim1]

    else:
        for dim0 in np.arange(a.shape[0]):
            temp_dictionary[(dim0 + 1)] = a[dim0]

    return temp_dictionary
