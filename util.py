"Module containing utility classes and functions."

import numpy as np

def unpack_r_data(s, n, k):
    """Unpacks an R data file containing a list of k matrices, each one representing the ordinal preferences of n agents over n goods. 

    Args:
        s (str): the name of the file containing the R data.
        n (int): the number of agents and goods in the R data.
        k (int): the number of matrices.


    Returns:
        three-dimensional np.array of integers where each row contains all values from 1 to n representing a list of agent ordinal preferences.
    """
    a = np.loadtxt(s, delimiter = ",")
    return np.reshape(a, (k,n,n))