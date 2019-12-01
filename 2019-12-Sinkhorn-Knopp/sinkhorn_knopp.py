"""
Created on Friday 6 October 2017
Last update: -

@author: Michiel Stock
michielfmstock@gmail.com

Implementation of the Sinkhorn-Knopp algorithm for optimal transport
"""

import numpy as np


def compute_optimal_transport(
    M: np.ndarray, r: np.ndarray, c: np.ndarray, lam: float, epsilon=1e-5
):
    """
    Computes the optimal transport matrix and Slinkhorn distance using the
    Sinkhorn-Knopp algorithm

    Inputs:
        - M : cost matrix (n x m)
        - r : vector of marginals (n, )
        - c : vector of marginals (m, )
        - lam : strength of the entropic regularization
        - epsilon : convergence parameter

    Output:
        - P : optimal transport matrix (n x m)
        - dist : Sinkhorn distance
    """
    n, m = M.shape
    P = np.exp(-lam * M)
    P /= P.sum()
    u = np.zeros(n)
    # normalize this matrix
    while np.max(np.abs(u - P.sum(1))) > epsilon:
        u = P.sum(1)
        P *= (r / u).reshape((-1, 1))
        P *= (c / P.sum(0)).reshape((1, -1))
    return P, np.sum(P * M)
