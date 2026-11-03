import numpy as np


def Legendre_poly(k, x):
    if k == 0:
        return 1
    elif k == 1:
        return x
    else:
        return ((2 * k - 1) * x * Legendre_poly(k - 1, x) - (k - 1) * Legendre_poly(k - 2, x)) / k

def naive_LSE(S: np.array, b):
    return np.linalg.inv(S.T @ S) @ S.T @ b

def V_estimate(theta: np.array, x):
    n = len(theta)
    p_vec = np.array([Legendre_poly(k, x) for k in range(n)])
    return theta @ p_vec