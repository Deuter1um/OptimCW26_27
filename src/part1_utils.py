import numpy as np


def Legendre_poly(k, x):
    if k == 0:
        return 1
    elif k == 1:
        return x
    else:
        return ((2* k + 1) * x * Legendre_poly(k, x) - k * Legendre_poly(k - 1, x)) / (k + 1)

def LSE(S: np.array, b):
    return np.linalg.inv(S.T @ S) @ S.T @ b