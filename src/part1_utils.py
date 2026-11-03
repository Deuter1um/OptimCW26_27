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

def np_to_latex(A: np.ndarray, precision=3) -> str:
    A = np.asarray(A)
    def fmt(x):
        if isinstance(x, (float, np.floating)):
            return f"{x:.{precision}f}"
        return str(x)
    if A.ndim == 1:
        row = " & ".join(fmt(x) for x in A)
        return r"\begin{bmatrix}" + row + r"\end{bmatrix}"
    if A.ndim == 2:
        rows = [
            " & ".join(fmt(x) for x in row)
            for row in A
        ]
        return r"\begin{bmatrix}" + r" \\ ".join(rows) + r"\end{bmatrix}"
    raise ValueError("Input must be a 1D vector or 2D matrix.")