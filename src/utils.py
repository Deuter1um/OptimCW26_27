import numpy as np
from IPython.display import clear_output


def Legendre_poly(k, x):
    if k == 0:
        return 1
    elif k == 1:
        return x
    else:
        return ((2 * k - 1) * x * Legendre_poly(k - 1, x) - (k - 1) * Legendre_poly(k - 2, x)) / k


def naive_LSE(S: np.ndarray, b):
    return np.linalg.inv(S.T @ S) @ S.T @ b


def V_estimate(theta: np.ndarray, x):
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


def get_condition_number(A: np.ndarray):
    eigenvalues = np.linalg.eigvals(A)
    return eigenvalues.max() / eigenvalues.min()


def L(eps, t):
    return np.where(
        np.abs(t) <= eps,
        0.5 * t ** 2,
        eps * (np.abs(t) - 0.5 * eps)
    )


def dL(eps, t):
    return np.select(
        [
            t < -eps,
            np.abs(t) <= eps,
            t > eps
        ],
        [
            -eps,
            t,
            eps
        ]
    )


def F(eps, theta: np.ndarray, A: np.ndarray, b: np.ndarray):
    l = L(eps, A @ theta - b)
    return l @ np.ones_like(l)


def F_grad(eps, theta: np.ndarray, A: np.ndarray, b: np.ndarray):
    return A.T @ dL(eps, A @ theta - b)


def F_gd(A: np.ndarray, b: np.ndarray, n=8, eps=1e-3, threshold=1e-9, max_iter=1e6, rand_seed=42):
    evalues = np.linalg.eigvals(A.T @ A)
    sigma_max_sqr = evalues.max()

    t = 1 / sigma_max_sqr
    eps = eps
    rng = np.random.default_rng(seed=rand_seed)
    theta_iter = rng.normal(size=n)

    threshold = threshold
    max_iter = max_iter
    loss = F(eps, theta_iter, A, b)
    last_loss = float("inf")
    iter_cnt = 0

    while np.abs(loss - last_loss) >= threshold and iter_cnt < max_iter:
        last_loss = loss.copy()
        theta_iter -= t * F_grad(eps, theta_iter, A, b)
        loss = F(eps, theta_iter, A, b)
        iter_cnt += 1
        # clear_output()
        # print(f"iteration:{iter_cnt} | loss = {loss}")

    return iter_cnt, theta_iter, loss


def get_mse(theta: np.ndarray, X_val: np.ndarray, Y_val: np.ndarray):
    V_val_estimate = np.array([V_estimate(theta, x) for x in X_val])
    return np.mean((V_val_estimate - Y_val) ** 2)