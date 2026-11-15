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
    return np.clip(t, -eps, eps)


def F(eps, theta: np.ndarray, A: np.ndarray, b: np.ndarray):
    l = L(eps, A @ theta - b)
    return l @ np.ones_like(l)


def F_grad(eps, theta: np.ndarray, A: np.ndarray, b: np.ndarray):
    return A.T @ dL(eps, A @ theta - b)


def F_gd(A: np.ndarray, b: np.ndarray, theta_init=None, n=8, eps=1e-3, threshold=1e-6, max_iter=1e6, rand_seed=42, return_loss_history=False):
    evalues = np.linalg.eigvals(A.T @ A)
    sigma_max_sqr = evalues.max()

    t = 1 / sigma_max_sqr
    if theta_init is None:
        rng = np.random.default_rng(seed=rand_seed)
        theta_iter = rng.normal(size=n)
    else:
        theta_iter = theta_init.copy()
    loss_history = []

    loss = F(eps, theta_iter, A, b)
    iter_cnt = 0

    while np.linalg.norm(F_grad(eps, theta_iter, A, b)) >= threshold and iter_cnt < max_iter:
        theta_iter -= t * F_grad(eps, theta_iter, A, b)
        loss = F(eps, theta_iter, A, b)
        loss_history.append(loss)
        iter_cnt += 1
        # clear_output()
        # print(f"iteration:{iter_cnt} | loss = {loss}")

    loss_history = np.array(loss_history)

    if return_loss_history:
        return iter_cnt, theta_iter, loss, loss_history
    else:
        return iter_cnt, theta_iter, loss


def get_mse(theta: np.ndarray, X_val: np.ndarray, Y_val: np.ndarray):
    V_val_estimate = np.array([V_estimate(theta, x) for x in X_val])
    return np.mean((V_val_estimate - Y_val) ** 2)


def hessian(p, anchors, r):
    H = np.zeros((2, 2))
    for s, ri in zip(anchors, r):
        d = p - s
        d = d.reshape(-1, 1)
        H += 2 * d @ d.T
        H += (d.T @ d - ri**2) * np.eye(2)
    return H


def g(t, p: np.ndarray, anchors: np.ndarray, ranges: np.ndarray):
    return 0.25 * np.sum((np.linalg.norm(p - anchors, axis=1) ** 2 - ranges[t] ** 2) ** 2)


def g_grad(t, p: np.ndarray, anchors: np.ndarray, ranges: np.ndarray):
    r = np.linalg.norm(p - anchors, axis=1) ** 2 - ranges[t] ** 2
    return np.sum(r[:, None] * (p - anchors), axis=0)


def g_gd(t, p_init: np.ndarray, anchors: np.ndarray, ranges: np.ndarray, threshold=1e-6, max_iter=1e6):
    step = 1e-3
    p_iter = p_init.copy()
    loss = g(t, p_iter, anchors, ranges)
    iter_cnt = 0

    while np.linalg.norm(g_grad(t, p_iter, anchors, ranges)) >= threshold and iter_cnt < max_iter:
        p_iter -= step * g_grad(t, p_iter, anchors, ranges)
        loss = g(t, p_iter, anchors, ranges)
        iter_cnt += 1

    return iter_cnt, p_iter, loss


def valid(traj: np.ndarray, checkpoints: np.ndarray):
    mask = np.arange(9, 60, 10)
    return np.sqrt(1.0 / 6 * np.sum((checkpoints - traj[mask]) ** 2))


def D_matrix(T) -> np.ndarray:
    D = np.zeros(((T-1), T))
    D[np.arange(T-1), np.arange(T-1)] = 1
    D[np.arange(T-1), np.arange(1, T)] = -1
    return D


def G(lamb, P: np.ndarray, anchors: np.ndarray, ranges: np.ndarray):
    T = P.shape[0]
    D = D_matrix(T)
    return np.sum(np.array([g(t, p, anchors, ranges) for t, p in enumerate(P)])) + lamb / 2 * np.linalg.norm(D @ P, ord="fro")


def G_grad(lamb, P: np.ndarray, anchors: np.ndarray, ranges: np.ndarray):
    T = P.shape[0]
    D = D_matrix(T)
    return np.array([g_grad(t, p, anchors, ranges) for t, p in enumerate(P)]) + lamb * D.T @ D @ P


def G_gd(lamb, P_init: np.ndarray, anchors: np.ndarray, ranges: np.ndarray, threshold=1e-6, max_iter=1e6):
    step = 1 / (1000 + 4 * lamb)
    P_iter = P_init.copy()
    loss = G(lamb, P_iter, anchors, ranges)
    iter_cnt = 0

    while np.linalg.norm(G_grad(lamb, P_iter, anchors, ranges), ord="fro") >= threshold and iter_cnt < max_iter:
        P_iter -= step * G_grad(lamb, P_iter, anchors, ranges)
        loss = G(lamb, P_iter, anchors, ranges)
        iter_cnt += 1

    return iter_cnt, P_iter, loss