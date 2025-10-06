import numpy as np

def fuse_information(local_estimates):
    """Federated algebra: given a list of (x_m, P_m), return fused (x, P).
    We use information form: Y = P^{-1}, y = Y x, and weight by trace(Y_m).
    """
    Ys, ys, wsum = [], [], 0.0
    for (x, P) in local_estimates:
        Y = np.linalg.pinv(P)
        w = np.trace(Y) + 1e-8
        Ys.append(w * Y)
        ys.append(w * (Y @ x))
        wsum += w
    Yg = sum(Ys) / max(1e-8, wsum)
    yg = sum(ys) / max(1e-8, wsum)
    Pg = np.linalg.pinv(Yg + 1e-8 * np.eye(Yg.shape[0]))
    xg = Pg @ yg
    return xg, Pg
