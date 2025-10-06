import numpy as np

class EKF:
    def __init__(self, state_dim, meas_dim, dt=1.0, q=0.02, r=0.05):
        self.n = state_dim
        self.m = meas_dim
        self.dt = dt
        self.Q = q * np.eye(self.n)
        self.R = r * np.eye(self.m)
        self.x = np.zeros(self.n)
        self.P = np.eye(self.n)

        # simple constant-velocity-like linearization; here: identity drift
        self.F = np.eye(self.n)
        self.H = np.zeros((self.m, self.n))

        # random projection for linearization of nonlinear h(x)
        rng = np.random.RandomState(0)
        idx = rng.choice(self.n, size=self.m, replace=False)
        self.H[np.arange(self.m), idx] = 1.0

    def f(self, x):
        return self.F @ x  # identity dynamics surrogate

    def h(self, x):
        # Nonlinear observation; EKF linearizes around x_k
        return np.tanh(self.H @ x)

    def predict(self):
        self.x = self.f(self.x)
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        # Jacobian of h around current x (approximate with diag(1 - tanh^2))
        Hx = self.H
        S = Hx @ self.P @ Hx.T + self.R
        K = self.P @ Hx.T @ np.linalg.inv(S)
        y = z - self.h(self.x)
        self.x = self.x + K @ y
        self.P = (np.eye(self.n) - K @ Hx) @ self.P
        return self.x, self.P
