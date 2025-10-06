import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Dict

@dataclass
class SynthConfig:
    n_tasks: int
    horizon: int
    state_dim: int
    n_sensors: int
    dt: float
    q: float
    r: float
    seed: int = 0

class SyntheticWorld:
    """A compact surrogate for the construction process dynamics:
    - latent 'true' state evolves with simple linear dynamics + noise
    - each sensor observes a nonlinear projection + noise
    """
    def __init__(self, cfg: SynthConfig):
        self.cfg = cfg
        rs = np.random.RandomState(cfg.seed)
        self.A = np.eye(cfg.state_dim) + 0.01 * rs.randn(cfg.state_dim, cfg.state_dim)
        # Ensure stability
        self.A *= 0.98 / max(1.0, np.linalg.norm(self.A, 2))
        self.B = 0.05 * rs.randn(cfg.state_dim, cfg.state_dim//4)
        self.C = [rs.randn(cfg.state_dim//2, cfg.state_dim) for _ in range(cfg.n_sensors)]
        self.state = rs.randn(cfg.state_dim) * 0.1
        self.rs = rs

    def step(self, u: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        cfg = self.cfg
        # process noise
        w = np.sqrt(cfg.q) * self.rs.randn(cfg.state_dim)
        self.state = self.A @ self.state + self.B[:, :u.shape[0]] @ u + w
        # observations: each sensor sees a projection + tanh nonlinearity + noise
        obs = {}
        for k, Ck in enumerate(self.C):
            z_lin = Ck @ self.state
            v = np.sqrt(cfg.r) * self.rs.randn(z_lin.shape[0])
            obs[f"s{k}"] = np.tanh(z_lin) + v
        return self.state.copy(), obs

def generate(cfg: SynthConfig) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    world = SyntheticWorld(cfg)
    T = cfg.horizon
    # control is zero for data generation; RL will provide later
    u = np.zeros(cfg.state_dim // 4)
    X, Zs = [], {f"s{k}": [] for k in range(cfg.n_sensors)}
    for t in range(T):
        x, obs = world.step(u)
        X.append(x)
        for k in range(cfg.n_sensors):
            Zs[f"s{k}"].append(obs[f"s{k}"])
    X = np.array(X)
    X_df = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    Zdfs = {k: pd.DataFrame(np.array(v), columns=[f"z_{k}_{i}" for i in range(np.array(v).shape[1])]) for k, v in Zs.items()}
    return X_df, Zdfs
