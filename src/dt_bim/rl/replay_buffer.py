import numpy as np
from collections import deque

class ReplayBuffer:
    def __init__(self, state_dim, action_dim, capacity=200000):
        self.s = np.zeros((capacity, state_dim), dtype=np.float32)
        self.a = np.zeros((capacity, action_dim), dtype=np.float32)
        self.r = np.zeros((capacity, 1), dtype=np.float32)
        self.s2 = np.zeros((capacity, state_dim), dtype=np.float32)
        self.d = np.zeros((capacity, 1), dtype=np.float32)
        self.size = 0
        self.ptr = 0
        self.capacity = capacity

    def push(self, s, a, r, s2, d):
        i = self.ptr % self.capacity
        self.s[i] = s
        self.a[i] = a
        self.r[i] = r
        self.s2[i] = s2
        self.d[i] = d
        self.ptr += 1
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            self.s[idx],
            self.a[idx],
            self.r[idx],
            self.s2[idx],
            self.d[idx],
        )
