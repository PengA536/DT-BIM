import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .replay_buffer import ReplayBuffer

def mlp(in_dim, hidden, out_dim, out_act=nn.Tanh):
    layers = []
    last = in_dim
    for h in hidden:
        layers += [nn.Linear(last, h), nn.ReLU()]
        last = h
    layers += [nn.Linear(last, out_dim), out_act()]
    return nn.Sequential(*layers)

class TD3Agent:
    def __init__(self, state_dim, action_dim, cfg):
        h = cfg["actor_hidden"]
        self.actor = mlp(state_dim, h, action_dim, out_act=nn.Tanh)
        self.actor_target = mlp(state_dim, h, action_dim, out_act=nn.Tanh)
        self.actor_target.load_state_dict(self.actor.state_dict())

        ch = cfg["critic_hidden"]
        self.q1 = mlp(state_dim + action_dim, ch, 1, out_act=nn.Identity)
        self.q2 = mlp(state_dim + action_dim, ch, 1, out_act=nn.Identity)
        self.q1_target = mlp(state_dim + action_dim, ch, 1, out_act=nn.Identity)
        self.q2_target = mlp(state_dim + action_dim, ch, 1, out_act=nn.Identity)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=cfg["actor_lr"])
        self.q1_opt = optim.Adam(self.q1.parameters(), lr=cfg["critic_lr"])
        self.q2_opt = optim.Adam(self.q2.parameters(), lr=cfg["critic_lr"])

        self.tau = cfg["tau"]
        self.gamma = cfg["gamma"]
        self.policy_delay = cfg["policy_delay"]
        self.noise_std = cfg["noise_std"]
        self.noise_clip = cfg["noise_clip"]
        self.batch_size = cfg["batch_size"]

        self.buffer = ReplayBuffer(state_dim, action_dim, capacity=cfg["buffer_size"])
        self.total_steps = 0

    @torch.no_grad()
    def act(self, s, noise=True):
        s = torch.as_tensor(s, dtype=torch.float32)
        a = self.actor(s).numpy()
        if noise:
            a += np.clip(np.random.normal(0, self.noise_std, size=a.shape), -self.noise_clip, self.noise_clip)
        return np.clip(a, -1.0, 1.0)

    def update(self):
        if self.buffer.size < self.batch_size:
            return None
        s, a, r, s2, d = self.buffer.sample(self.batch_size)
        s = torch.as_tensor(s, dtype=torch.float32)
        a = torch.as_tensor(a, dtype=torch.float32)
        r = torch.as_tensor(r, dtype=torch.float32)
        s2 = torch.as_tensor(s2, dtype=torch.float32)
        d = torch.as_tensor(d, dtype=torch.float32)

        with torch.no_grad():
            noise = torch.clamp(
                torch.randn_like(a) * self.noise_std,
                -self.noise_clip,
                self.noise_clip,
            )
            a2 = torch.clamp(self.actor_target(s2) + noise, -1.0, 1.0)
            q1t = self.q1_target(torch.cat([s2, a2], dim=-1))
            q2t = self.q2_target(torch.cat([s2, a2], dim=-1))
            qt = torch.min(q1t, q2t)
            y = r + self.gamma * (1.0 - d) * qt

        # Critic update
        q1 = self.q1(torch.cat([s, a], dim=-1))
        q2 = self.q2(torch.cat([s, a], dim=-1))
        q1_loss = ((q1 - y)**2).mean()
        q2_loss = ((q2 - y)**2).mean()
        self.q1_opt.zero_grad(); q1_loss.backward(); self.q1_opt.step()
        self.q2_opt.zero_grad(); q2_loss.backward(); self.q2_opt.step()

        info = {"q1_loss": q1_loss.item(), "q2_loss": q2_loss.item()}

        # Delayed policy updates
        if self.total_steps % self.policy_delay == 0:
            # Actor update
            a_pi = self.actor(s)
            q1_pi = self.q1(torch.cat([s, a_pi], dim=-1))
            actor_loss = -q1_pi.mean()
            self.actor_opt.zero_grad(); actor_loss.backward(); self.actor_opt.step()
            info["actor_loss"] = actor_loss.item()

            # Polyak averaging
            with torch.no_grad():
                for p, pt in zip(self.q1.parameters(), self.q1_target.parameters()):
                    pt.data.mul_(1 - self.tau); pt.data.add_(self.tau * p.data)
                for p, pt in zip(self.q2.parameters(), self.q2_target.parameters()):
                    pt.data.mul_(1 - self.tau); pt.data.add_(self.tau * p.data)
                for p, pt in zip(self.actor.parameters(), self.actor_target.parameters()):
                    pt.data.mul_(1 - self.tau); pt.data.add_(self.tau * p.data)

        self.total_steps += 1
        return info
