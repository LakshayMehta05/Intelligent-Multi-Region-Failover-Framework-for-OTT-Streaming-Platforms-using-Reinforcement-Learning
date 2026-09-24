"""
Train two independent DQN agents in the lightweight multi-agent
failover setup (each managing a pair of regions, coordinated by
picking whichever agent's proposal is healthier).
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

from env.multi_agent_env import MultiAgentCoordinator

MODEL_DIR = "results/multiagent_model"
os.makedirs(MODEL_DIR, exist_ok=True)


class QNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class SimpleDQNAgent:
    def __init__(self, obs_dim, n_actions, lr=1e-3, gamma=0.98, buffer_size=20000):
        self.q_net = QNetwork(obs_dim, n_actions)
        self.target_net = QNetwork(obs_dim, n_actions)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.gamma = gamma
        self.buffer = deque(maxlen=buffer_size)
        self.n_actions = n_actions
        self.epsilon = 1.0

    def act(self, obs):
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            q = self.q_net(torch.as_tensor(obs).float().unsqueeze(0))
            return int(torch.argmax(q).item())

    def store(self, obs, action, reward, next_obs, done):
        self.buffer.append((obs, action, reward, next_obs, done))

    def train_step(self, batch_size=64):
        if len(self.buffer) < batch_size:
            return
        batch = random.sample(self.buffer, batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)

        obs = torch.as_tensor(np.array(obs)).float()
        actions = torch.as_tensor(actions).long().unsqueeze(1)
        rewards = torch.as_tensor(rewards).float()
        next_obs = torch.as_tensor(np.array(next_obs)).float()
        dones = torch.as_tensor(dones).float()

        q_values = self.q_net(obs).gather(1, actions).squeeze(1)
        with torch.no_grad():
            next_q = self.target_net(next_obs).max(1)[0]
            target = rewards + self.gamma * next_q * (1 - dones)

        loss = nn.functional.mse_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def decay_epsilon(self, min_eps=0.05, decay=0.98):
        self.epsilon = max(min_eps, self.epsilon * decay)


if __name__ == "__main__":
    coordinator = MultiAgentCoordinator(n_regions=4, episode_length=200, seed=0)
    obs_dim = coordinator.agent_a_view.observation_space.shape[0]

    agent_a = SimpleDQNAgent(obs_dim, n_actions=2)
    agent_b = SimpleDQNAgent(obs_dim, n_actions=2)

    N_EPISODES = 800
    UPDATE_TARGET_EVERY = 20

    for ep in range(N_EPISODES):
        obs_a, obs_b = coordinator.reset(seed=ep)
        ep_reward = 0

        for step in range(200):
            action_a = agent_a.act(obs_a)
            action_b = agent_b.act(obs_b)

            next_obs_a, next_obs_b, reward, term, trunc, info = coordinator.step(action_a, action_b)
            done = term or trunc

            agent_a.store(obs_a, action_a, reward, next_obs_a, done)
            agent_b.store(obs_b, action_b, reward, next_obs_b, done)

            agent_a.train_step()
            agent_b.train_step()

            obs_a, obs_b = next_obs_a, next_obs_b
            ep_reward += reward

            if done:
                break

        agent_a.decay_epsilon()
        agent_b.decay_epsilon()

        if ep % UPDATE_TARGET_EVERY == 0:
            agent_a.update_target()
            agent_b.update_target()

        if ep % 25 == 0:
            print(f"Episode {ep}: reward={ep_reward:.2f}, epsilon={agent_a.epsilon:.3f}")

    torch.save(agent_a.q_net.state_dict(), f"{MODEL_DIR}/agent_a.pt")
    torch.save(agent_b.q_net.state_dict(), f"{MODEL_DIR}/agent_b.pt")
    print("Training complete. Models saved to", MODEL_DIR)