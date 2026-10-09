"""
QMIX-based multi-agent training for StreamFailOverAI.

True CTDE (Centralized Training, Decentralized Execution):
- Each agent picks its own action from its own LOCAL observation only
  (this is what runs at inference time / decentralized execution).
- A centralized Mixing Network combines both agents' Q-values using
  GLOBAL state (all 4 regions' full metrics) to produce a joint Q-total,
  which is trained against a joint TD target. This lets agents learn to
  cooperate using information that is only available during training.

This replaces the earlier "independent DQN + hand-coded arbitration"
multi-agent setup with a real literature-standard CTDE algorithm.
"""
import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.multi_agent_env import MultiAgentCoordinator

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOCAL_OBS_DIM = 14      # per-agent local observation (as defined in multi_agent_env.py)
GLOBAL_STATE_DIM = 20   # 4 regions x 5 metrics = full joint state
N_ACTIONS = 2
N_EPISODES = 800
BATCH_SIZE = 64
BUFFER_CAPACITY = 20000
GAMMA = 0.99
LR = 1e-3
UPDATE_TARGET_EVERY = 20
MIN_EPS = 0.05
EPS_DECAY = 0.98


class AgentQNetwork(nn.Module):
    """Local per-agent Q-network - used for decentralized execution."""
    def __init__(self, obs_dim=LOCAL_OBS_DIM, n_actions=N_ACTIONS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class MixingNetwork(nn.Module):
    """
    Centralized mixing network (simplified QMIX).
    Combines each agent's chosen-action Q-value into Q_total, conditioned
    on the GLOBAL state. Only used during training - never at execution time.
    Weights are forced non-negative (via abs) to preserve monotonicity:
    improving any individual agent's Q can only improve (or not hurt) Q_total.
    """
    def __init__(self, n_agents=2, state_dim=GLOBAL_STATE_DIM, mixing_hidden=32):
        super().__init__()
        self.n_agents = n_agents
        self.hyper_w1 = nn.Linear(state_dim, n_agents * mixing_hidden)
        self.hyper_b1 = nn.Linear(state_dim, mixing_hidden)
        self.hyper_w2 = nn.Linear(state_dim, mixing_hidden)
        self.hyper_b2 = nn.Sequential(nn.Linear(state_dim, mixing_hidden), nn.ReLU(), nn.Linear(mixing_hidden, 1))
        self.mixing_hidden = mixing_hidden

    def forward(self, agent_qs, global_state):
        bs = agent_qs.shape[0]
        w1 = torch.abs(self.hyper_w1(global_state)).view(bs, self.n_agents, self.mixing_hidden)
        b1 = self.hyper_b1(global_state).view(bs, 1, self.mixing_hidden)
        hidden = torch.relu(torch.bmm(agent_qs.view(bs, 1, self.n_agents), w1) + b1)

        w2 = torch.abs(self.hyper_w2(global_state)).view(bs, self.mixing_hidden, 1)
        b2 = self.hyper_b2(global_state).view(bs, 1, 1)
        q_total = torch.bmm(hidden, w2) + b2
        return q_total.view(bs, 1)


class ReplayBuffer:
    def __init__(self, capacity=BUFFER_CAPACITY):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(args)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


def get_global_state(coordinator):
    """Flatten all 4 regions' raw metrics into one global state vector."""
    env = coordinator.shared_env
    state = []
    for r in range(env.n_regions):
        state.extend([env.latency[r], env.error_rate[r], env.buffer_stall[r], env.cpu_util[r], env.network_load[r]])
    return np.array(state, dtype=np.float32)


def select_action(q_net, local_obs, epsilon):
    if random.random() < epsilon:
        return random.randint(0, N_ACTIONS - 1)
    with torch.no_grad():
        q_vals = q_net(torch.FloatTensor(local_obs).unsqueeze(0).to(DEVICE))
        return int(torch.argmax(q_vals, dim=1).item())


def train():
    coordinator = MultiAgentCoordinator()

    agent_a_net = AgentQNetwork().to(DEVICE)
    agent_b_net = AgentQNetwork().to(DEVICE)
    agent_a_target = AgentQNetwork().to(DEVICE)
    agent_b_target = AgentQNetwork().to(DEVICE)
    agent_a_target.load_state_dict(agent_a_net.state_dict())
    agent_b_target.load_state_dict(agent_b_net.state_dict())

    mixer = MixingNetwork().to(DEVICE)
    mixer_target = MixingNetwork().to(DEVICE)
    mixer_target.load_state_dict(mixer.state_dict())

    params = list(agent_a_net.parameters()) + list(agent_b_net.parameters()) + list(mixer.parameters())
    optimizer = optim.Adam(params, lr=LR)
    buffer = ReplayBuffer()

    epsilon = 1.0
    episode_rewards = []

    for episode in range(N_EPISODES):
        obs_a, obs_b = coordinator.reset()
        global_state = get_global_state(coordinator)
        done = False
        ep_reward = 0.0

        while not done:
            action_a = select_action(agent_a_net, obs_a, epsilon)
            action_b = select_action(agent_b_net, obs_b, epsilon)

            next_obs_a, next_obs_b, team_reward, term, trunc, info = coordinator.step(action_a, action_b)
            done = term or trunc
            next_global_state = get_global_state(coordinator)

            buffer.push(obs_a, obs_b, action_a, action_b, global_state,
                        team_reward, next_obs_a, next_obs_b, next_global_state, done)

            obs_a, obs_b, global_state = next_obs_a, next_obs_b, next_global_state
            ep_reward += team_reward

            if len(buffer) >= BATCH_SIZE:
                batch = buffer.sample(BATCH_SIZE)
                (b_oa, b_ob, b_aa, b_ab, b_gs, b_r, b_noa, b_nob, b_ngs, b_done) = zip(*batch)

                b_oa = torch.FloatTensor(np.array(b_oa)).to(DEVICE)
                b_ob = torch.FloatTensor(np.array(b_ob)).to(DEVICE)
                b_aa = torch.LongTensor(b_aa).to(DEVICE)
                b_ab = torch.LongTensor(b_ab).to(DEVICE)
                b_gs = torch.FloatTensor(np.array(b_gs)).to(DEVICE)
                b_r = torch.FloatTensor(b_r).to(DEVICE)
                b_noa = torch.FloatTensor(np.array(b_noa)).to(DEVICE)
                b_nob = torch.FloatTensor(np.array(b_nob)).to(DEVICE)
                b_ngs = torch.FloatTensor(np.array(b_ngs)).to(DEVICE)
                b_done = torch.FloatTensor(b_done).to(DEVICE)

                q_a = agent_a_net(b_oa).gather(1, b_aa.unsqueeze(1)).squeeze(1)
                q_b = agent_b_net(b_ob).gather(1, b_ab.unsqueeze(1)).squeeze(1)
                agent_qs = torch.stack([q_a, q_b], dim=1)
                q_total = mixer(agent_qs, b_gs).squeeze(1)

                with torch.no_grad():
                    next_q_a = agent_a_target(b_noa).max(dim=1)[0]
                    next_q_b = agent_b_target(b_nob).max(dim=1)[0]
                    next_agent_qs = torch.stack([next_q_a, next_q_b], dim=1)
                    next_q_total = mixer_target(next_agent_qs, b_ngs).squeeze(1)
                    target = b_r + GAMMA * next_q_total * (1 - b_done)

                loss = nn.functional.mse_loss(q_total, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        epsilon = max(MIN_EPS, epsilon * EPS_DECAY)
        episode_rewards.append(ep_reward)

        if episode % UPDATE_TARGET_EVERY == 0:
            agent_a_target.load_state_dict(agent_a_net.state_dict())
            agent_b_target.load_state_dict(agent_b_net.state_dict())
            mixer_target.load_state_dict(mixer.state_dict())

        if episode % 25 == 0:
            avg_r = np.mean(episode_rewards[-25:])
            print(f"Episode {episode}/{N_EPISODES} | epsilon={epsilon:.3f} | avg_reward(last25)={avg_r:.2f}")

    os.makedirs("results/qmix_model", exist_ok=True)
    torch.save(agent_a_net.state_dict(), "results/qmix_model/agent_a.pt")
    torch.save(agent_b_net.state_dict(), "results/qmix_model/agent_b.pt")
    torch.save(mixer.state_dict(), "results/qmix_model/mixer.pt")
    print("Saved QMIX models to results/qmix_model/")
    print("NOTE: at execution/inference time, only agent_a.pt and agent_b.pt are needed -")
    print("mixer.pt is training-only, confirming true decentralized execution.")


if __name__ == "__main__":
    train()