import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from env.multi_agent_env import MultiAgentCoordinator
from agents.train_multiagent import QNetwork


def load_agent(path, obs_dim, n_actions=2):
    net = QNetwork(obs_dim, n_actions)
    net.load_state_dict(torch.load(path))
    net.eval()
    return net


def act_deterministic(net, obs):
    with torch.no_grad():
        q = net(torch.as_tensor(obs).float().unsqueeze(0))
        return int(torch.argmax(q).item())


def run_episode(agent_a, agent_b, seed):
    coordinator = MultiAgentCoordinator(n_regions=4, episode_length=200, seed=seed)
    obs_a, obs_b = coordinator.reset(seed=seed)
    total_reward, n_switches, n_degraded = 0, 0, 0

    for step in range(200):
        action_a = act_deterministic(agent_a, obs_a)
        action_b = act_deterministic(agent_b, obs_b)
        obs_a, obs_b, reward, term, trunc, info = coordinator.step(action_a, action_b)
        total_reward += reward
        n_switches += info['switched']
        if info['degraded'][info['active_region']]:
            n_degraded += 1
        if term or trunc:
            break

    return total_reward, n_switches, n_degraded


if __name__ == "__main__":
    SEEDS = list(range(1000, 1010))

    coordinator = MultiAgentCoordinator(n_regions=4, episode_length=200, seed=0)
    obs_dim = coordinator.agent_a_view.observation_space.shape[0]

    agent_a = load_agent("results/multiagent_model/agent_a.pt", obs_dim)
    agent_b = load_agent("results/multiagent_model/agent_b.pt", obs_dim)

    results = [run_episode(agent_a, agent_b, s) for s in SEEDS]
    rewards, switches, degraded = zip(*results)

    print("=== Multi-Agent (2 coordinating DQN agents) ===")
    print(f"Avg reward:    {np.mean(rewards):.2f}")
    print(f"Avg switches:  {np.mean(switches):.1f}")
    print(f"Avg degraded steps served: {np.mean(degraded):.1f} / 200")