"""
Lightweight multi-agent extension of StreamFailoverEnv.

Two independent agents each manage a pair of regions:
    Agent A: regions [0, 1]
    Agent B: regions [2, 3]

Each agent observes only its own pair's health metrics plus a small
"cross-agent" signal (the other pair's aggregate health), and chooses
which of its own 2 regions to route to. A simple coordinator then picks
the final serving region as whichever agent's chosen region has the
better health score overall - a lightweight decentralized-execution
approximation of full CTDE (centralized training, decentralized execution).

This is a simplified, defensible multi-agent extension - not a full CTDE
implementation, but a genuine two-agent decentralized decision system.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from env.failover_env import StreamFailoverEnv


class RegionPairAgentEnv(gym.Env):
    """Single-agent view over a pair of regions, with a cross-agent health signal."""

    def __init__(self, shared_state: StreamFailoverEnv, region_pair: list, other_pair: list):
        super().__init__()
        self.shared_state = shared_state
        self.region_pair = region_pair
        self.other_pair = other_pair

        obs_dim = 5 * 2 + 2 + 1 + 1
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)

    def _get_obs(self):
        env = self.shared_state
        metrics = []
        for r in self.region_pair:
            metrics.extend([env.latency[r], env.error_rate[r], env.buffer_stall[r], env.cpu_util[r], env.network_load[r]])

        active_onehot = [0.0, 0.0]
        if env.active_region in self.region_pair:
            active_onehot[self.region_pair.index(env.active_region)] = 1.0

        time_feat = min(env.steps_since_switch / env.episode_length, 1.0)

        other_health = np.mean([
            env.error_rate[r] + env.buffer_stall[r] + 0.3 * env.latency[r] for r in self.other_pair
        ])

        return np.array(metrics + active_onehot + [time_feat, other_health], dtype=np.float32)

    def local_action_to_global(self, local_action: int) -> int:
        return self.region_pair[local_action]

    def local_health_score(self, local_action: int) -> float:
        env = self.shared_state
        r = self.region_pair[local_action]
        return env.error_rate[r] + env.buffer_stall[r] + 0.3 * env.latency[r]


class MultiAgentCoordinator:
    """Runs the shared environment with 2 independent agents, coordinating their proposals."""

    def __init__(self, n_regions=4, episode_length=200, seed=None):
        self.shared_env = StreamFailoverEnv(n_regions=n_regions, episode_length=episode_length, seed=seed)
        self.agent_a_view = RegionPairAgentEnv(self.shared_env, region_pair=[0, 1], other_pair=[2, 3])
        self.agent_b_view = RegionPairAgentEnv(self.shared_env, region_pair=[2, 3], other_pair=[0, 1])

    def reset(self, seed=None):
        self.shared_env.reset(seed=seed)
        return self.agent_a_view._get_obs(), self.agent_b_view._get_obs()

    def step(self, action_a: int, action_b: int):
        """Each agent proposes a region from its own pair; coordinator picks the healthier proposal."""
        global_a = self.agent_a_view.local_action_to_global(action_a)
        global_b = self.agent_b_view.local_action_to_global(action_b)

        score_a = self.agent_a_view.local_health_score(action_a)
        score_b = self.agent_b_view.local_health_score(action_b)

        final_action = global_a if score_a <= score_b else global_b

        obs, reward, term, trunc, info = self.shared_env.step(final_action)

        obs_a = self.agent_a_view._get_obs()
        obs_b = self.agent_b_view._get_obs()

        info["final_region"] = final_action
        info["agent_a_proposed"] = global_a
        info["agent_b_proposed"] = global_b

        return obs_a, obs_b, reward, term, trunc, info