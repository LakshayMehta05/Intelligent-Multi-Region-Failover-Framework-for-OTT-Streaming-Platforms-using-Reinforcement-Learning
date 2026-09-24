"""
Resilience test: force a coordinated multi-region outage (2-3 regions
down simultaneously) and measure how DQN vs the threshold baseline
handle it, compared to normal single-region-failure conditions.

This directly supports the "Better Security & Resilience" objective.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from stable_baselines3 import DQN
from env.failover_env import StreamFailoverEnv
from baselines.threshold_policy import ThresholdFailoverPolicy


class ForcedOutageEnv(StreamFailoverEnv):
    """Wraps StreamFailoverEnv to force specific regions degraded during a window."""

    def __init__(self, outage_regions, outage_start=50, outage_end=120, **kwargs):
        self.outage_regions = outage_regions
        self.outage_start = outage_start
        self.outage_end = outage_end
        super().__init__(**kwargs)

    def step(self, action):
        obs, reward, term, trunc, info = super().step(action)
        if self.outage_start <= self.t <= self.outage_end:
            for r in self.outage_regions:
                self.degraded[r] = True
                self.error_rate[r] = max(self.error_rate[r], 0.6)
                self.buffer_stall[r] = max(self.buffer_stall[r], 0.8)
            info["degraded"] = self.degraded.copy()
        return self._get_obs(), reward, term, trunc, info


def run_episode(env_factory, policy_fn, seed):
    env = env_factory(seed)
    obs, info = env.reset(seed=seed)
    total_reward, n_switches, n_degraded, n_outage_degraded = 0, 0, 0, 0
    for step in range(200):
        action = policy_fn(obs)
        obs, reward, term, trunc, info = env.step(action)
        total_reward += reward
        n_switches += info['switched']
        if info['degraded'][info['active_region']]:
            n_degraded += 1
            if hasattr(env, 'outage_start') and env.outage_start <= step <= env.outage_end:
                n_outage_degraded += 1
        if trunc or term:
            break
    return total_reward, n_switches, n_degraded, n_outage_degraded


if __name__ == "__main__":
    SEEDS = list(range(2000, 2010))
    OUTAGE_REGIONS = [0, 1]  # 2 of 4 regions go down simultaneously

    def make_env(seed):
        return ForcedOutageEnv(outage_regions=OUTAGE_REGIONS, outage_start=50, outage_end=120,
                                n_regions=4, episode_length=200, seed=seed)

    dqn_model = DQN.load("results/dqn_resilient_model/best_model.zip")
    threshold_policy = ThresholdFailoverPolicy(n_regions=4, degrade_threshold=0.5)

    print("=" * 70)
    print(f"RESILIENCE TEST: regions {OUTAGE_REGIONS} forced degraded during steps 50-120")
    print("=" * 70)

    for name, policy_fn in [
        ("Threshold Baseline", lambda obs: threshold_policy.act(obs)),
        ("DQN", lambda obs: int(dqn_model.predict(obs, deterministic=True)[0])),
    ]:
        results = [run_episode(make_env, policy_fn, s) for s in SEEDS]
        rewards, switches, degraded, outage_degraded = zip(*results)
        print(f"\n=== {name} ===")
        print(f"Avg total reward:              {np.mean(rewards):.2f}")
        print(f"Avg switches:                  {np.mean(switches):.1f}")
        print(f"Avg total degraded steps:      {np.mean(degraded):.1f} / 200")
        print(f"Avg degraded steps DURING outage window (of 70 possible): {np.mean(outage_degraded):.1f} / 70")