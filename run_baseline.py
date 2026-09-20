import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.failover_env import StreamFailoverEnv
from baselines.threshold_policy import ThresholdFailoverPolicy
import numpy as np

def run_episode(seed):
    env = StreamFailoverEnv(n_regions=4, episode_length=200, seed=seed)
    policy = ThresholdFailoverPolicy(n_regions=4, degrade_threshold=0.5)
    obs, info = env.reset(seed=seed)
    total_reward, n_switches, n_degraded_steps = 0, 0, 0
    for step in range(200):
        action = policy.act(obs)
        obs, reward, term, trunc, info = env.step(action)
        total_reward += reward
        n_switches += info['switched']
        n_degraded_steps += info['degraded'][info['active_region']]
        if trunc or term:
            break
    return total_reward, n_switches, n_degraded_steps

results = [run_episode(s) for s in range(10)]
rewards, switches, degraded = zip(*results)

print("=== Rule-Based Threshold Baseline (represents 'existing methods') ===")
print(f"Average total reward over 10 episodes: {np.mean(rewards):.2f}")
print(f"Average number of region switches:     {np.mean(switches):.1f}")
print(f"Average steps served from degraded region: {np.mean(degraded):.1f} / 200")