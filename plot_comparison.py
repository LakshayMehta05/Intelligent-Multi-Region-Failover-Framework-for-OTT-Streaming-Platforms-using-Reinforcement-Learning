import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from env.failover_env import StreamFailoverEnv
from baselines.threshold_policy import ThresholdFailoverPolicy


def run_episode_threshold(seed):
    env = StreamFailoverEnv(n_regions=4, episode_length=200, seed=seed)
    policy = ThresholdFailoverPolicy(n_regions=4, degrade_threshold=0.5)
    obs, info = env.reset(seed=seed)
    total_reward, n_switches, n_degraded = 0, 0, 0
    for _ in range(200):
        action = policy.act(obs)
        obs, reward, term, trunc, info = env.step(action)
        total_reward += reward
        n_switches += info['switched']
        n_degraded += info['degraded'][info['active_region']]
        if trunc or term:
            break
    return total_reward, n_switches, n_degraded


def run_episode_dqn(seed, model):
    env = StreamFailoverEnv(n_regions=4, episode_length=200, seed=seed)
    obs, info = env.reset(seed=seed)
    total_reward, n_switches, n_degraded = 0, 0, 0
    for _ in range(200):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, info = env.step(int(action))
        total_reward += reward
        n_switches += info['switched']
        n_degraded += info['degraded'][info['active_region']]
        if trunc or term:
            break
    return total_reward, n_switches, n_degraded


if __name__ == "__main__":
    N_EVAL_EPISODES = 10
    SEEDS = list(range(1000, 1000 + N_EVAL_EPISODES))

    model = DQN.load("results/dqn_model/best_model.zip")

    results = {}
    for name, fn in [
        ("Threshold\nBaseline", lambda s: run_episode_threshold(s)),
        ("DQN\n(ours)", lambda s: run_episode_dqn(s, model)),
    ]:
        rewards, switches, degraded = zip(*[fn(s) for s in SEEDS])
        results[name] = {"reward": np.mean(rewards), "switches": np.mean(switches), "degraded": np.mean(degraded)}

    methods = list(results.keys())
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    axes[0].bar(methods, [results[m]["reward"] for m in methods], color=["orange", "steelblue"])
    axes[0].set_title("Avg Total Reward\n(higher = better)")

    axes[1].bar(methods, [results[m]["switches"] for m in methods], color=["orange", "steelblue"])
    axes[1].set_title("Avg Region Switches\n(lower = more stable)")

    axes[2].bar(methods, [results[m]["degraded"] for m in methods], color=["orange", "steelblue"])
    axes[2].set_title("Steps on Degraded Region\n(lower = better)")

    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/comparison_plot.png", dpi=150)
    print("Saved plot to results/comparison_plot.png")
    plt.show()