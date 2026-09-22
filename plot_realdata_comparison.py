import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env.failover_env import StreamFailoverEnv
from baselines.threshold_policy import ThresholdFailoverPolicy

TRACE_PATHS = [
    "data/5G_dataset/extracted/5G-production-dataset/Netflix/Static/Season3-StrangerThings/B_2019.11.26_13.50.48.csv",
    "data/5G_dataset/extracted/5G-production-dataset/Netflix/Static/Season3-StrangerThings/B_2019.12.03_08.02.05.csv",
    "data/5G_dataset/extracted/5G-production-dataset/Netflix/Static/animated-RickandMorty/B_2019.11.26_08.02.38.csv",
    "data/5G_dataset/extracted/5G-production-dataset/Netflix/Static/animated-RickandMorty/B_2019.11.28_08.02.19.csv",
]

def make_env(seed):
    env = StreamFailoverEnv(n_regions=4, episode_length=200, seed=seed)
    env.load_real_traces(TRACE_PATHS)
    return env

def run_episode_threshold(seed):
    env = make_env(seed)
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
    env = make_env(seed)
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

def run_episode_ppo(seed, model, vecnorm):
    env = make_env(seed)
    obs, info = env.reset(seed=seed)
    total_reward, n_switches, n_degraded = 0, 0, 0
    for _ in range(200):
        norm_obs = vecnorm.normalize_obs(obs)
        action, _ = model.predict(norm_obs, deterministic=True)
        obs, reward, term, trunc, info = env.step(int(action))
        total_reward += reward
        n_switches += info['switched']
        n_degraded += info['degraded'][info['active_region']]
        if trunc or term:
            break
    return total_reward, n_switches, n_degraded

if __name__ == "__main__":
    SEEDS = list(range(1000, 1010))

    dqn_model = DQN.load("results/dqn_realdata_model/best_model.zip")
    ppo_model = PPO.load("results/ppo_realdata_model/best_model.zip")
    dummy_env = DummyVecEnv([lambda: StreamFailoverEnv(n_regions=4, episode_length=200, seed=0)])
    vecnorm = VecNormalize.load("results/ppo_realdata_model/vecnormalize.pkl", dummy_env)
    vecnorm.training = False

    results = {}
    for name, fn in [
        ("Threshold\nBaseline", lambda s: run_episode_threshold(s)),
        ("DQN", lambda s: run_episode_dqn(s, dqn_model)),
        ("PPO", lambda s: run_episode_ppo(s, ppo_model, vecnorm)),
    ]:
        rewards, switches, degraded = zip(*[fn(s) for s in SEEDS])
        results[name] = {"reward": np.mean(rewards), "switches": np.mean(switches), "degraded": np.mean(degraded)}

    methods = list(results.keys())
    colors = ["orange", "steelblue", "seagreen"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle("Performance on REAL 5G Trace Data (UCC Dataset, Real Netflix Streaming Sessions)", fontsize=12)

    axes[0].bar(methods, [results[m]["reward"] for m in methods], color=colors)
    axes[0].set_title("Avg Total Reward\n(higher = better)")

    axes[1].bar(methods, [results[m]["switches"] for m in methods], color=colors)
    axes[1].set_title("Avg Region Switches")

    axes[2].bar(methods, [results[m]["degraded"] for m in methods], color=colors)
    axes[2].set_title("Steps on Degraded Region\n(lower = better)")

    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/realdata_comparison_plot.png", dpi=150)
    print("Saved plot to results/realdata_comparison_plot.png")
    plt.show()