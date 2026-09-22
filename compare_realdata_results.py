import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from stable_baselines3 import DQN
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

if __name__ == "__main__":
    SEEDS = list(range(1000, 1010))
    model = DQN.load("results/dqn_realdata_model/best_model.zip")

    for name, fn in [
        ("Threshold Baseline (real data)", lambda s: run_episode_threshold(s)),
        ("DQN (real data)", lambda s: run_episode_dqn(s, model)),
    ]:
        rewards, switches, degraded = zip(*[fn(s) for s in SEEDS])
        print(f"\n=== {name} ===")
        print(f"Avg reward:    {np.mean(rewards):.2f}")
        print(f"Avg switches:  {np.mean(switches):.1f}")
        print(f"Avg degraded steps served: {np.mean(degraded):.1f} / 200")