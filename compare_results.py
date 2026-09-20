import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
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


def run_episode_ppo(seed, model, vecnorm):
    # Use the RAW env for real reward tracking, but normalize obs before feeding to model
    env = StreamFailoverEnv(n_regions=4, episode_length=200, seed=seed)
    obs, info = env.reset(seed=seed)
    total_reward, n_switches, n_degraded = 0, 0, 0
    for _ in range(200):
        norm_obs = vecnorm.normalize_obs(obs)
        action, _ = model.predict(norm_obs, deterministic=True)
        obs, reward, term, trunc, info = env.step(int(action))
        total_reward += reward  # raw reward, for fair comparison
        n_switches += info['switched']
        n_degraded += info['degraded'][info['active_region']]
        if trunc or term:
            break
    return total_reward, n_switches, n_degraded


if __name__ == "__main__":
    N_EVAL_EPISODES = 10
    SEEDS = list(range(1000, 1000 + N_EVAL_EPISODES))

    dqn_model = DQN.load("results/dqn_model/best_model.zip")
    ppo_model = PPO.load("results/ppo_model/best_model.zip")

    # Load the normalization stats PPO trained with
    dummy_env = DummyVecEnv([lambda: StreamFailoverEnv(n_regions=4, episode_length=200, seed=0)])
    vecnorm = VecNormalize.load(f"results/ppo_model/vecnormalize.pkl", dummy_env)
    vecnorm.training = False

    for name, fn in [
        ("Threshold Baseline", lambda s: run_episode_threshold(s)),
        ("DQN", lambda s: run_episode_dqn(s, dqn_model)),
        ("PPO", lambda s: run_episode_ppo(s, ppo_model, vecnorm)),
    ]:
        rewards, switches, degraded = zip(*[fn(s) for s in SEEDS])
        print(f"\n=== {name} ===")
        print(f"Avg reward:    {np.mean(rewards):.2f}")
        print(f"Avg switches:  {np.mean(switches):.1f}")
        print(f"Avg degraded steps served: {np.mean(degraded):.1f} / 200")