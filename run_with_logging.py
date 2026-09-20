"""
Run the trained DQN agent and log every decision with full explainability
(Q-values + reasoning) to results/decision_log.jsonl
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from stable_baselines3 import DQN
from env.failover_env import StreamFailoverEnv
from explainability import DecisionLogger, REGION_NAMES

def get_q_values(model, obs):
    obs_tensor = torch.as_tensor(obs).float().unsqueeze(0)
    with torch.no_grad():
        q_values = model.q_net(obs_tensor).squeeze(0).numpy()
    return q_values

def run(seed=7, n_steps=50):
    env = StreamFailoverEnv(n_regions=4, episode_length=n_steps, seed=seed)
    model = DQN.load("results/dqn_model/best_model.zip")
    logger = DecisionLogger("results/decision_log.jsonl")

    obs, info = env.reset(seed=seed)
    total_reward = 0

    for step in range(n_steps):
        active_before = env.active_region
        degraded_before = np.where(env.degraded)[0].tolist()

        q_values = get_q_values(model, obs)
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)

        obs, reward, term, trunc, info = env.step(action)
        total_reward += reward

        reason = logger.log_decision(
            step=step, obs=obs, action=action, q_values=q_values,
            active_region_before=active_before, degraded_regions=degraded_before,
            reward=reward,
        )

        if step < 5 or action != active_before:
            print(f"Step {step}: {reason}")

        if trunc or term:
            break

    logger.close()
    print(f"\nTotal reward: {total_reward:.2f}")
    print("Full decision log saved to results/decision_log.jsonl")

if __name__ == "__main__":
    run(seed=7, n_steps=50)