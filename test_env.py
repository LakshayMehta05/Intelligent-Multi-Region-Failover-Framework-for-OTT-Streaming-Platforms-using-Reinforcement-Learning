import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.failover_env import StreamFailoverEnv
import numpy as np

env = StreamFailoverEnv(n_regions=4, episode_length=50, seed=42)
obs, info = env.reset()
print("Environment created successfully.")
print("Observation shape:", obs.shape)

total_reward = 0
for step in range(50):
    action = env.action_space.sample()
    obs, reward, term, trunc, info = env.step(action)
    total_reward += reward
    if trunc or term:
        break

print(f"Random policy ran 50 steps successfully. Total reward: {total_reward:.2f}")
print("ENV CHECK PASSED")