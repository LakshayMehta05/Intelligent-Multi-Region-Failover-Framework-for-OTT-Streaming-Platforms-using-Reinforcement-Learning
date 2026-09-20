import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.failover_env import StreamFailoverEnv

trace_paths = [
    "data/5G_dataset/extracted/5G-production-dataset/Netflix/Static/Season3-StrangerThings/B_2019.11.26_13.50.48.csv",
    "data/5G_dataset/extracted/5G-production-dataset/Netflix/Static/Season3-StrangerThings/B_2019.12.03_08.02.05.csv",
    "data/5G_dataset/extracted/5G-production-dataset/Netflix/Static/animated-RickandMorty/B_2019.11.26_08.02.38.csv",
    "data/5G_dataset/extracted/5G-production-dataset/Netflix/Static/animated-RickandMorty/B_2019.11.28_08.02.19.csv",
]

env = StreamFailoverEnv(n_regions=4, episode_length=200, seed=42)
env.load_real_traces(trace_paths)
print("Real traces loaded successfully.")

obs, info = env.reset(seed=42)
total_reward = 0
for step in range(50):
    action = env.action_space.sample()
    obs, reward, term, trunc, info = env.step(action)
    total_reward += reward
    if step < 5:
        print(f"Step {step}: latency={env.latency.round(2)}, error={env.error_rate.round(2)}, degraded={info['degraded']}")
    if trunc or term:
        break

print(f"\nRan 50 steps on REAL trace data. Total reward: {total_reward:.2f}")
print("REAL DATA ENV CHECK PASSED")