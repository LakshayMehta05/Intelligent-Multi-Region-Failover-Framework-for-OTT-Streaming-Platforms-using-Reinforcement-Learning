import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.failover_env import StreamFailoverEnv

env = StreamFailoverEnv(n_regions=4, episode_length=100, seed=42)
env.load_codeco_traces()
print("CODECO traces loaded successfully. Nodes:", env.region_names_override)

obs, info = env.reset(seed=42)
total_reward = 0
for step in range(100):
    action = env.action_space.sample()
    obs, reward, term, trunc, info = env.step(action)
    total_reward += reward
    if step < 5:
        print(f"Step {step}: latency={env.latency.round(2)}, error={env.error_rate.round(2)}, degraded={info['degraded']}")
    if trunc or term:
        break

print(f"\nRan 100 steps on REAL CODECO data. Total reward: {total_reward:.2f}")
print("CODECO ENV CHECK PASSED")