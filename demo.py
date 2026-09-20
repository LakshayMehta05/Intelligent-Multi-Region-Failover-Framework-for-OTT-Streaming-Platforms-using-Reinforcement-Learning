
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from stable_baselines3 import DQN
from env.failover_env import StreamFailoverEnv

REGION_NAMES = ["us-east-1", "us-west-2", "eu-west-1", "ap-south-1"]

def explain_decision(obs, action, active_region_before, n_regions=4):
    metrics = obs[: n_regions * 5].reshape(n_regions, 5)
    latency, error, stall, cpu, net = metrics[action]

    if action == active_region_before:
                return f"Staying on {REGION_NAMES[action]} - current region is healthy enough (latency={latency:.2f}, error={error:.2f})."
    else:
                return (f"Switching from {REGION_NAMES[active_region_before]} to {REGION_NAMES[action]} - "
                f"target region has latency={latency:.2f}, error_rate={error:.2f}, stall_rate={stall:.2f}.")


def run_demo(seed=7, n_steps=40, delay=False):
    env = StreamFailoverEnv(n_regions=4, episode_length=n_steps, seed=seed)
    model = DQN.load("results/dqn_model/best_model.zip")

    obs, info = env.reset(seed=seed)
    print("=" * 70)
    print("LIVE DEMO: DQN Multi-Region Failover Agent")
    print("=" * 70)

    total_reward = 0
    for step in range(n_steps):
        active_before = env.active_region
        degraded_regions = [REGION_NAMES[i] for i in np.where(env.degraded)[0]]

        action, _ = model.predict(obs, deterministic=True)
        action = int(action)

        explanation = explain_decision(obs, action, active_before)

        obs, reward, term, trunc, info = env.step(action)
        total_reward += reward

        status = "OUTAGE SIMULATED: " + ", ".join(degraded_regions) if degraded_regions else "all regions healthy"
        marker = " >>> SWITCH <<<" if action != active_before else ""
        print(f"\nStep {step+1:02d} | {status}{marker}")
        print(f"  Decision: {explanation}")
        print(f"  Reward this step: {reward:.2f}")

        if trunc or term:
            break

    print("\n" + "=" * 70)
    print(f"Demo complete. Total reward over {n_steps} steps: {total_reward:.2f}")
    print("=" * 70)


if __name__ == "__main__":
    run_demo(seed=7, n_steps=40)