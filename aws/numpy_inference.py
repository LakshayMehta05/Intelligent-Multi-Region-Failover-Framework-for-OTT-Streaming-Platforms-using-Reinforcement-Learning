"""
Pure NumPy forward pass through the trained DQN's Q-network.
Loads the exported real trained weights (no PyTorch/stable-baselines3
dependency needed) - suitable for AWS Lambda's size constraints.

Network architecture (matches Stable-Baselines3's default MlpPolicy):
    Linear(obs_dim, 64) -> ReLU -> Linear(64, 64) -> ReLU -> Linear(64, n_actions)
"""

import json
import numpy as np
import os

REGION_NAMES = ["us-east-1", "us-west-2", "eu-west-1", "ap-south-1"]

_weights = None


def _load_weights():
    global _weights
    if _weights is None:
        weights_path = os.path.join(os.path.dirname(__file__), "model_weights.json")
        with open(weights_path) as f:
            raw = json.load(f)
        _weights = {k: np.array(v) for k, v in raw.items()}
    return _weights


def relu(x):
    return np.maximum(0, x)


def forward(obs: np.ndarray) -> np.ndarray:
    """Run the real trained Q-network forward pass using exported weights."""
    w = _load_weights()

    x = obs
    x = x @ w["q_net.0.weight"].T + w["q_net.0.bias"]
    x = relu(x)
    x = x @ w["q_net.2.weight"].T + w["q_net.2.bias"]
    x = relu(x)
    x = x @ w["q_net.4.weight"].T + w["q_net.4.bias"]
    return x  # raw Q-values, one per action/region


def build_observation(regions: list, active_region_index: int, steps_since_switch: int, episode_length: int = 200) -> np.ndarray:
    n = len(regions)
    metrics = []
    for r in regions:
        metrics.extend([r["latency"], r["error_rate"], r["buffer_stall"], r["cpu_util"], r["network_load"]])

    active_onehot = [0.0] * n
    active_onehot[active_region_index] = 1.0

    time_feat = [min(steps_since_switch / episode_length, 1.0)]

    return np.array(metrics + active_onehot + time_feat, dtype=np.float32)


if __name__ == "__main__":
    test_regions = [
        {"latency": 0.6, "error_rate": 0.4, "buffer_stall": 0.5, "cpu_util": 0.5, "network_load": 0.5},
        {"latency": 0.2, "error_rate": 0.05, "buffer_stall": 0.1, "cpu_util": 0.3, "network_load": 0.3},
        {"latency": 0.3, "error_rate": 0.1, "buffer_stall": 0.2, "cpu_util": 0.4, "network_load": 0.4},
        {"latency": 0.25, "error_rate": 0.08, "buffer_stall": 0.15, "cpu_util": 0.35, "network_load": 0.35},
    ]
    obs = build_observation(test_regions, active_region_index=0, steps_since_switch=20)
    q_values = forward(obs)
    action = int(np.argmax(q_values))

    print("Q-values:", dict(zip(REGION_NAMES, q_values.round(2))))
    print(f"Chosen action: {REGION_NAMES[action]}")