"""
Pure Python (no NumPy) forward pass through the trained DQN's Q-network.
Avoids all Lambda packaging issues - stdlib only.
"""

import json
import os

REGION_NAMES = ["us-east-1", "us-west-2", "eu-west-1", "ap-south-1"]

_weights = None


def _load_weights():
    global _weights
    if _weights is None:
        weights_path = os.path.join(os.path.dirname(__file__), "model_weights.json")
        with open(weights_path) as f:
            _weights = json.load(f)
    return _weights


def _matvec(matrix, vector, bias):
    """y = matrix @ vector + bias, where matrix is list-of-lists (out_dim x in_dim)."""
    out = []
    for row, b in zip(matrix, bias):
        s = sum(row[i] * vector[i] for i in range(len(vector))) + b
        out.append(s)
    return out


def _relu(vec):
    return [max(0.0, v) for v in vec]


def forward(obs: list) -> list:
    w = _load_weights()

    x = _matvec(w["q_net.0.weight"], obs, w["q_net.0.bias"])
    x = _relu(x)
    x = _matvec(w["q_net.2.weight"], x, w["q_net.2.bias"])
    x = _relu(x)
    x = _matvec(w["q_net.4.weight"], x, w["q_net.4.bias"])
    return x


def build_observation(regions: list, active_region_index: int, steps_since_switch: int, episode_length: int = 200) -> list:
    n = len(regions)
    metrics = []
    for r in regions:
        metrics.extend([r["latency"], r["error_rate"], r["buffer_stall"], r["cpu_util"], r["network_load"]])

    active_onehot = [0.0] * n
    active_onehot[active_region_index] = 1.0

    time_feat = [min(steps_since_switch / episode_length, 1.0)]

    return metrics + active_onehot + time_feat


if __name__ == "__main__":
    test_regions = [
        {"latency": 0.6, "error_rate": 0.4, "buffer_stall": 0.5, "cpu_util": 0.5, "network_load": 0.5},
        {"latency": 0.2, "error_rate": 0.05, "buffer_stall": 0.1, "cpu_util": 0.3, "network_load": 0.3},
        {"latency": 0.3, "error_rate": 0.1, "buffer_stall": 0.2, "cpu_util": 0.4, "network_load": 0.4},
        {"latency": 0.25, "error_rate": 0.08, "buffer_stall": 0.15, "cpu_util": 0.35, "network_load": 0.35},
    ]
    obs = build_observation(test_regions, active_region_index=0, steps_since_switch=20)
    q_values = forward(obs)
    action = q_values.index(max(q_values))

    print("Q-values:", dict(zip(REGION_NAMES, [round(q, 2) for q in q_values])))
    print(f"Chosen action: {REGION_NAMES[action]}")