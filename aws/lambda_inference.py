"""
AWS Lambda handler for real-time failover routing decisions.

This function is designed to be deployed as an AWS Lambda function.
It loads the trained DQN model and, given current region health metrics
(as would be pushed by CloudWatch/Route53 ARC health checks), returns
which region should serve traffic.

Can be tested locally (see __main__ block) before actual AWS deployment.
"""

import json
import numpy as np

REGION_NAMES = ["us-east-1", "us-west-2", "eu-west-1", "ap-south-1"]

_model = None


def _get_model():
    global _model
    if _model is None:
        from stable_baselines3 import DQN
        _model = DQN.load("results/dqn_realdata_model/best_model.zip")
    return _model


def _build_observation(regions: list, active_region_index: int, steps_since_switch: int, episode_length: int = 200):
    n = len(regions)
    metrics = []
    for r in regions:
        metrics.extend([r["latency"], r["error_rate"], r["buffer_stall"], r["cpu_util"], r["network_load"]])

    active_onehot = [0.0] * n
    active_onehot[active_region_index] = 1.0

    time_feat = [min(steps_since_switch / episode_length, 1.0)]

    obs = np.array(metrics + active_onehot + time_feat, dtype=np.float32)
    return obs


def lambda_handler(event, context=None):
    """AWS Lambda entry point."""
    import torch

    regions = event["regions"]
    active_region_index = event.get("active_region_index", 0)
    steps_since_switch = event.get("steps_since_switch", 0)

    model = _get_model()
    obs = _build_observation(regions, active_region_index, steps_since_switch)

    obs_tensor = torch.as_tensor(obs).float().unsqueeze(0)
    with torch.no_grad():
        q_values = model.q_net(obs_tensor).squeeze(0).numpy()

    action = int(np.argmax(q_values))
    switched = action != active_region_index

    n = len(regions)
    region_names = REGION_NAMES[:n]

    response = {
        "action_region_index": action,
        "action_region_name": region_names[action],
        "switch_recommended": switched,
        "q_values": {region_names[i]: float(q_values[i]) for i in range(n)},
        "explanation": (
            f"{'Switch to' if switched else 'Stay on'} {region_names[action]} "
            f"(Q-value={q_values[action]:.2f}, the highest among all {n} regions)"
        ),
    }
    return {
        "statusCode": 200,
        "body": json.dumps(response),
    }


if __name__ == "__main__":
    test_event = {
        "regions": [
            {"latency": 0.6, "error_rate": 0.4, "buffer_stall": 0.5, "cpu_util": 0.5, "network_load": 0.5},
            {"latency": 0.2, "error_rate": 0.05, "buffer_stall": 0.1, "cpu_util": 0.3, "network_load": 0.3},
            {"latency": 0.3, "error_rate": 0.1, "buffer_stall": 0.2, "cpu_util": 0.4, "network_load": 0.4},
            {"latency": 0.25, "error_rate": 0.08, "buffer_stall": 0.15, "cpu_util": 0.35, "network_load": 0.35},
        ],
        "active_region_index": 0,
        "steps_since_switch": 20,
    }

    result = lambda_handler(test_event)
    print("Lambda response:")
    print(json.dumps(json.loads(result["body"]), indent=2))