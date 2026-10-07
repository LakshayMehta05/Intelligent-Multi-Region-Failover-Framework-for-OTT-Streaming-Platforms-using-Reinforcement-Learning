"""
AWS Lambda handler for real-time failover routing decisions.
Uses the REAL trained DQN model via pure NumPy inference (no PyTorch
dependency, fits well within Lambda's deployment size limits).
"""

import json
from pure_python_inference import forward, build_observation, REGION_NAMES


def lambda_handler(event, context=None):
    body = event
    if "body" in event:
        body = json.loads(event["body"]) if isinstance(event["body"], str) else event["body"]

    regions = body["regions"]
    active_region_index = body.get("active_region_index", 0)
    steps_since_switch = body.get("steps_since_switch", 0)

    obs = build_observation(regions, active_region_index, steps_since_switch)
    q_values = forward(obs)
    action = q_values.index(max(q_values))
    switched = action != active_region_index

    n = len(regions)
    names = REGION_NAMES[:n]

    response = {
        "action_region_index": action,
        "action_region_name": names[action],
        "switch_recommended": switched,
        "q_values": {names[i]: float(q_values[i]) for i in range(n)},
        "explanation": (
            f"{'Switch to' if switched else 'Stay on'} {names[action]} "
            f"(Q-value={q_values[action]:.2f}, highest among all {n} regions)"
        ),
    }

    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(response),
    }