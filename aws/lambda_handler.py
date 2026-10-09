"""
AWS Lambda handler for real-time failover routing decisions.
Uses the REAL trained DQN model via pure-Python inference (no dependencies).
Logs every decision to S3. Includes input validation for security.
"""

import json
import os
import boto3
from datetime import datetime, timezone
from pure_python_inference import forward, build_observation, REGION_NAMES

S3_BUCKET = "streamfailover-decision-logs-aakash"
s3_client = boto3.client("s3")

EXPECTED_N_REGIONS = 4
REQUIRED_METRIC_KEYS = {"latency", "error_rate", "buffer_stall", "cpu_util", "network_load"}


def validate_input(body):
    """Reject malformed, out-of-range, or malicious input before it reaches the model."""
    if not isinstance(body, dict):
        return "Request body must be a JSON object"

    regions = body.get("regions")
    if not isinstance(regions, list) or len(regions) != EXPECTED_N_REGIONS:
        return f"'regions' must be a list of exactly {EXPECTED_N_REGIONS} items"

    for i, r in enumerate(regions):
        if not isinstance(r, dict):
            return f"Region {i} must be an object"
        missing = REQUIRED_METRIC_KEYS - set(r.keys())
        if missing:
            return f"Region {i} missing keys: {missing}"
        for k in REQUIRED_METRIC_KEYS:
            v = r[k]
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                return f"Region {i} field '{k}' must be a number"
            if not (0.0 <= v <= 1.0):
                return f"Region {i} field '{k}' must be between 0 and 1 (got {v})"

    active_region_index = body.get("active_region_index", 0)
    if not isinstance(active_region_index, int) or not (0 <= active_region_index < EXPECTED_N_REGIONS):
        return f"'active_region_index' must be an integer between 0 and {EXPECTED_N_REGIONS - 1}"

    steps_since_switch = body.get("steps_since_switch", 0)
    if not isinstance(steps_since_switch, (int, float)) or steps_since_switch < 0:
        return "'steps_since_switch' must be a non-negative number"

    return None  # valid


def lambda_handler(event, context=None):
    body = event
    if "body" in event:
        try:
            body = json.loads(event["body"]) if isinstance(event["body"], str) else event["body"]
        except (json.JSONDecodeError, TypeError):
            return {
                "statusCode": 400,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({"error": "Invalid JSON in request body"}),
            }

    error = validate_input(body)
    if error:
        return {
            "statusCode": 400,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": error}),
        }

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

    try:
        timestamp = datetime.now(timezone.utc).isoformat()
        log_entry = {"timestamp": timestamp, "input": body, "decision": response}
        key = f"decisions/{timestamp.replace(':', '-')}.json"
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=key,
            Body=json.dumps(log_entry),
            ContentType="application/json",
        )
    except Exception as e:
        print(f"S3 logging failed: {e}")

    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(response),
    }