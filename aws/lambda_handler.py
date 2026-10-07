"""
AWS Lambda handler for real-time failover routing decisions.
Uses the REAL trained DQN model via pure-Python inference (no dependencies).
Logs every decision to S3 for permanent, auditable record-keeping.
"""

import json
import os
import boto3
from datetime import datetime, timezone
from pure_python_inference import forward, build_observation, REGION_NAMES

S3_BUCKET = "streamfailover-decision-logs-aakash"
s3_client = boto3.client("s3")


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

    # Log this decision to S3 for a permanent audit trail
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
        # Don't fail the whole request if logging fails - log the error but still return the decision
        print(f"S3 logging failed: {e}")

    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(response),
    }