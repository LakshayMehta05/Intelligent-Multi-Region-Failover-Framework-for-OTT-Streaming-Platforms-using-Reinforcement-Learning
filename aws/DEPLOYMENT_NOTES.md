# AWS Deployment Notes

## Live Deployment Summary

The trained DQN failover model is deployed as a working AWS Lambda function with public HTTP access and persistent audit logging.

### Architecture

### Components

**Lambda Function:** `streamfailover-inference`
- Runtime: Python 3.13
- Handler: `lambda_handler.lambda_handler`
- Runs the actual trained DQN model (exported weights, pure-Python inference - no heavy ML framework dependency needed for inference)
- Package size: ~60KB (no external dependencies beyond Python stdlib + boto3, which is built into Lambda)

**API Gateway:** HTTP API, open security
- Public endpoint: `https://0r30ov6qrd.execute-api.ap-southeast-2.amazonaws.com/default/streamfailover-inference`
- Accepts POST requests with region health metrics, returns routing decision

**S3 Bucket:** `streamfailover-decision-logs-aakash`
- Every decision permanently logged as JSON (timestamp, input metrics, Q-values, decision, explanation)
- Provides an auditable trail of all routing decisions made

**CloudWatch Dashboard:** `streamfailover-monitoring`
- Tracks Invocations, Errors, and Duration in real time

### How the model got here
1. Trained DQN using Stable-Baselines3 locally (`results/dqn_realdata_model/`)
2. Exported the trained Q-network's weights to JSON (`export_model_weights.py`) - 127KB
3. Rewrote the forward pass in pure Python/NumPy-free code (`aws/pure_python_inference.py`) to avoid Lambda's dependency size constraints while using the exact same trained weights
4. Verified numerically identical outputs between the original PyTorch model and the lightweight deployed version
5. Packaged and deployed to Lambda, wired to API Gateway and S3

### Known limitations / honest scope notes
- Route 53 ARC and multi-region active deployment were not implemented (would incur ongoing cost beyond free tier); the Lambda inference pattern demonstrates the same decision-making capability that would plug into such a system
- Single Lambda instance, not deployed across multiple AWS regions (would require replication for true multi-region failover infrastructure)
- IAM permissions properly scoped: Lambda's execution role now has only `s3:PutObject` permission restricted to the specific `streamfailover-decision-logs-aakash` bucket, following least-privilege principles (verified working after removing the earlier broad `AmazonS3FullAccess` grant).

### Test command
```powershell
$body = @{
    regions = @(
        @{latency=0.6; error_rate=0.4; buffer_stall=0.5; cpu_util=0.5; network_load=0.5},
        @{latency=0.2; error_rate=0.05; buffer_stall=0.1; cpu_util=0.3; network_load=0.3},
        @{latency=0.3; error_rate=0.1; buffer_stall=0.2; cpu_util=0.4; network_load=0.4},
        @{latency=0.25; error_rate=0.08; buffer_stall=0.15; cpu_util=0.35; network_load=0.35}
    )
    active_region_index = 0
    steps_since_switch = 20
} | ConvertTo-Json

Invoke-RestMethod -Uri "https://0r30ov6qrd.execute-api.ap-southeast-2.amazonaws.com/default/streamfailover-inference" -Method Post -Body $body -ContentType "application/json"
```
## Current Deployment Status (Updated)

**Architecture (live):** API Gateway (HTTP API) → Lambda (`streamfailover-inference-mumbai`, ap-southeast-2) → S3 (decision audit logs) + CloudWatch (metrics/logs)

**Model:** Real trained DQN (Stable-Baselines3, trained on real UCC 5G + CODECO traces). Weights exported to `aws/model_weights.json` and run via a dependency-free pure-Python forward pass (`aws/pure_python_inference.py`) — this was required because PyTorch/Stable-Baselines3 exceed Lambda's 250MB unzipped package limit.

**Security:**
- IAM scoped to least-privilege: a custom inline policy (`streamfailover-s3-write-only`) replacing the earlier broad `AmazonS3FullAccess`, granting only `s3:PutObject` on the specific log bucket.
- Input validation (`validate_input()` in `lambda_handler.py`) rejects malformed, missing, out-of-range, or type-mismatched fields with HTTP 400 before any inference runs. Verified live against a simulated attack payload (out-of-range values) — correctly rejected with status 400.

**Known limitations (honest, by design):**
1. **Single-region only.** The original proposal describes multi-region AWS failover infrastructure. This AWS account is restricted to its sign-up region (ap-southeast-2) until "Activate advanced features" is enabled at the account level. This was a deliberate choice to avoid any risk to account credits/free-tier status — not a technical blocker, a safety decision.
2. **Services used vs. proposed:** Implemented with Lambda + API Gateway + S3 + CloudWatch + IAM. DynamoDB Global Tables, SNS/EventBridge, and ECS/Fargate from the original proposal's architecture are not implemented — the current services were sufficient to demonstrate real-time inference, logging, and monitoring end-to-end.
3. **Multi-agent coordination** is a simplified 2-agent decentralized-execution approximation, not a full CTDE (Centralized Training, Decentralized Execution) architecture.

## Current Deployment Status (Updated)

**Architecture (live):**
API Gateway (HTTP API) → Lambda (`streamfailover-inference`, ap-southeast-2)
→ S3 (decision audit logs) + CloudWatch (metrics/logs) + SNS (real-time failover alerts) + DynamoDB (state-transition records)

**Model:** Real trained DQN (Stable-Baselines3, trained on real UCC 5G + CODECO traces). Weights exported to `aws/model_weights.json` and run via a dependency-free pure-Python forward pass (`aws/pure_python_inference.py`) — required because PyTorch/Stable-Baselines3 exceed Lambda's 250MB unzipped package limit.

**Alerting:** Every failover switch decision publishes a message to an SNS topic (`streamfailover-alerts`), triggering a real-time email notification with the routing decision and Q-value reasoning. Verified live.

**State tracking:** Every failover switch decision is also written as a structured record to a DynamoDB table (`streamfailover-state-transitions`) — `transition_id`, `timestamp`, `from_region`, `to_region`, `q_value`. Verified live via console scan (item write confirmed).

**Security:**
- IAM scoped to least-privilege: separate inline policies for S3 (`streamfailover-s3-write-only`), SNS (`streamfailover-sns-publish`), and DynamoDB (`streamfailover-dynamodb-write`) — each grants only the single specific action needed (`PutObject`, `Publish`, `PutItem`) on the single specific resource, replacing any broader default permissions.
- Input validation (`validate_input()` in `lambda_handler.py`) rejects malformed, missing, out-of-range, or type-mismatched fields with HTTP 400 before any inference runs. Verified live against a simulated attack payload (out-of-range values) — correctly rejected with status 400.

**Known limitations (honest, by design):**
1. **Single-region only.** The original proposal describes multi-region AWS failover infrastructure. This AWS account is restricted to its sign-up region (ap-southeast-2) until "Activate advanced features" is enabled at the account level. This was a deliberate choice to avoid any risk to account credits/free-tier status — not a technical blocker, a safety decision. DynamoDB was still implemented (for state-transition logging) since it has genuine value even single-region; true cross-region Global Tables would only make sense once multi-region exists.
2. **ECS/Fargate not implemented.** Unlike EC2, AWS Fargate has no free tier — billing starts immediately per vCPU/memory-second, and common supporting infrastructure (Application Load Balancer, NAT Gateway) carries real recurring cost even at low usage. Given the project's zero-spend constraint, serverless Lambda was used instead, which fulfills the same compute role (event-driven inference) without this cost risk.
3. **No API authentication.** The public endpoint has input validation (rejects malformed/malicious payloads) but no API key or IAM-based auth — anyone with the URL can call it. A production deployment would add this.
4. **Multi-agent coordination** is a simplified 2-agent CTDE (Centralized Training, Decentralized Execution) implementation using QMIX — a centralized mixing network combines both agents' Q-values during training (using global state), while each agent acts from its own local observation only at execution time. This is a genuine CTDE architecture, but uses 2 agents (one per region-pair) rather than a full N-agent system with one agent per region. Training converges (epsilon decay, loss behavior all correct) but the reward curve has not yet fully stabilized — documented as an honest finding, not a bug.