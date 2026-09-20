"""
Explainability layer: logs every failover decision the agent makes,
along with the Q-values behind it (for DQN) and a human-readable
justification, to a structured JSON log file.
"""

import json
import numpy as np
from datetime import datetime, timezone

REGION_NAMES = ["us-east-1", "us-west-2", "eu-west-1", "ap-south-1"]


class DecisionLogger:
    def __init__(self, log_path="results/decision_log.jsonl"):
        self.log_path = log_path
        self._file = open(log_path, "w")

    def log_decision(self, step, obs, action, q_values, active_region_before,
                      degraded_regions, reward, n_regions=4):
        metrics = obs[: n_regions * 5].reshape(n_regions, 5)
        latency, error, stall, cpu, net = metrics[action]

        switched = action != active_region_before
        if switched:
            reason = (
                f"Switched from {REGION_NAMES[active_region_before]} to {REGION_NAMES[action]} "
                f"because {REGION_NAMES[action]} had the highest expected value "
                f"(Q={q_values[action]:.2f}) among all options, with latency={latency:.2f}, "
                f"error_rate={error:.2f}, stall_rate={stall:.2f}."
            )
        else:
            reason = (
                f"Stayed on {REGION_NAMES[action]} because it remained the highest-value "
                f"option (Q={q_values[action]:.2f}); switching away was not worth the "
                f"estimated migration cost."
            )

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "step": step,
            "action_region": REGION_NAMES[action],
            "previous_region": REGION_NAMES[active_region_before],
            "switched": bool(switched),
            "degraded_regions": [REGION_NAMES[i] for i in degraded_regions],
            "q_values": {REGION_NAMES[i]: float(q_values[i]) for i in range(n_regions)},
            "region_metrics": {
                "latency": float(latency), "error_rate": float(error),
                "stall_rate": float(stall), "cpu_util": float(cpu), "network_load": float(net),
            },
            "reward": float(reward),
            "explanation": reason,
        }
        self._file.write(json.dumps(entry) + "\n")
        self._file.flush()
        return reason

    def close(self):
        self._file.close()