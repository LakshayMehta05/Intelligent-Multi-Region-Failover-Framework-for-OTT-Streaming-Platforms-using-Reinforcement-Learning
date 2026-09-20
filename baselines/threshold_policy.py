import numpy as np


class ThresholdFailoverPolicy:
    def __init__(self, n_regions: int, degrade_threshold: float = 0.5):
        self.n_regions = n_regions
        self.degrade_threshold = degrade_threshold

    def _unpack_obs(self, obs: np.ndarray):
        n = self.n_regions
        metrics = obs[: n * 5].reshape(n, 5)  # latency, error, stall, cpu, net
        active_onehot = obs[n * 5 : n * 5 + n]
        active_region = int(np.argmax(active_onehot))
        return metrics, active_region

    def _health_score(self, metrics_row):
        # Lower is better; simple weighted sum of the 5 normalized metrics
        latency, error, stall, cpu, net = metrics_row
        return 0.25 * latency + 0.35 * error + 0.35 * stall + 0.025 * cpu + 0.025 * net

    def act(self, obs: np.ndarray) -> int:
        metrics, active_region = self._unpack_obs(obs)
        scores = np.array([self._health_score(metrics[i]) for i in range(self.n_regions)])

        if scores[active_region] < self.degrade_threshold:
            return active_region  # current region still fine, stay put

        # current region degraded past threshold -> switch to best-scoring other region
        return int(np.argmin(scores))