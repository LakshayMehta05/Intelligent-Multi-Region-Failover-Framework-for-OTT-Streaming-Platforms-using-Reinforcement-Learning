import numpy as np
import gymnasium as gym
from gymnasium import spaces


class StreamFailoverEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        n_regions: int = 4,
        episode_length: int = 200,
        seed: int | None = None,
        region_cost_tiers: np.ndarray | None = None,
        failure_prob: float = 0.02,
        recovery_prob: float = 0.10,
        reward_weights: dict | None = None,
    ):
        super().__init__()
        self.n_regions = n_regions
        self.episode_length = episode_length
        self.failure_prob = failure_prob
        self.recovery_prob = recovery_prob

        self.region_cost_tiers = (
            region_cost_tiers
            if region_cost_tiers is not None
            else np.linspace(0.7, 1.3, n_regions)
        )

        self.w = reward_weights or dict(
            qoe=1.0,
            switch_cost=0.5,
            flap_penalty=1.0,
            infra_cost=0.2,
            availability=1.0,
        )

        self.metrics_per_region = 5
        obs_dim = self.n_regions * self.metrics_per_region + self.n_regions + 1
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.n_regions)

        self.region_traces = None  # set via load_real_traces()
        self._trace_idx = 0

        self._rng = np.random.default_rng(seed)
        self.reset(seed=seed)

    def load_real_traces(self, trace_paths: list):
        """Load real 5G trace CSVs, one per region, to drive dynamics instead of pure synthetic random walk."""
        from data.trace_loader import load_region_traces
        assert len(trace_paths) == self.n_regions, "Need one trace file per region"
        self.region_traces = load_region_traces(trace_paths)
        self._trace_idx = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.t = 0
        self.active_region = int(self._rng.integers(0, self.n_regions))
        self.steps_since_switch = self.episode_length

        self.latency = self._rng.uniform(0.1, 0.3, self.n_regions)
        self.error_rate = self._rng.uniform(0.0, 0.05, self.n_regions)
        self.buffer_stall = self._rng.uniform(0.0, 0.05, self.n_regions)
        self.cpu_util = self._rng.uniform(0.2, 0.5, self.n_regions)
        self.network_load = self._rng.uniform(0.2, 0.5, self.n_regions)

        self.degraded = np.zeros(self.n_regions, dtype=bool)

        return self._get_obs(), {}

    def _get_obs(self):
        metrics = np.stack(
            [self.latency, self.error_rate, self.buffer_stall, self.cpu_util, self.network_load],
            axis=1,
        ).flatten()
        active_onehot = np.zeros(self.n_regions, dtype=np.float32)
        active_onehot[self.active_region] = 1.0
        time_feat = np.array(
            [min(self.steps_since_switch / self.episode_length, 1.0)], dtype=np.float32
        )
        return np.concatenate([metrics, active_onehot, time_feat]).astype(np.float32)

    def _simulate_region_dynamics(self):
        n = self.n_regions

        if self.region_traces is not None:
            idx = self._trace_idx % min(t["length"] for t in self.region_traces)
            for i in range(n):
                trace = self.region_traces[i]
                self.latency[i] = trace["latency"][idx]
                self.error_rate[i] = trace["error_rate"][idx]
                self.buffer_stall[i] = trace["buffer_stall"][idx]
                self.cpu_util[i] = np.clip(self.cpu_util[i] + self._rng.normal(0, 0.02), 0.05, 1.0)
                self.network_load[i] = np.clip(self.network_load[i] + self._rng.normal(0, 0.02), 0.05, 1.0)
                self.degraded[i] = (self.error_rate[i] > 0.3) or (self.buffer_stall[i] > 0.8)
            self._trace_idx += 1
            return

        for i in range(n):
            if not self.degraded[i] and self._rng.random() < self.failure_prob:
                self.degraded[i] = True
            elif self.degraded[i] and self._rng.random() < self.recovery_prob:
                self.degraded[i] = False

        drift = self._rng.normal(0, 0.02, (5, n))
        self.latency = np.clip(self.latency + drift[0], 0.05, 1.0)
        self.error_rate = np.clip(self.error_rate + drift[1] * 0.5, 0.0, 1.0)
        self.buffer_stall = np.clip(self.buffer_stall + drift[2] * 0.5, 0.0, 1.0)
        self.cpu_util = np.clip(self.cpu_util + drift[3], 0.05, 1.0)
        self.network_load = np.clip(self.network_load + drift[4], 0.05, 1.0)

        for i in range(n):
            if self.degraded[i]:
                self.latency[i] = np.clip(self.latency[i] + 0.4, 0, 1.0)
                self.error_rate[i] = np.clip(self.error_rate[i] + 0.5, 0, 1.0)
                self.buffer_stall[i] = np.clip(self.buffer_stall[i] + 0.5, 0, 1.0)
                self.cpu_util[i] = np.clip(self.cpu_util[i] + 0.3, 0, 1.0)

    def step(self, action: int):
        assert self.action_space.contains(action)
        self._simulate_region_dynamics()

        switched = action != self.active_region
        if switched:
            self.active_region = action
            self.steps_since_switch = 0
        else:
            self.steps_since_switch += 1

        i = self.active_region
        w = self.w

        qoe_penalty = self.error_rate[i] + self.buffer_stall[i] + 0.3 * self.latency[i]
        availability_bonus = 1.0 - float(self.degraded[i])
        switch_cost = 1.0 if switched else 0.0
        flap_penalty = 1.0 if (switched and self.steps_since_switch == 0 and self._recent_switch()) else 0.0
        infra_cost = self.region_cost_tiers[i] * (0.5 * self.cpu_util[i] + 0.5 * self.network_load[i])

        reward = (
            -w["qoe"] * qoe_penalty
            - w["switch_cost"] * switch_cost
            - w["flap_penalty"] * flap_penalty
            - w["infra_cost"] * infra_cost
            + w["availability"] * availability_bonus
        )

        self.t += 1
        terminated = False
        truncated = self.t >= self.episode_length

        info = {
            "active_region": i,
            "switched": switched,
            "degraded": self.degraded.copy(),
            "qoe_penalty": qoe_penalty,
            "infra_cost": infra_cost,
        }
        return self._get_obs(), float(reward), terminated, truncated, info

    def _recent_switch(self):
        return getattr(self, "_last_switch_step", -999) == self.t - 1

    def render(self):
        print(
            f"t={self.t} active={self.active_region} "
            f"degraded={np.where(self.degraded)[0].tolist()} "
            f"lat={self.latency.round(2)} err={self.error_rate.round(2)}"
        )