"""
Train DQN with coordinated multi-region outage events included in training,
so the agent generalizes to adversarial scenarios (not just independent
single-region failures).
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from env.failover_env import StreamFailoverEnv

MODEL_DIR = "results/dqn_resilient_model"
LOG_DIR = "results/dqn_resilient_logs"
TOTAL_TIMESTEPS = 150_000


class CoordinatedOutageEnv(StreamFailoverEnv):
    """Adds randomized coordinated multi-region outage events on top of normal dynamics."""

    def __init__(self, coordinated_outage_prob=0.01, **kwargs):
        self.coordinated_outage_prob = coordinated_outage_prob
        self._outage_timer = 0
        self._outage_regions = []
        super().__init__(**kwargs)

    def reset(self, *, seed=None, options=None):
        self._outage_timer = 0
        self._outage_regions = []
        return super().reset(seed=seed, options=options)

    def step(self, action):
        obs, reward, term, trunc, info = super().step(action)

        if self._outage_timer > 0:
            self._outage_timer -= 1
            for r in self._outage_regions:
                self.degraded[r] = True
                self.error_rate[r] = max(self.error_rate[r], 0.6)
                self.buffer_stall[r] = max(self.buffer_stall[r], 0.8)
            info["degraded"] = self.degraded.copy()
        elif self._rng.random() < self.coordinated_outage_prob:
            n_affected = self._rng.integers(2, min(3, self.n_regions) + 1)
            self._outage_regions = list(self._rng.choice(self.n_regions, size=n_affected, replace=False))
            self._outage_timer = int(self._rng.integers(20, 50))

        return self._get_obs(), reward, term, trunc, info


def make_env(seed):
    env = CoordinatedOutageEnv(n_regions=4, episode_length=200, seed=seed)
    return Monitor(env)


if __name__ == "__main__":
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    train_env = make_env(seed=0)
    eval_env = make_env(seed=999)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=MODEL_DIR,
        log_path=LOG_DIR,
        eval_freq=5000,
        n_eval_episodes=10,
        deterministic=True,
    )

    model = DQN(
        "MlpPolicy",
        train_env,
        learning_rate=1e-3,
        buffer_size=50_000,
        learning_starts=1000,
        batch_size=64,
        gamma=0.98,
        target_update_interval=500,
        exploration_fraction=0.3,
        exploration_final_eps=0.05,
        verbose=1,
    )

    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_callback)
    model.save(f"{MODEL_DIR}/final_model")
    print("Training complete. Best model saved to", MODEL_DIR)