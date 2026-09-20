"""
Train DQN on StreamFailoverEnv using REAL 5G trace data
(UCC 5G production dataset, recorded during real Netflix streaming).
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from env.failover_env import StreamFailoverEnv

MODEL_DIR = "results/dqn_realdata_model"
LOG_DIR = "results/dqn_realdata_logs"
TOTAL_TIMESTEPS = 100_000

TRACE_PATHS = [
    "data/5G_dataset/extracted/5G-production-dataset/Netflix/Static/Season3-StrangerThings/B_2019.11.26_13.50.48.csv",
    "data/5G_dataset/extracted/5G-production-dataset/Netflix/Static/Season3-StrangerThings/B_2019.12.03_08.02.05.csv",
    "data/5G_dataset/extracted/5G-production-dataset/Netflix/Static/animated-RickandMorty/B_2019.11.26_08.02.38.csv",
    "data/5G_dataset/extracted/5G-production-dataset/Netflix/Static/animated-RickandMorty/B_2019.11.28_08.02.19.csv",
]

def make_env(seed):
    env = StreamFailoverEnv(n_regions=4, episode_length=200, seed=seed)
    env.load_real_traces(TRACE_PATHS)
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