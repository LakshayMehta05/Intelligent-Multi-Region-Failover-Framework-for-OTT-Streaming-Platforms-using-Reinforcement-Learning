"""
Train a PPO agent on StreamFailoverEnv, for comparison against DQN.
Uses VecNormalize to normalize observations/rewards, which PPO is
much more sensitive to than DQN.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from env.failover_env import StreamFailoverEnv

MODEL_DIR = "results/ppo_model"
LOG_DIR = "results/ppo_logs"
TOTAL_TIMESTEPS = 300_000

def make_env(seed):
    def _init():
        env = StreamFailoverEnv(n_regions=4, episode_length=200, seed=seed)
        return Monitor(env)
    return _init

if __name__ == "__main__":
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    train_env = DummyVecEnv([make_env(seed=0)])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_reward=10.0)

    eval_env = DummyVecEnv([make_env(seed=999)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=MODEL_DIR,
        log_path=LOG_DIR,
        eval_freq=5000,
        n_eval_episodes=10,
        deterministic=True,
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        gamma=0.98,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
    )

    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_callback)
    model.save(f"{MODEL_DIR}/final_model")
    train_env.save(f"{MODEL_DIR}/vecnormalize.pkl")
    print("Training complete. Best model saved to", MODEL_DIR)