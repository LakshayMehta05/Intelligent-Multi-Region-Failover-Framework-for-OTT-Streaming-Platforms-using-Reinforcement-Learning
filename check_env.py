import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.failover_env import StreamFailoverEnv
from stable_baselines3.common.env_checker import check_env

env = StreamFailoverEnv(n_regions=4, episode_length=200, seed=1)
check_env(env, warn=True)
print("Environment passed SB3 check!")