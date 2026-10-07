"""
Export the trained DQN's Q-network weights to a lightweight JSON file,
so Lambda can run the REAL trained model via pure NumPy (no torch needed).
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import numpy as np
from stable_baselines3 import DQN

model = DQN.load("results/dqn_realdata_model/best_model.zip")

weights = {}
for name, param in model.q_net.named_parameters():
    weights[name] = param.detach().numpy().tolist()

os.makedirs("aws", exist_ok=True)
with open("aws/model_weights.json", "w") as f:
    json.dump(weights, f)

print("Exported layers:", list(weights.keys()))
print("Saved to aws/model_weights.json")

# Check file size
size_kb = os.path.getsize("aws/model_weights.json") / 1024
print(f"File size: {size_kb:.1f} KB")