import os
import pandas as pd
from stable_baselines3 import PPO
from rl.environment import BloodInventoryEnv

FORECAST = "data/tft_forecast.csv"
MODEL = "rl/ppo_blood_inventory"

df = pd.read_csv(FORECAST)

# Train one policy on all 8 blood-group forecast sequences.
# A policy is trained against a representative combined sequence.
sequences = [
    g["tft_predicted_demand"].astype(float).values
    for _, g in df.groupby("blood_group")
]
forecast = max(sequences, key=len)

env = BloodInventoryEnv(
    forecast,
    initial_inventory=10,
    max_order=20,
    safety_stock=5,
)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    seed=42,
    learning_rate=3e-4,
    n_steps=128,
    batch_size=64,
    gamma=0.99,
)

model.learn(total_timesteps=30000)
os.makedirs("rl", exist_ok=True)
model.save(MODEL)
print("Saved:", MODEL + ".zip")
