import pandas as pd
from stable_baselines3 import PPO
from rl.environment import BloodInventoryEnv

FORECAST = "data/tft_forecast.csv"
MODEL = "rl/ppo_blood_inventory.zip"
OUT = "data/ppo_inventory_results.csv"

df = pd.read_csv(FORECAST)
model = PPO.load(MODEL)
print("PPO model loaded successfully.")

results = []

for blood_group, g in df.groupby("blood_group"):
    g = g.sort_values("date").reset_index(drop=True)
    forecast = g["tft_predicted_demand"].astype(float).values

    env = BloodInventoryEnv(
        forecast,
        initial_inventory=10,
        max_order=20,
        safety_stock=5,
    )
    obs, _ = env.reset()

    for i in range(len(g)):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)

        results.append({
            "date": g.loc[i, "date"],
            "blood_group": blood_group,
            "predicted_demand": float(info["demand"]),
            "recommended_order": float(info["order"]),
            "remaining_inventory": float(info["inventory"]),
            "shortage": float(info["shortage"]),
            "excess": float(info["excess"]),
            "reward": float(reward),
        })
        if done:
            break

out = pd.DataFrame(results)
out.to_csv(OUT, index=False)

print("\nPPO RESULTS")
print("Total recommended units:", round(out["recommended_order"].sum(), 2))
print("Total shortage:", round(out["shortage"].sum(), 2))
print("Total excess:", round(out["excess"].sum(), 2))
print("\nOrders by blood group:")
print(out.groupby("blood_group")["recommended_order"].sum())
print("Saved:", OUT)
