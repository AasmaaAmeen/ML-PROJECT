import pandas as pd

FILE = "data/ppo_inventory_results.csv"
df = pd.read_csv(FILE)

summary = (
    df.groupby("blood_group")
      .agg(
          Demand=("predicted_demand", "sum"),
          Order=("recommended_order", "sum"),
          Shortage=("shortage", "sum"),
          Excess=("excess", "sum"),
          Reward=("reward", "sum"),
      )
      .reset_index()
)

total_demand = summary["Demand"].sum()
total_order = summary["Order"].sum()
total_shortage = summary["Shortage"].sum()
total_excess = summary["Excess"].sum()
total_reward = summary["Reward"].sum()

service_level = (
    100 * (1 - total_shortage / total_demand)
    if total_demand > 0 else 100.0
)
order_efficiency = (
    total_demand / total_order
    if total_order > 0 else float("inf")
)

print("\n==========================================")
print("PPO MODEL EVALUATION")
print("==========================================")
print(summary.to_string(index=False))
print("\nTotal predicted demand :", round(total_demand, 2))
print("Total recommended order:", round(total_order, 2))
print("Total shortage         :", round(total_shortage, 2))
print("Total excess           :", round(total_excess, 2))
print("Total reward           :", round(total_reward, 2))
print("Service level          :", f"{service_level:.2f}%")
print("Order efficiency       :", f"{order_efficiency:.2f}")

summary.to_csv("data/ppo_evaluation.csv", index=False)
print("\nSaved: data/ppo_evaluation.csv")
