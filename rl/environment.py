import gymnasium as gym
from gymnasium import spaces
import numpy as np

class BloodInventoryEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        demand_forecast,
        initial_inventory=10.0,
        max_order=20.0,
        safety_stock=5.0,
        holding_cost=0.05,
        order_cost=0.10,
        shortage_cost=12.0,
        excess_cost=2.0,
    ):
        super().__init__()
        self.forecast = np.asarray(demand_forecast, dtype=np.float32)
        self.initial_inventory = float(initial_inventory)
        self.max_order = float(max_order)
        self.default_safety_stock = float(safety_stock)
        self.holding_cost = holding_cost
        self.order_cost = order_cost
        self.shortage_cost = shortage_cost
        self.excess_cost = excess_cost

        self.action_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([self.max_order], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=0.0, high=np.inf, shape=(4,), dtype=np.float32
        )

    def _obs(self):
        d = float(self.forecast[min(self.t, len(self.forecast)-1)])
        remaining = max(0, len(self.forecast) - self.t)
        return np.array(
            [self.inventory, d, self.safety_stock, remaining],
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.inventory = self.initial_inventory
        self.safety_stock = self.default_safety_stock
        return self._obs(), {}

    def step(self, action):
        order = float(np.clip(action[0], 0, self.max_order))
        demand = float(self.forecast[min(self.t, len(self.forecast)-1)])

        self.inventory += order
        served = min(self.inventory, demand)
        shortage = max(0.0, demand - self.inventory)
        self.inventory -= served

        # Simple perishability proxy: stock above 2x safety stock is excess.
        excess = max(0.0, self.inventory - 2.0 * self.safety_stock)

        reward = -(
            self.order_cost * order
            + self.holding_cost * self.inventory
            + self.shortage_cost * shortage
            + self.excess_cost * excess
        )

        self.t += 1
        terminated = self.t >= len(self.forecast)

        info = {
            "order": order,
            "demand": demand,
            "shortage": shortage,
            "excess": excess,
            "inventory": self.inventory,
            "reward": reward,
        }
        return self._obs(), reward, terminated, False, info
