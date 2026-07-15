import gymnasium as gym
from gymnasium import spaces
import numpy as np

from utils.config import (
    INITIAL_INVENTORY,
    MAX_DAYS,
    PRICE_LEVELS,
    UNSOLD_PENALTY
)

from utils.demand import calculate_demand


class DynamicPricingEnv(gym.Env):
    """
    Dynamic Pricing Environment

    State:
        [Remaining Inventory, Remaining Days]

    Actions:
        0 -> ₹3000
        1 -> ₹3500
        2 -> ₹4000
        3 -> ₹4500
        4 -> ₹5000

    Reward:
        Revenue - Unsold Inventory Penalty
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self):

        super().__init__()

        # Observation Space
        self.observation_space = spaces.Box(
            low=np.array([0, 0]),
            high=np.array([INITIAL_INVENTORY, MAX_DAYS]),
            dtype=np.int32
        )

        # 5 Price Levels
        self.action_space = spaces.Discrete(len(PRICE_LEVELS))

        self.reset()

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.inventory = INITIAL_INVENTORY

        self.days_left = MAX_DAYS

        self.total_revenue = 0

        self.current_day = 1

        state = np.array(
            [
                self.inventory,
                self.days_left
            ],
            dtype=np.int32
        )

        info = {}

        return state, info

    def step(self, action):

        price = PRICE_LEVELS[action]

        demand = calculate_demand(
            price,
            self.days_left
        )

        tickets_sold = min(
            demand,
            self.inventory
        )

        revenue = tickets_sold * price

        self.total_revenue += revenue

        self.inventory -= tickets_sold

        self.days_left -= 1

        self.current_day += 1

        reward = revenue

        terminated = False

        truncated = False

        if self.inventory == 0:

            terminated = True

        if self.days_left == 0:

            terminated = True

            reward -= (
                self.inventory *
                UNSOLD_PENALTY
            )

        next_state = np.array(
            [
                self.inventory,
                self.days_left
            ],
            dtype=np.int32
        )

        info = {

            "price": price,

            "tickets_sold": tickets_sold,

            "demand": demand,

            "revenue": revenue,

            "total_revenue": self.total_revenue

        }

        return (
            next_state,
            reward,
            terminated,
            truncated,
            info
        )

    def render(self):

        print("=" * 40)

        print(f"Day : {self.current_day}")

        print(f"Inventory : {self.inventory}")

        print(f"Days Left : {self.days_left}")

        print(f"Revenue : {self.total_revenue}")

        print("=" * 40)

    def close(self):
        pass