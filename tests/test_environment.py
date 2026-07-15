import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.pricing_env import DynamicPricingEnv


env = DynamicPricingEnv()

state, info = env.reset()

print("Initial State :", state)

done = False

while not done:

    action = env.action_space.sample()

    next_state, reward, done, _, info = env.step(action)

    print("-" * 50)

    print("Next State :", next_state)

    print("Price :", info["price"])

    print("Demand :", info["demand"])

    print("Sold :", info["tickets_sold"])

    print("Reward :", reward)

    print("Revenue :", info["total_revenue"])

env.close()