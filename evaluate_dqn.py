import matplotlib.pyplot as plt

from env.pricing_env import DynamicPricingEnv

from agents.baseline import (
    FixedPricingAgent,
    DiscountPricingAgent,
    RandomPricingAgent
)

from agents.dqn_agent import DQNAgent
env = DynamicPricingEnv()

dqn_agent = DQNAgent(
    state_size=2,
    action_size=env.action_space.n
)

dqn_agent.load_model("models/dqn_model.pth")

# Evaluation mode
dqn_agent.epsilon = 0