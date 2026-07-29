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

fixed_agent = FixedPricingAgent()

discount_agent = DiscountPricingAgent()

random_agent = RandomPricingAgent()

def evaluate_agent(agent, episodes=20):

    rewards = []

    for _ in range(episodes):

        state, _ = env.reset()

        done = False

        total_reward = 0

        while not done:

            if isinstance(agent, DQNAgent):

                action = agent.choose_action(state)

            else:

                action = agent.select_action(state)

            next_state, reward, done, _, _ = env.step(action)

            state = next_state

            total_reward += reward

        rewards.append(total_reward)

    return sum(rewards) / len(rewards)

