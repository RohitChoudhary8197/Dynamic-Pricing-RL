from env.pricing_env import DynamicPricingEnv
from agents.baseline import (
    FixedPricingAgent,
    DiscountPricingAgent,
    RandomPricingAgent
)

NUM_EPISODES = 100


def evaluate_agent(agent, agent_name):
    revenues = []

    for episode in range(NUM_EPISODES):

        env = DynamicPricingEnv()

        state, _ = env.reset()

        done = False

        total_reward = 0

        while not done:

            action = agent.select_action(state)

            state, reward, done, _, info = env.step(action)

            total_reward += reward

        revenues.append(total_reward)

        env.close()

    average_revenue = sum(revenues) / len(revenues)

    print(f"{agent_name}")
    print(f"Average Revenue : ₹{average_revenue:,.2f}")
    print(f"Best Revenue    : ₹{max(revenues):,.2f}")
    print(f"Worst Revenue   : ₹{min(revenues):,.2f}")
    print()


if __name__ == "__main__":

    print("\nRunning Baseline Pricing Strategies...\n")

    evaluate_agent(
        FixedPricingAgent(),
        "Fixed Pricing"
    )

    evaluate_agent(
        DiscountPricingAgent(),
        "Discount Pricing"
    )

    evaluate_agent(
        RandomPricingAgent(),
        "Random Pricing"
    )