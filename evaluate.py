import matplotlib.pyplot as plt

from env.pricing_env import DynamicPricingEnv

from agents.baseline import (
    FixedPricingAgent,
    DiscountPricingAgent,
    RandomPricingAgent,
)

from agents.q_learning import QLearningAgent
env = DynamicPricingEnv()
def evaluate_agent(agent, episodes=100):

    revenues = []

    for _ in range(episodes):

        state, _ = env.reset()

        done = False

        total_reward = 0

        while not done:

            # Q-Learning Agent
            if isinstance(agent, QLearningAgent):
                action = agent.choose_action(state)

            # Baseline Agents
            else:
                action = agent.select_action(state)

            next_state, reward, done, _, _ = env.step(action)

            state = next_state

            total_reward += reward

        revenues.append(total_reward)

    return sum(revenues) / len(revenues)

fixed_agent = FixedPricingAgent()

discount_agent = DiscountPricingAgent()

random_agent = RandomPricingAgent()

q_agent = QLearningAgent(
    state_size=2,
    action_size=env.action_space.n,
)

q_agent.load_q_table("models/q_table.pkl")

# Evaluation ke time exploration band
q_agent.epsilon = 0

print("Evaluating Agents...\n")

fixed_score = evaluate_agent(fixed_agent)

discount_score = evaluate_agent(discount_agent)

random_score = evaluate_agent(random_agent)

q_score = evaluate_agent(q_agent)

print("=" * 10)

print(f"Fixed Pricing    : ₹{fixed_score:,.2f}")

print(f"Discount Pricing : ₹{discount_score:,.2f}")

print(f"Random Pricing   : ₹{random_score:,.2f}")

print(f"Q-Learning       : ₹{q_score:,.2f}")

print("=" * 10)
agents = [
    "Fixed",
    "Discount",
    "Random",
    "Q-Learning",
]

revenues = [
    fixed_score,
    discount_score,
    random_score,
    q_score,
]

plt.figure(figsize=(8,5))

bars = plt.bar(agents, revenues)

plt.title("Revenue Comparison")

plt.ylabel("Revenue")

for bar in bars:

    height = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width()/2,
        height,
        f"{height:.0f}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()

plt.savefig("models/revenue_comparison.png")

plt.show()