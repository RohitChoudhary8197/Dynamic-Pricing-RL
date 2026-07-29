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

# =====================================
# Evaluate All Agents
# =====================================

print("\nEvaluating Agents...\n")

fixed_score = evaluate_agent(fixed_agent)

discount_score = evaluate_agent(discount_agent)

random_score = evaluate_agent(random_agent)

dqn_score = evaluate_agent(dqn_agent)

# =====================================
# Print Results
# =====================================

print("=" * 45)

print(f"Fixed Pricing     : ₹{fixed_score:.2f}")

print(f"Discount Pricing  : ₹{discount_score:.2f}")

print(f"Random Pricing    : ₹{random_score:.2f}")

print(f"DQN Agent         : ₹{dqn_score:.2f}")

print("=" * 45)

# =====================================
# Plot Revenue Comparison
# =====================================

agents = [
    "Fixed",
    "Discount",
    "Random",
    "DQN"
]

revenues = [
    fixed_score,
    discount_score,
    random_score,
    dqn_score
]

plt.figure(figsize=(8, 5))

plt.bar(agents, revenues)

plt.title("Revenue Comparison")

plt.xlabel("Pricing Strategy")

plt.ylabel("Average Revenue")

plt.grid(axis="y")

plt.savefig("models/revenue_comparison.png")

plt.show()
print("\nEvaluation Completed Successfully.")