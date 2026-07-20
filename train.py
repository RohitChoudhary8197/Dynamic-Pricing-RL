import os
import matplotlib.pyplot as plt

from env.pricing_env import DynamicPricingEnv
from agents.q_learning import QLearningAgent


# ============================================
# Create Environment
# ============================================

env = DynamicPricingEnv()

state_size = 2              # (Inventory, Days Left)
action_size = env.action_space.n


agent = QLearningAgent(
    state_size=state_size,
    action_size=action_size
)

EPISODES = 5000

episode_rewards = []

print("Starting Q-Learning Training...")

best_reward = float("-inf")

for episode in range(EPISODES):

    state, _ = env.reset()

    done = False

    total_reward = 0

    while not done:

        action = agent.choose_action(state)

        next_state, reward, done, _, _ = env.step(action)

        agent.update_q_table(
            state,
            action,
            reward,
            next_state,
            done
        )

        state = next_state

        total_reward += reward

    agent.decay_epsilon()

    episode_rewards.append(total_reward)

if total_reward > best_reward:
    best_reward = total_reward

    if (episode + 1) % 100 == 0:

        print(
            f"Episode {episode+1}/{EPISODES}"
            f" | Reward = {total_reward:.2f}"
            f" | Epsilon = {agent.epsilon:.3f}"
        )


os.makedirs("models", exist_ok=True)

agent.save_q_table("models/q_table.pkl")

average_reward = sum(episode_rewards) / len(episode_rewards)
plt.figure(figsize=(10,5))

plt.plot(
    episode_rewards,
    linewidth=2,
    label="Training Reward"
)
plt.legend()

plt.title("Training Reward")

plt.xlabel("Episode")

plt.ylabel("Revenue")

plt.grid(True)

plt.savefig("models/training_reward.png")
plt.close()
plt.show()

print("\nTraining Completed Successfully.")

print(f"\nAverage Revenue : ₹{average_reward:,.2f}")
print(f"Best Revenue    : ₹{best_reward:,.2f}")