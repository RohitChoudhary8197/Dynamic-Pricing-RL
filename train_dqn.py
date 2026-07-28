import os
import matplotlib.pyplot as plt

from env.pricing_env import DynamicPricingEnv
from agents.dqn_agent import DQNAgent
# =====================================
# Create Environment
# =====================================

env = DynamicPricingEnv()

state_size = 2
action_size = env.action_space.n
# =====================================
# Create DQN Agent
# =====================================

agent = DQNAgent(
    state_size=state_size,
    action_size=action_size
)
# =====================================
# Training Parameters
# =====================================

EPISODES = 5000

episode_rewards = []
# =====================================
# Training Loop
# =====================================

for episode in range(EPISODES):

    state, _ = env.reset()

    done = False

    total_reward = 0

    while not done:

        # Select Action
        action = agent.choose_action(state)

        # Perform Action
        next_state, reward, done, _, info = env.step(action)

        # Store Experience
        agent.remember(
            state,
            action,
            reward,
            next_state,
            done
        )

        # Train DQN
        agent.train()

        # Move to Next State
        state = next_state

        total_reward += reward

    episode_rewards.append(total_reward)

    if (episode + 1) % 100 == 0:

        print(
            f"Episode {episode+1}/{EPISODES}"
            f" | Reward = {total_reward:.2f}"
            f" | Epsilon = {agent.epsilon:.3f}"
        )

# =====================================
# Save Trained Model
# =====================================

os.makedirs("models", exist_ok=True)

agent.save_model("models/dqn_model.pth")

# =====================================
# Plot Training Reward
# =====================================

plt.figure(figsize=(10, 5))

plt.plot(episode_rewards)

plt.title("DQN Training Reward")

plt.xlabel("Episode")

plt.ylabel("Reward")

plt.grid(True)

plt.savefig("models/dqn_training_reward.png")

plt.show()

print("\nDQN Training Completed Successfully.")