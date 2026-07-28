from agents.dqn_agent import DQNAgent
import random

# Create Agent
agent = DQNAgent(
    state_size=2,
    action_size=5
)

# Fill Replay Buffer
for _ in range(100):

    state = [
        random.randint(0, 100),
        random.randint(0, 30)
    ]

    action = random.randint(0, 4)

    reward = random.randint(100, 500)

    next_state = [
        random.randint(0, 100),
        random.randint(0, 30)
    ]

    done = random.choice([True, False])

    agent.remember(
        state,
        action,
        reward,
        next_state,
        done
    )

print("Memory Size :", len(agent.memory))

# Train One Step
agent.train()

print("Training Step Completed Successfully.")

print("Current Epsilon :", agent.epsilon)

# Save Model
agent.save_model()

# Load Model
agent.load_model()

print("\nModel Save & Load Test Passed Successfully.")