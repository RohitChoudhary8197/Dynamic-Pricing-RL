import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.dqn_model import DQN
from memory.replay_buffer import ReplayBuffer


class DQNAgent:
    """
    Deep Q-Network Agent

    Responsible for:
    - Choosing actions
    - Storing experiences
    - Training Neural Network
    """

    def __init__(
        self,
        state_size,
        action_size,
        learning_rate=0.001,
        gamma=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        batch_size=64,
        memory_size=10000,
    ):

        # ===============================
        # Basic Parameters
        # ===============================

        self.state_size = state_size
        self.action_size = action_size

        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.batch_size = batch_size

        # ===============================
        # Replay Memory
        # ===============================

        self.memory = ReplayBuffer(memory_size)

        # ===============================
        # Neural Network
        # ===============================

        self.model = DQN(
            state_size,
            action_size
        )

        # ===============================
        # Optimizer
        # ===============================

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate
        )

        # ===============================
        # Loss Function
        # ===============================

        self.loss_function = nn.MSELoss()

        print("\nDQN Agent Initialized Successfully.")

    def choose_action(self, state):
        """
        Select an action using epsilon-greedy policy.
        """

        # Exploration
        if random.random() < self.epsilon:
            return random.randint(
                0,
                self.action_size - 1
            )

        # Exploitation
        state = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():

            q_values = self.model(state)

        return torch.argmax(q_values).item()

    def remember(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):
        """
        Store experience into Replay Buffer.
        """

        self.memory.add(
            state,
            action,
            reward,
            next_state,
            done
        )

    def train(self):
        """
        Train the DQN using experiences from Replay Buffer.
        """

        # Not enough experiences yet
        if len(self.memory) < self.batch_size:
            return

        # Sample mini-batch
        states, actions, rewards, next_states, dones = self.memory.sample(
            self.batch_size
        )

        # Convert to tensors
        states = torch.FloatTensor(states)
        next_states = torch.FloatTensor(next_states)

        actions = torch.LongTensor(actions).unsqueeze(1)

        rewards = torch.FloatTensor(rewards)

        dones = torch.FloatTensor(dones)

        # Current Q-values
        current_q = self.model(states).gather(1, actions).squeeze()

        # Next Q-values
        with torch.no_grad():
            next_q = self.model(next_states).max(1)[0]

        # Target Q-values
        target_q = rewards + (
            (1 - dones) * self.gamma * next_q
        )

        # Loss
        loss = self.loss_function(
            current_q,
            target_q
        )

        # Backpropagation
        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

        # Reduce exploration
        if self.epsilon > self.epsilon_min:

            self.epsilon *= self.epsilon_decay

            if self.epsilon < self.epsilon_min:

                self.epsilon = self.epsilon_min

    def save_model(
        self,
        filename="models/dqn_model.pth"
    ):
        """
        Save the trained DQN model.
        """

        torch.save(
            self.model.state_dict(),
            filename
        )

        print(f"DQN model saved to {filename}")


    def load_model(
        self,
        filename="models/dqn_model.pth"
    ):
        """
        Load a trained DQN model.
        """

        self.model.load_state_dict(
            torch.load(filename)
        )

        self.model.eval()

        print(f"DQN model loaded from {filename}")