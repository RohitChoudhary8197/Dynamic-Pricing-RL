import random
import pickle
import numpy as np


class QLearningAgent:

    def __init__(
        self,
        state_size,
        action_size,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
    ):

        self.state_size = state_size
        self.action_size = action_size

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.q_table = {}
        
    def get_q_values(self, state):
        """
        Returns Q-values for a state.
        If state is not present, initialize it with zeros.
        """

        state = tuple(state)

        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_size)

        return self.q_table[state]
    
    def choose_action(self, state):
        """
        Select an action using epsilon-greedy policy.
        """

        q_values = self.get_q_values(state)

        # Exploration
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.action_size - 1)

        # Exploitation
        return int(np.argmax(q_values))
    
    def update_q_table(
        self,
        state,
        action,
        reward,
        next_state,
        done,
    ):
        """
        Update the Q-table using the Q-Learning formula.
        """

        state = tuple(state)
        next_state = tuple(next_state)

        current_q = self.get_q_values(state)[action]

        next_max_q = np.max(self.get_q_values(next_state))

        if done:
            target = reward
        else:
            target = reward + (
                self.discount_factor * next_max_q
            )

        updated_q = current_q + self.learning_rate * (
            target - current_q
        )

        self.q_table[state][action] = updated_q

    def decay_epsilon(self):
        """
        Reduce exploration after each episode.
        """

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

            if self.epsilon < self.epsilon_min:
                self.epsilon = self.epsilon_min

    def save_q_table(
        self,
        filename="models/q_table.pkl",
    ):
        """
        Save the trained Q-table.
        """

        with open(filename, "wb") as file:
            pickle.dump(self.q_table, file)

        print(f"Q-table saved to {filename}")

    def load_q_table(
        self,
        filename="models/q_table.pkl",
    ):
        """
        Load a previously trained Q-table.
        """

        with open(filename, "rb") as file:
            self.q_table = pickle.load(file)

        print(f"Q-table loaded from {filename}")