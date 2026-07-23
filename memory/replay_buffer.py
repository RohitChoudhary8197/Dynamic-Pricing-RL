import random
from collections import deque
import numpy as np


class ReplayBuffer:
    """
    Stores past experiences for DQN training.
    """

    def __init__(self, capacity=10000):

        self.memory = deque(maxlen=capacity)

    def add(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):
        """
        Save one experience.
        """

        self.memory.append(
            (
                state,
                action,
                reward,
                next_state,
                done
            )
        )

    def sample(self, batch_size):
        """
        Randomly sample experiences.
        """

        batch = random.sample(
            self.memory,
            batch_size
        )

        states, actions, rewards, next_states, dones = zip(*batch)

        return (

            np.array(states),

            np.array(actions),

            np.array(rewards),

            np.array(next_states),

            np.array(dones)

        )

    def __len__(self):
        """
        Returns current memory size.
        """

        return len(self.memory)