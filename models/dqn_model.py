import torch
import torch.nn as nn


class DQN(nn.Module):
    """
    Deep Q-Network

    Input:
        Inventory
        Days Left

    Output:
        Q-values for all pricing actions
    """

    def __init__(self, state_size, action_size):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(state_size, 64),

            nn.ReLU(),

            nn.Linear(64, 64),

            nn.ReLU(),

            nn.Linear(64, action_size)

        )

    def forward(self, x):

        return self.network(x)