import torch

from models.dqn_model import DQN

model = DQN(
    state_size=2,
    action_size=5
)

sample = torch.tensor(
    [[100.0, 30.0]]
)

output = model(sample)

print(output)
print(output.shape)