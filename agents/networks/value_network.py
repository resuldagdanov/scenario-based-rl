import torch
import torch.nn as nn


class ValueNetwork(nn.Module):
    def __init__(self, device):
        super(ValueNetwork, self).__init__()

        self.fc1 = nn.Linear(64 + 2, 64)
        self.fc2 = nn.Linear(64, 64)

        self.q_layer = nn.Linear(64, 1)

        self.to(device)

    def forward(self, state_space, action):
        concatenate_features = torch.cat((state_space, action), dim=1)

        net_out = torch.relu(self.fc1(concatenate_features))
        net_out = torch.relu(self.fc2(net_out))

        q_value = self.q_layer(net_out)

        return q_value