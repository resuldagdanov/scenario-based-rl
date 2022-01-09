import torch
torch.manual_seed(0)
import torch.nn as nn
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

class ValueNetwork(nn.Module):
    def __init__(self, device):
        super(ValueNetwork, self).__init__()

        state_size = 64
        action_size = 2
        hidden_size = 64

        self.fc1 = nn.Linear(state_size + action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        self.q_layer = nn.Linear(hidden_size, 1)

        self.to(device)

    def forward(self, state_space, action):
        concatenate_features = torch.cat((state_space, action), dim=1)

        net_out = torch.relu(self.fc1(concatenate_features))
        net_out = torch.relu(self.fc2(net_out))

        q_value = self.q_layer(net_out)

        return q_value