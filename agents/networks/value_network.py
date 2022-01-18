import torch as T
import numpy as np
import random

seed = 0
T.manual_seed(seed)
np.random.seed(seed)
random.seed(seed) 
# for cuda
T.cuda.manual_seed_all(seed)
T.backends.cudnn.deterministic = True
T.backends.cudnn.benchmark = False

import torch.nn as nn

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
        concatenate_features = T.cat((state_space, action), dim=1)

        net_out = T.relu(self.fc1(concatenate_features))
        net_out = T.relu(self.fc2(net_out))

        q_value = self.q_layer(net_out)

        return q_value