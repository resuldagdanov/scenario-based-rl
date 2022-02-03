import os
import torch as T
import numpy as np
import random

"""
seed = 0
T.manual_seed(seed)
np.random.seed(seed)
random.seed(seed) 
# for cuda
T.cuda.manual_seed_all(seed)
T.backends.cudnn.deterministic = True
T.backends.cudnn.benchmark = False
"""

import torch.nn as nn

class DQNNetwork(nn.Module):
    def __init__(self, state_size, n_actions):
        super(DQNNetwork, self).__init__()
        
        self.n_actions = n_actions

        # mlp layers
        self.fc = nn.Sequential(
            nn.Linear(state_size + 3, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )

    def forward(self, image_features, fused_input):
        # image feature size: 1000 and fused inputs 3
        concatenate_features = T.cat((image_features, fused_input), dim=1)

        action_values = self.fc(concatenate_features)

        return action_values