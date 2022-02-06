import torch as T
import torch.nn as nn
import numpy as np


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


class DQNNetwork2(nn.Module):
    def __init__(self, state_size, n_actions, device):
        super(DQNNetwork2, self).__init__()
        
        self.n_actions = n_actions
        self.device = device

        # mlp layers
        self.fc = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )

    def forward(self, image_features):
        action_values = self.fc(image_features)
        return action_values

    def normalize_rgb(self, x):
        x = x.clone()
        x[:, 0] = (x[:, 0] - 0.485) / 0.229
        x[:, 1] = (x[:, 1] - 0.456) / 0.224
        x[:, 2] = (x[:, 2] - 0.406) / 0.225
        return x

    def image_to_dnn_input(self, image):
        # convert width height channel to channel width height
        image = np.array(image.transpose((2, 0, 1)), np.float32)
        # BGRA to BGR
        image = image[:3, :, :]
        # BGR to RGB
        image = image[::-1, :, :]
        # normalize to 0 - 1
        image = image / 255
        # convert image to torch tensor
        image = T.from_numpy(image.copy()).unsqueeze(0)
        
        # normalize input image (using default torch normalization technique)
        image = self.normalize_rgb(image)
        image = image.to(self.device)

        return image