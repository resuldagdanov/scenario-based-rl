import os
import torch as T
import torch.nn as nn
import torch.optim as optim


class CriticNetwork(nn.Module):
    def __init__(self, device, state_size, n_actions, name, checkpoint_dir):
        super(CriticNetwork, self).__init__()

        fused_size = 128
        action_size = 2
        hidden_size = 64
        
        self.device = device
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = checkpoint_dir

        # fusion data layer
        self.fused_encoder = nn.Linear(3, fused_size, bias=True)

        # mlp layers
        self.fc1 = nn.Linear(state_size + fused_size + action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        self.q_layer = nn.Linear(hidden_size, 1)

        self.to(self.device)

    def forward(self, image_features, fused_input, action):
        fused_features = T.relu(self.fused_encoder(fused_input))

        # image feature size: 1000, and fused location and speed information size: 128, and action values: 2
        concatenate_features = T.cat((image_features, fused_features, action), dim=1)

        net_out = T.relu(self.fc1(concatenate_features))
        net_out = T.relu(self.fc2(net_out))

        q_value = self.q_layer(net_out)

        return q_value

    def save_checkpoint(self, episode_number):
        checkpoint_file = os.path.join(self.checkpoint_dir, self.name + "-ep_" + str(episode_number))
        T.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, episode_number):
        checkpoint_file = os.path.join(self.checkpoint_dir, self.name + "-ep_" + str(episode_number))
        self.load_state_dict(T.load(checkpoint_file))
