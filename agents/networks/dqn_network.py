import os
import torch as T
T.manual_seed(0)
import torch.nn as nn
T.backends.cudnn.benchmark = False
#T.use_deterministic_algorithms(True)

class DQNNetwork(nn.Module):
    def __init__(self, device, state_size, n_actions, name, checkpoint_dir):
        super(DQNNetwork, self).__init__()
        
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = checkpoint_dir
        
        # fusion data layer
        self.fused_encoder = nn.Linear(3, 128, bias=True)

        # mlp layers
        self.fc = nn.Sequential(
            nn.Linear(state_size + 128, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )

        self.to(device)

    def forward(self, image_features, fused_input):
        fused_features = T.relu(self.fused_encoder(fused_input))

        # image feature size: 1000 and fused location and speed information size: 128
        concatenate_features = T.cat((image_features, fused_features), dim=1)

        action_values = self.fc(concatenate_features)

        return action_values

    def save_checkpoint(self, episode_number):
        checkpoint_file = os.path.join(self.checkpoint_dir, self.name + "-ep_" + str(episode_number))
        T.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, episode_number):
        checkpoint_file = os.path.join(self.checkpoint_dir, self.name + "-ep_" + str(episode_number))
        self.load_state_dict(T.load(checkpoint_file))