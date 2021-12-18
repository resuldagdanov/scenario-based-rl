import os
import torch as T
import torch.nn as nn
import torch.optim as optim
from resnet_backbone import ResNetBackbone


class CriticNetwork(nn.Module):
    def __init__(self, device, resnet_backbone, state_size, lrvalue, n_actions, max_action, name, checkpoint_dir):
        super(CriticNetwork, self).__init__()
        
        self.device = device
        self.n_actions = n_actions
        self.max_action = max_action
        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        # load pretrained ResNet
        self.resnet_backbone = resnet_backbone

        # fusion data layer
        self.fused_encoder = nn.Linear(3, 128, bias=True)

        # mlp layers
        self.fc1 = nn.Linear(state_size + 128, 64)
        self.fc2 = nn.Linear(64, 64)

        self.q_layer = nn.Linear(64, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=lrvalue)
        self.to(self.device)

    def forward(self, image, fused_inputs, action):
        image_features = self.resnet_backbone(image)
        fused_features = T.relu(self.fused_encoder(fused_inputs))

        # image feature size: 1000 and fused location and speed information size: 128
        concatenate_features = T.cat((image_features, fused_features), dim=1)

        net_out = T.relu(self.fc1(concatenate_features))
        net_out = T.relu(self.fc2(net_out))

        q_value = self.q_layer(net_out)

        return q_value

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))
