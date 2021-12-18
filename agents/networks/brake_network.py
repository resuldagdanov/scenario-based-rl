import os
import torch as T
import torch.nn as nn
from torchvision import models


class BrakeNetwork(nn.Module):
    def __init__(self, device, input_dims, name='resnet34', checkpoint_dir='tmp/sac'):
        super(BrakeNetwork, self).__init__()

        self.device = device
        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        # import ResNet-34
        self.resnet34_backbone = models.resnet34(pretrained=True)

        # remove last layer of ResNet-34
        self.resnet34_backbone.fc = nn.Linear(512, 128, bias=True)

        self.fused_encoder = nn.Linear(3, 128, bias=True)

        self.brake_network = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, image, fused_inputs):
        out_image_features = T.relu(self.resnet34_backbone(image))
        out_fused_features = T.relu(self.fused_encoder(fused_inputs))

        concatenate_features = T.cat((out_image_features, out_fused_features), dim=1)

        brake_action = self.brake_network(concatenate_features)

        return brake_action
