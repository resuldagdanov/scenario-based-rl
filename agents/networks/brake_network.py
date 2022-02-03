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
import torchvision


class BrakeNetwork(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')

        # for RGB part import ResNet-50
        self.front_rgb_backbone = torchvision.models.resnet50(pretrained=pretrained)

        # remove last layer of ResNet-50
        self.front_rgb_backbone.fc = nn.Linear(2048, 512, bias=True)

        # encoder for fused inputs
        self.fused_encoder = nn.Linear(3, 128, bias=True)

        # brake network
        self.brake_network = nn.Sequential(
            nn.Linear(640, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def normalize_rgb(self, x):
        x = x.clone()
        x[:, 0] = (x[:, 0] - 0.485) / 0.229
        x[:, 1] = (x[:, 1] - 0.456) / 0.224
        x[:, 2] = (x[:, 2] - 0.406) / 0.225
        return x

    def forward(self, rgb, fused_input):
        rgb_features = T.relu(self.front_rgb_backbone(rgb))
        
        fused_features = T.relu(self.fused_encoder(fused_input.float()))

        mid_features = T.cat((rgb_features, fused_features), dim=1)

        brake = self.brake_network(mid_features) 

        return brake

    # assuming cv2 BGRA image received 
    def inference(self, image, fused_inputs):
        # convert width height channel to channel width height
        image = np.array( image.transpose((2, 0, 1)), np.float32)
        
        # BGRA to BGR
        image = image[:3, :, :]
        # BGR to RGB
        image = image[::-1, :, :]
        
        # normalize to 0 - 1
        image = image / 255
        # to tensor and unsquueze for batch dimension
        image_torch = T.from_numpy(image.copy()).unsqueeze(0)
        
        # normalize input image
        image_torch = self.normalize_rgb(image_torch)
        image_torch = image_torch.to(self.device)

        # fused inputs to torch
        fused_inputs = np.array(fused_inputs, np.float32)
        fused_inputs_torch = T.from_numpy(fused_inputs.copy()).unsqueeze(0).to(self.device)

        # inference
        with T.no_grad():
            brake_torch = self.forward(image_torch, fused_inputs_torch)
        
        # torch control to CPU numpy array
        brake = brake_torch.squeeze(0).cpu().detach().numpy()

        return brake
