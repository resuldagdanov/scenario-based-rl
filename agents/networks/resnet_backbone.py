import os
import torch as T
import torch.nn as nn
from torchvision import models


class ResNetBackbone(nn.Module):
    def __init__(self, device, name='resnet50', checkpoint_dir='tmp/sac'):
        super(ResNetBackbone, self).__init__()

        self.device = device
        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        # import ResNet-50
        self.resnet50_backbone = models.resnet34(pretrained=True)

        # freeze weights
        for param in self.resnet50_backbone.parameters():
            param.requires_grad = False

        self.resnet50_model = self.resnet50_backbone.to(self.device)

    def normalize_rgb(self, x):
        x = x.clone()
        x[:, 0] = (x[:, 0] - 0.485) / 0.229
        x[:, 1] = (x[:, 1] - 0.456) / 0.224
        x[:, 2] = (x[:, 2] - 0.406) / 0.225
        return x

    def forward(self, image):
        # to tensor and unsquueze for batch dimension
        image = T.from_numpy(image.copy()).unsqueeze(0)

        # normalize input image (using default torch normalization technique)
        image = self.normalize_rgb(image)

        image = T.reshape(image, (-1, self.input_dims[0], self.input_dims[1], self.input_dims[2]))
        image = image.to(self.device, dtype=T.float)
        
        out_image_features = self.resnet50_model(image)
        return out_image_features
