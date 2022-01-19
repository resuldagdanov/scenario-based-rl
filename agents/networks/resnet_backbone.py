import os
from pathlib import Path

import torch as T
import numpy as np
import random
from torchvision import models

seed = 0
T.manual_seed(seed)
np.random.seed(seed)
random.seed(seed) 
# for cuda
T.cuda.manual_seed_all(seed)
T.backends.cudnn.deterministic = True
T.backends.cudnn.benchmark = False

import torch.nn as nn

class ResNetBackbone(nn.Module):
    def __init__(self, device, checkpoint_dir, name='resnet50'):
        super(ResNetBackbone, self).__init__()
        
        self.name = name
        self.checkpoint_dir = str(Path(checkpoint_dir).parent.absolute()) # to get into the models folder, since only one resnet50 exists

        # import ResNet-50
        self.resnet50_backbone = models.resnet50(pretrained=False) # TODO: solve the problem, this doesnt work when is_cpu = False
        self.load_weights()

        # freeze weights
        for param in self.resnet50_backbone.parameters():
            param.requires_grad = False

        self.eval()
        self.to(device)

    def forward(self, image):
        out_image_features = self.resnet50_backbone(image)
        
        return out_image_features

    def load_weights(self):
        checkpoint_file = os.path.join(self.checkpoint_dir, self.name)
        self.resnet50_backbone.load_state_dict(T.load(checkpoint_file))

        """
        for name, param in self.resnet50_backbone.named_parameters():
            print(f"load weights for resnet50 {name} {param}")
        """