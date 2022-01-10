import os
import torch as T
T.manual_seed(0)
import torch.nn as nn
from torchvision import models
T.backends.cudnn.benchmark = False
T.use_deterministic_algorithms(True)

class ResNetBackbone(nn.Module):
    def __init__(self, device, name='resnet50', checkpoint_dir='tmp/sac'):
        super(ResNetBackbone, self).__init__()
        
        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        # import ResNet-50
        self.resnet50_backbone = models.resnet50(pretrained=True)

        # freeze weights
        for param in self.resnet50_backbone.parameters():
            param.requires_grad = False

        self.resnet50_backbone.to(device)

    def forward(self, image):
        out_image_features = self.resnet50_backbone(image)
        
        return out_image_features
