import os
import torch as T
#T.manual_seed(0)
import torch.nn as nn
from torchvision import models
#T.backends.cudnn.benchmark = False

T.manual_seed(0)
#np.random.seed(0)
#random.seed(seed) 
# for cuda
T.cuda.manual_seed_all(0)
#T.backends.cudnn.deterministic = True
T.backends.cudnn.benchmark = False
T.backends.cudnn.enabled = False
#T.use_deterministic_algorithms(True)

class ResNetBackbone(nn.Module):
    def __init__(self, device, name='resnet50', checkpoint_dir='tmp/sac'):
        super(ResNetBackbone, self).__init__()
        
        self.name = name
        self.checkpoint_dir = checkpoint_dir

        # import ResNet-50
        self.resnet50_backbone = models.resnet50(pretrained=True) # TODO: solve the problem, this doesnt work when is_cpu = False

        # freeze weights
        for param in self.resnet50_backbone.parameters():
            param.requires_grad = False

        self.resnet50_backbone.eval()
        self.resnet50_backbone.to(device)

        for name, param in self.resnet50_backbone.named_parameters():
            print(f"resnetbackbone {name} {param}")

    def forward(self, image):
        out_image_features = self.resnet50_backbone(image)
        
        return out_image_features