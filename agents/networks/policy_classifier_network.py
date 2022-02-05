import torch
import torch.nn as nn
import torchvision
import numpy as np


class PolicyClassifierNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # front RGB part import ResNet-50
        self.front_rgb_backbone = torchvision.models.resnet50(pretrained=True)

        # remove last layer of front RGB of ResNet-50
        self.front_rgb_backbone.fc = nn.Sequential(
            nn.Linear(2048, 512, bias=True),
            nn.ReLU()
        )

        # use recurrent structure with speed sequencial input
        self.speed_lstm = nn.LSTM(
            input_size=120,
            hidden_size=128,
            num_layers=1,
            batch_first=False
        )

        # concatenate fused inputs
        self.mlps = nn.Sequential(
            nn.Linear(640, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # policy classifier branch
        self.policy_classifier = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 6),
        )

    def normalize_rgb(self, x):
        x = x.clone()
        x[:, 0] = (x[:, 0] - 0.485) / 0.229
        x[:, 1] = (x[:, 1] - 0.456) / 0.224
        x[:, 2] = (x[:, 2] - 0.406) / 0.225
        return x

    def forward(self, front_images, speed_sequence):

        # pre-trained ResNet backbone
        front_rgb_features = self.front_rgb_backbone(front_images)

        # ego speed list to lstm network
        # input shape : (sequence length, batch size, input size)
        speed_features, (h_n, c_n) = self.speed_lstm(speed_sequence)
        speed_features = speed_features[-1, :, :]

        # concatenate rgb and fused features
        mixed_features = torch.cat((front_rgb_features, speed_features), dim=1)

        # pass joined features through MLP
        mlp_out = self.mlps(mixed_features)

        label = self.policy_classifier(mlp_out)
        return label
    
    def inference(self, front_images, speed_sequence):
        # convert width height channel to channel width height
        front_images = np.array(front_images.transpose((2, 0, 1)), np.float32)
        
        # BGRA to BGR
        front_images = front_images[:3, :, :]
        # BGR to RGB
        front_images = front_images[::-1, :, :]
        
        # normalize to 0 - 1
        front_images = front_images / 255
        # to tensor and unsquueze for batch dimension
        front_images_torch = torch.from_numpy(front_images.copy()).unsqueeze(0)
        
        # normalize input image
        front_images_torch = self.normalize_rgb(front_images_torch)
        front_images_torch = front_images_torch.to(self.device)

        # convert list of sequenced speed data to torch
        speed_sequence = np.array(speed_sequence, np.float32)
        speed_sequence_torch = torch.from_numpy(speed_sequence.copy()).unsqueeze(0).to(self.device)
        speed_sequence_torch = speed_sequence_torch.view(len(speed_sequence_torch), 1, -1).to(self.device)

        # inference
        with torch.no_grad():
            label_torch = self.forward(front_images=front_images_torch, speed_sequence=speed_sequence_torch)

        print("label_torch : ", label_torch)
        
        # torch switch decision to CPU numpy array
        label = torch.argmax(nn.functional.softmax(label_torch))
        return int(label)