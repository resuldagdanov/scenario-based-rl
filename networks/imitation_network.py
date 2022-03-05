import os
import torch
import torch.nn as nn
import torchvision
import numpy as np


class ImitationNetwork(nn.Module):
    def __init__(self, device):
        super().__init__()

        self.device = device
        
        # front RGB part import ResNet-50
        # NOTE: pretrained=True -> download new version of resnet network
        self.front_rgb_backbone = torchvision.models.resnet50(pretrained=False)

        # NOTE: comment out the following two lines if resnet model is not pre-save and "pretrained=True" in previous line
        resnet_model_path = os.path.join(os.path.join(os.environ.get("BASE_CODE_PATH"), "checkpoint/models/"), "resnet50.zip")
        self.front_rgb_backbone.load_state_dict(torch.load(resnet_model_path))

        # remove last layer of front RGB of ResNet-50
        self.front_rgb_backbone.fc = nn.Sequential(
            nn.Linear(2048, 512, bias=True),
            nn.ReLU()
        )

        # encoder for fused inputs
        self.waypoint_encoder = nn.Sequential(
            nn.Linear(2, 64, bias=True),
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
            nn.Linear(704, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # brake classifier branch
        self.brake_classifier = nn.Sequential(
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

    def forward(self, front_images, waypoint_input, speed_sequence):
        # pre-trained ResNet backbone
        front_rgb_features = self.front_rgb_backbone(front_images)
        
        # fuse relative far waypoints
        waypoint_features = self.waypoint_encoder(waypoint_input.float())

        # ego speed list to lstm network
        # input shape : (sequence length, batch size, input size)
        speed_features, (h_n, c_n) = self.speed_lstm(speed_sequence)
        speed_features = speed_features[-1, :, :]

        # concatenate rgb and fused features
        mixed_features = torch.cat((front_rgb_features, waypoint_features, speed_features), dim=1)

        # pass joined features through MLP
        mlp_out = self.mlps(mixed_features)

        brake = self.brake_classifier(mlp_out)
        return brake
    
    def inference(self, front_images, waypoint_input, speed_sequence):
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

        # fused waypoint inputs to torch
        waypoint_input = np.array(waypoint_input, np.float32)
        waypoint_input_torch = torch.from_numpy(waypoint_input.copy()).unsqueeze(0).to(self.device)

        # convert list of sequenced speed data to torch
        speed_sequence = np.array(speed_sequence, np.float32)
        speed_sequence_torch = torch.from_numpy(speed_sequence.copy()).unsqueeze(0).to(self.device)
        speed_sequence_torch = speed_sequence_torch.view(len(speed_sequence_torch), 1, -1).to(self.device)

        # inference
        with torch.no_grad():
            brake_torch = self.forward(front_images=front_images_torch, waypoint_input=waypoint_input_torch, speed_sequence=speed_sequence_torch)
        
        # torch control and switch decision to CPU numpy array
        brake = brake_torch.squeeze(0).cpu().detach().numpy()
        return int(brake)