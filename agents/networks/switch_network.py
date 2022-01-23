import torch
import torch.nn as nn
import torchvision


class SwitchNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        # front RGB part import ResNet-50
        self.front_rgb_backbone = torchvision.models.resnet50(pretrained=True)

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
        self.brake_network = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # switch classifier between IL and RL branch
        self.switch_network = nn.Sequential(
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

        brake = self.brake_network(mlp_out)
        switch = self.switch_network(mlp_out)

        return brake, switch