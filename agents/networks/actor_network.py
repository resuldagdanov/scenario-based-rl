import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torchvision import models


class ActorNetwork(nn.Module):
    def __init__(self, device, lrpolicy, n_actions, max_action, name, checkpoint_dir):
        super(ActorNetwork, self).__init__()

        self.device = device
        self.n_actions = n_actions
        self.max_action = max_action
        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        # import ResNet-34
        self.resnet34_backbone = models.resnet34(pretrained=True)

        # remove last layer of ResNet-34
        self.resnet34_backbone.fc = nn.Linear(512, 128, bias=True)

        # fusion data layer
        self.fused_encoder = nn.Linear(3, 128, bias=True)

        # mlp layers
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 64)

        self.mu_layer = nn.Linear(64, self.n_actions)
        self.std_layer = nn.Linear(64, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lrpolicy)
        self.to(self.device)

    def forward(self, image, fused_inputs):
        out_image_features = T.relu(self.resnet34_backbone(image))
        out_fused_features = T.relu(self.fused_encoder(fused_inputs))

        concatenate_features = T.cat((out_image_features, out_fused_features), dim=1)

        net_out = T.relu(self.fc1(concatenate_features))
        net_out = T.relu(self.fc2(net_out))

        mu = self.mu_layer(net_out)
        log_sigma = self.std_layer(net_out)

        log_sigma = T.clamp(log_sigma, min=-20, max=2)
        std = T.exp(log_sigma)

        return mu, std

    def sample_normal(self, image, fused_inputs, deterministic=False, with_logprob=True):
        mu, std = self.forward(image, fused_inputs)

        # pre-squash distribution and sample
        pi_distribution = Normal(mu, std)

        if deterministic:
            # only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2 * (np.log(2) - pi_action - nn.functional.softplus(-2 * pi_action))).sum(axis=1)
        else:
            logp_pi = None

        # only %60 of the steering command will be used
        steer = 0.6 * pi_action[:, 0].reshape(-1, 1)

        # acceleration is from 0 ti 1;  braking is from 0 to -1
        accel_brake = pi_action[:, 1].reshape(-1, 1)

        # apply tangent hyperbolic activation functions to actions
        steer = T.tanh(steer)
        accel_brake = T.tanh(accel_brake)

        pi_action= T.cat((steer, accel_brake), 1)

        return pi_action, logp_pi

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))
