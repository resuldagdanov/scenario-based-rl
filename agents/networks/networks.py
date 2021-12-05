import os
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torchvision import models
from torchsummary import summary
import numpy as np

class RESNET50Model(nn.Module):
    def __init__(self, device, input_dims, name='resnet50', checkpoint_dir='tmp/sac'):
        super(RESNET50Model, self).__init__()

        self.input_dims = input_dims
        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)
        
        self.device = device
        resnet50_model = models.resnet50(pretrained=True)

        #freeze weights
        for param in resnet50_model.parameters():
            param.requires_grad = False

        self.resnet50_model = resnet50_model.to(self.device)

        #summary(self.resnet50_model, (3, 30, 40), device = self.device)

    def forward(self, image):
        if type(image).__module__ == np.__name__:
            image = T.from_numpy(image)
        image = T.reshape(image, (-1, self.input_dims[0], self.input_dims[1], self.input_dims[2]))
        image = image.to(self.device, dtype=T.float)
        out = self.resnet50_model(image)
        return out

class CriticNetwork(nn.Module):
    def __init__(self, resnet50_model, device, beta, input_dims, cnn_output_size, n_actions, fc1_dims=256, fc2_dims=256,
            name='critic', checkpoint_dir='tmp/sac'):
        super(CriticNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        self.resnet50_model = resnet50_model

        self.fc1 = nn.Linear(cnn_output_size + self.n_actions, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)

        self.device = device
        self.to(self.device)

    def forward(self, state, action):
        state = self.resnet50_model(state)
        action_value = self.fc1(T.cat([state, action], dim=1))
        action_value = F.relu(action_value)
        action_value = self.fc2(action_value)
        action_value = F.relu(action_value)

        q = self.q(action_value)

        return q

    def save_checkpoint(self):
        model_weights_dict = {}
        for name in self.state_dict():
            if name.find("resnet") == -1:
                model_weights_dict[name] = self.state_dict()[name]

        T.save(model_weights_dict, self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file), strict=False)

class ValueNetwork(nn.Module):
    def __init__(self, resnet50_model, device, beta, input_dims, cnn_output_size, fc1_dims=256, fc2_dims=256,
            name='value', checkpoint_dir='tmp/sac'):
        super(ValueNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        self.resnet50_model = resnet50_model

        self.fc1 = nn.Linear(cnn_output_size, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, fc2_dims)
        self.v = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        
        self.device = device
        self.to(self.device)

    def forward(self, state):
        state = self.resnet50_model(state)
        state_value = self.fc1(state)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = F.relu(state_value)

        v = self.v(state_value)
        return v

    def save_checkpoint(self):
        model_weights_dict = {}
        for name in self.state_dict():
            if name.find("resnet") == -1:
                model_weights_dict[name] = self.state_dict()[name]

        T.save(model_weights_dict, self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file), strict=False)

class ActorNetwork(nn.Module):
    def __init__(self, resnet50_model, device, alpha, input_dims, cnn_output_size, max_action, fc1_dims=256, 
            fc2_dims=256, n_actions=2, name='actor', checkpoint_dir='tmp/sac'):
        super(ActorNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)
        self.max_action = max_action
        self.reparam_noise = 1e-6

        self.resnet50_model = resnet50_model

        self.fc1 = nn.Linear(cnn_output_size, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        self.sigma = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        
        self.device = device
        self.to(self.device)

    def forward(self, state):
        state = self.resnet50_model(state)
        prob = self.fc1(state)
        prob = F.relu(prob)
        prob = self.fc2(prob)
        prob = F.relu(prob)

        mu = self.mu(prob)
        sigma = self.sigma(prob)

        sigma = T.clamp(sigma, min=self.reparam_noise, max=1)

        return mu, sigma

    def sample_normal(self, state, reparameterize=True):
        mu, sigma = self.forward(state)
        probabilities = Normal(mu, sigma)

        if reparameterize:
            actions = probabilities.rsample()
        else:
            actions = probabilities.sample()

        tanh_result = T.tanh(actions)
        action = tanh_result*T.tensor(self.max_action).to(self.device)
        log_probs = probabilities.log_prob(actions)
        log_probs -= T.log(1-action.pow(2)+self.reparam_noise)
        log_probs = log_probs.sum(1, keepdim=True)

        return action, log_probs

    def save_checkpoint(self):
        model_weights_dict = {}
        for name in self.state_dict():
            if name.find("resnet") == -1:
                model_weights_dict[name] = self.state_dict()[name]

        T.save(model_weights_dict, self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file), strict=False)