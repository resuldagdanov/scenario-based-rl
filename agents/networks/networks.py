import os
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

class CNNModel(nn.Module):
    def __init__(self, input_dims, cnn_output_size = 10, name='cnn', chkpt_dir='tmp/sac'):
        super(CNNModel, self).__init__()

        self.input_dims = input_dims
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_sac')
        
        self.cnn_active =  True #False
        if self.cnn_active: 
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3)
            self.conv2 = nn.Conv2d(16, 16, kernel_size=3)
            self.maxpool1 = nn.MaxPool2d(4,4)

            """
            self.conv3 = nn.Conv2d(16, 16, kernel_size=3)
            self.conv4 = nn.Conv2d(16, 16, kernel_size=3)
            self.maxpool2 = nn.MaxPool2d(4,4)
            """

            self.dropout1 = nn.Dropout(0.25)
            self.linear1 = nn.Linear(864, 128) #nn.Linear(6256, 128) #[256, 128]
            self.dropout2 = nn.Dropout(0.5)
            self.linear2 = nn.Linear(128, cnn_output_size)
        else:
            self.linear = nn.Linear(input_dims[0], cnn_output_size)

    def forward(self, s):
        if self.cnn_active: #todo:change this structure
            s = T.reshape(s, (-1, self.input_dims[2], self.input_dims[0], self.input_dims[1])) #(-1, 3, 400, 300) #(-1,3,96,96)) #todo:change this according to the env
            s = F.relu(self.conv1(s))
            s = F.relu(self.conv2(s))
            s = self.maxpool1(s)

            """
            s = F.relu(self.conv3(s))
            s = F.relu(self.conv4(s))
            s = self.maxpool2(s)
            """

            s = T.flatten(s, 1)
            s = self.linear1(self.dropout1(s))
            s = F.relu(s)
            s = self.linear2(self.dropout2(s))
        else:
            s = self.linear(s)
        return s

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class CriticNetwork(nn.Module):
    def __init__(self, device, beta, input_dims, cnn_output_size, n_actions, fc1_dims=256, fc2_dims=256,
            name='critic', chkpt_dir='tmp/sac'):
        super(CriticNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

        self.cnnmodel = CNNModel(input_dims, name=name + '_cnn', cnn_output_size=cnn_output_size, chkpt_dir=chkpt_dir)  #summary(cnnmodel, (3,96,96)) #print(cnnmodel)   

        self.fc1 = nn.Linear(cnn_output_size + self.n_actions, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)

        self.device = device
        self.to(self.device)

    def forward(self, state, action):
        state = self.cnnmodel(state)
        action_value = self.fc1(T.cat([state, action], dim=1))
        action_value = F.relu(action_value)
        action_value = self.fc2(action_value)
        action_value = F.relu(action_value)

        q = self.q(action_value)

        return q

    def save_checkpoint(self):
        self.cnnmodel.save_checkpoint()
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.cnnmodel.load_checkpoint()
        self.load_state_dict(T.load(self.checkpoint_file))

class ValueNetwork(nn.Module):
    def __init__(self, device, beta, input_dims, cnn_output_size, fc1_dims=256, fc2_dims=256,
            name='value', chkpt_dir='tmp/sac'):
        super(ValueNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

        self.cnnmodel = CNNModel(input_dims, name=name + '_cnn', cnn_output_size = cnn_output_size, chkpt_dir=chkpt_dir)  #summary(cnnmodel, (3,96,96)) #print(cnnmodel)

        self.fc1 = nn.Linear(cnn_output_size, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, fc2_dims)
        self.v = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        
        self.device = device
        self.to(self.device)

    def forward(self, state):
        state = self.cnnmodel(state)
        state_value = self.fc1(state)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = F.relu(state_value)

        v = self.v(state_value)
        return v

    def save_checkpoint(self):
        self.cnnmodel.save_checkpoint()
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.cnnmodel.load_checkpoint()
        self.load_state_dict(T.load(self.checkpoint_file))

class ActorNetwork(nn.Module):
    def __init__(self, device, alpha, input_dims, cnn_output_size, max_action, fc1_dims=256, 
            fc2_dims=256, n_actions=2, name='actor', chkpt_dir='tmp/sac'):
        super(ActorNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')
        self.max_action = max_action
        self.reparam_noise = 1e-6

        self.cnnmodel = CNNModel(input_dims, name=name + '_cnn', cnn_output_size = cnn_output_size, chkpt_dir=chkpt_dir)  #summary(cnnmodel, (3,96,96)) #print(cnnmodel)

        self.fc1 = nn.Linear(cnn_output_size, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        self.sigma = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        
        self.device = device
        self.to(self.device)

    def forward(self, state):
        state = self.cnnmodel(state)
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
        #print("tanh_result ", tanh_result)
        action = tanh_result*T.tensor(self.max_action).to(self.device)
        #print("action ", action)
        #action = action + T.tensor([1., 0., 1.]).to(self.device)
        #action = action / T.tensor([2., 1., 2.]).to(self.device)
        #print("real action ", action)
        log_probs = probabilities.log_prob(actions)
        log_probs -= T.log(1-action.pow(2)+self.reparam_noise)
        log_probs = log_probs.sum(1, keepdim=True)

        return action, log_probs

    def save_checkpoint(self):
        self.cnnmodel.save_checkpoint()
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.cnnmodel.load_checkpoint()
        self.load_state_dict(T.load(self.checkpoint_file))