import os
import sys
import itertools
import torch as T

# to add the parent "agents" folder to sys path and import models
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from utils.buffer import ReplayBuffer
from networks.actor_network import ActorNetwork
from networks.critic_network import CriticNetwork
from networks.resnet_backbone import ResNetBackbone


class RLModel():
    def __init__(self):

        # TODO: forward hyperparameters to global input argument
        lrpolicy=0.0001
        lrvalue = 0.0005
        n_actions = 2
        buffer_size=200000
        state_size = 1000 # output size of resnet

        #self.device = T.device('cuda' if T.cuda.is_available() else 'cpu') # TODO: open cuda
        self.device = T.device('cpu')

        self.tau = 0.001
        self.alpha = 0.2
        self.gamma = 0.97
        self.batch_size = 64

        self.debug = False

        # load pretrained ResNet
        self.resnet_backbone = ResNetBackbone(device=self.device)

        self.actor = ActorNetwork(device=self.device, state_size=state_size, n_actions=n_actions, name='actor', checkpoint_dir='tmp/sac')
        self.actor_target = ActorNetwork(device=self.device, state_size=state_size, n_actions=n_actions, name='actor', checkpoint_dir='tmp/sac')

        self.critic_1 = CriticNetwork(device=self.device, state_size=state_size, n_actions=n_actions, name='critic_1', checkpoint_dir='tmp/sac')
        self.critic_target_1 = CriticNetwork(device=self.device, state_size=state_size, n_actions=n_actions, name='critic_1', checkpoint_dir='tmp/sac')

        self.critic_2 = CriticNetwork(device=self.device, state_size=state_size, n_actions=n_actions, name='critic_1', checkpoint_dir='tmp/sac')
        self.critic_target_2 = CriticNetwork(device=self.device, state_size=state_size, n_actions=n_actions, name='critic_1', checkpoint_dir='tmp/sac')
        
        self.memory = ReplayBuffer(buffer_size=buffer_size)

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target_1.parameters(), self.critic_1.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target_2.parameters(), self.critic_2.parameters()):
            target_param.data.copy_(param.data)

        for p in self.actor_target.parameters():
            p.requires_grad = False
        for p in self.critic_target_1.parameters():
            p.requires_grad = False
        for p in self.critic_target_2.parameters():
            p.requires_grad = False
        
        self.q_params = itertools.chain(self.critic_1.parameters(), self.critic_2.parameters())

        # mse loss algoritm is applied
        self.critic_criterion = T.nn.MSELoss()

        # define actor and critic network optimizers
        self.actor_optimizer = T.optim.Adam(self.actor.parameters(), lr=lrpolicy)
        self.critic_optimizer = T.optim.Adam(self.q_params, lr=lrvalue)

    def select_action(self, image_features, fused_input, deterministic=True):
        with T.no_grad():
            actions, _ = self.actor(image_features=image_features, fused_input=fused_input, deterministic=deterministic, with_logprob=False)

        return actions.cpu().detach().numpy()[0]

    # compute q-loss
    def calculate_loss_q(self, image_features, fused_inputs, actions, rewards, next_image_features, next_fused_inputs, dones):
        q_1 = self.critic_1(image_features, fused_inputs, actions)
        q_2 = self.critic_2(image_features, fused_inputs, actions)

        with T.no_grad():
            next_action, logp_next_action = self.actor(image_features, fused_inputs)

            q_1_pi_target = self.critic_target_1(next_image_features, next_fused_inputs, next_action)
            q_2_pi_target = self.critic_target_2(next_image_features, next_fused_inputs, next_action)
            q_pi_target = T.min(q_1_pi_target, q_2_pi_target)

            # apply q-function
            backup = rewards + self.gamma * (1 - dones) * (q_pi_target - self.alpha * logp_next_action)

        # get average q-loss from both critic networks
        loss_q_1 = ((q_1 - backup)**2).mean()
        loss_q_2 = ((q_2 - backup)**2).mean()
        loss_q = loss_q_1 + loss_q_2

        return loss_q

    # compute pi-loss
    def calculate_loss_pi(self, image_features, fused_inputs):
        pi_action, logp_pi_action = self.actor(image_features, fused_inputs)

        q_1_pi = self.critic_1(image_features, fused_inputs, pi_action)
        q_2_pi = self.critic_2(image_features, fused_inputs, pi_action)
        q_pi = T.min(q_1_pi, q_2_pi)
        
        loss_pi = (self.alpha * logp_pi_action - q_pi).mean()

        return loss_pi

    # update actor and critic networks
    def update(self, experience):
        image_features, fused_inputs, actions, rewards, next_image_features, next_fused_inputs, dones = experience

        # convert experience vectors to tensor
        image_features = T.FloatTensor(image_features)
        fused_inputs = T.FloatTensor(fused_inputs)
        actions = T.FloatTensor(actions)
        rewards = T.FloatTensor(rewards)
        next_image_features = T.FloatTensor(next_image_features)
        next_fused_inputs = T.FloatTensor(next_fused_inputs)
        dones = T.tensor(dones, dtype=T.uint8)

        # compute and backward q-loss for critic network
        self.critic_optimizer.zero_grad()
        loss_q = self.calculate_loss_q(image_features, fused_inputs, actions, rewards, next_image_features, next_fused_inputs, dones)
        loss_q.backward()
        self.critic_optimizer.step()

        # freezing q-network
        for q_param in self.q_params:
            q_param.requires_grad = False

        # compute and backward q-loss for actor network
        self.actor_optimizer.zero_grad()
        loss_pi = self.calculate_loss_pi(image_features, fused_inputs)
        loss_pi.backward()
        self.actor_optimizer.step()

        # unfreezing q-network
        for q_param in self.q_params:
            q_param.requires_grad = True
            
        # update target networks
        with T.no_grad():
            for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

            for target_param, param in zip(self.critic_target_1.parameters(), self.critic_1.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

            for target_param, param in zip(self.critic_target_2.parameters(), self.critic_2.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

    def _totorch(self, container, dtype):
        if isinstance(container[0], T.Tensor):
            tensor = T.stack(container)
        else:
            tensor = T.tensor(container, dtype=dtype)
        return tensor

    def save_models(self):
        print('.... saving models ....')
        self.actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()

    def load_models(self):
        print('.... loading models ....')
        self.actor.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()