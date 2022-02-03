import os
import sys
import itertools
from tensorboardX import SummaryWriter
import torch as T
import numpy as np
import random

"""
seed = 0
T.manual_seed(seed)
np.random.seed(seed)
random.seed(seed) 
# for cuda
T.cuda.manual_seed_all(seed)
T.backends.cudnn.deterministic = True
T.backends.cudnn.benchmark = False
"""

import torch.nn as nn

# to add the parent "agents" folder to sys path and import models
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from agent_utils.buffer import ReplayBuffer
from networks.value_network import ValueNetwork
from networks.offset_network import OffsetNetwork


CHECKPOINT_PATH = os.environ.get('CHECKPOINT_PATH', None)


class SAC():
    def __init__(self, db, evaluate=False):
        self.db = db
        self.evaluate = evaluate

        # TODO: IMPORTANT! load specific models correctly without defining as None
        trained_actor_path = None
        trained_critic_1_path = None
        trained_critic_2_path = None

        if self.evaluate: # evaluate
            is_critic_pretrained = True

            self.evaluation_id = self.db.get_evaluation_id()

            is_cpu = db.get_evaluation_is_cpu(self.evaluation_id)
            state_size = db.get_evaluation_state_size(self.evaluation_id)
            n_actions = db.get_evaluation_n_actions(self.evaluation_id)
            self.debug = db.get_evaluation_debug(self.evaluation_id)

            load_episode_number = self.db.get_evaluation_model_episode_number(self.evaluation_id)
            self.model_name = self.db.get_evaluation_model_name(self.evaluation_id)
            
            self.checkpoint_dir = CHECKPOINT_PATH + 'models/' + self.model_name + "/"
            log_dir = CHECKPOINT_PATH + 'logs/' + self.model_name + "_model_ep_num_" + str(load_episode_number) + "_id_" + str(self.evaluation_id) + "/"
        
        else: # train
            self.training_id = self.db.get_training_id()

            is_cpu = db.get_is_cpu(self.training_id)
            state_size = db.get_state_size(self.training_id)
            n_actions = db.get_n_actions(self.training_id)
            self.debug = db.get_debug(self.training_id)

            self.model_name = self.db.get_model_name(self.training_id)

            self.checkpoint_dir = CHECKPOINT_PATH + 'models/' + self.model_name + "/"
            log_dir = CHECKPOINT_PATH + 'logs/' + self.model_name + "/"

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        print(f"models will be in {self.checkpoint_dir}")
        print(f"logs will be saved to {log_dir}")

        if is_cpu:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print("device: ", self.device)

        self.writer = SummaryWriter(logdir=log_dir, comment="_carla_model")

        # load pretrained policy network
        self.actor = OffsetNetwork()
        self.actor_target = OffsetNetwork()

        self.actor.to(self.device)
        self.actor_target.to(self.device)

        self.actor.load_state_dict(torch.load(trained_actor_path))

        # freeze weights until brake network and offset network including mlp network
        for param in self.actor.front_rgb_backbone.parameters():
            param.requires_grad = False
        for param in self.actor.waypoint_fuser.parameters():
            param.requires_grad = False
        for param in self.actor.mlp_encoder_network.parameters():
            param.requires_grad = False

        # create value calculator networks
        self.critic_1 = ValueNetwork(device=self.device)
        self.critic_target_1 = ValueNetwork(device=self.device)
        self.critic_2 = ValueNetwork(device=self.device)
        self.critic_target_2 = ValueNetwork(device=self.device)

        if is_critic_pretrained is True:
            self.critic_1.load_state_dict(torch.load(trained_critic_1_path))
            self.critic_2.load_state_dict(torch.load(trained_critic_2_path))

        # during training
        if not self.evaluate:
            self.alpha = db.get_alpha(self.training_id)
            self.gamma = db.get_gamma(self.training_id)
            self.tau = db.get_tau(self.training_id)

            self.batch_size = db.get_batch_size(self.training_id)

            buffer_size = db.get_buffer_size(self.training_id)
            random_seed = db.get_random_seed(self.training_id)

            lrpolicy = db.get_lrpolicy(self.training_id)
            lrvalue = db.get_lrvalue(self.training_id)

            self.memory = ReplayBuffer(self.db, buffer_size=buffer_size, seed=random_seed)

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
            self.critic_criterion = torch.nn.MSELoss()

            # define actor and critic network optimizers
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lrpolicy)
            self.critic_optimizer = torch.optim.Adam(self.q_params, lr=lrvalue)

    def get_state_space(self, front_image, fused_input):
        with torch.no_grad():
            state_space = self.actor(fronts=front_image, fused_input=fused_input)

        return state_space.cpu().detach().numpy()[0]

    def select_action(self, network, state_space):
        with torch.no_grad():
            actions = network.compute_action(state_space=state_space)

        # brake and offset amount
        return actions.cpu().detach().numpy()[0]

    # compute q-loss
    def calculate_loss_q(self, state_space, actions, rewards, next_state_space, dones):
        q_1 = self.critic_1(state_space=state_space, action=actions)
        q_2 = self.critic_2(state_space=state_space, action=actions)

        with torch.no_grad():
            next_action = self.select_action(network=self.actor, state_space=state_space)

            q_1_pi_target = self.critic_target_1(state_space=next_state_space, action=next_action)
            q_2_pi_target = self.critic_target_2(state_space=next_state_space, action=next_action)
            q_pi_target = torch.min(q_1_pi_target, q_2_pi_target)

            # apply q-function
            backup = rewards + self.gamma * (1 - dones) * (q_pi_target - self.alpha * next_action)

        # get average q-loss from both critic networks
        loss_q_1 = ((q_1 - backup)**2).mean()
        loss_q_2 = ((q_2 - backup)**2).mean()
        loss_q = loss_q_1 + loss_q_2

        return loss_q

    # compute pi-loss
    def calculate_loss_pi(self, state_space):
        pi_action = self.select_action(network=self.actor, state_space=state_space)

        q_1_pi = self.critic_1(state_space=state_space, action=pi_action)
        q_2_pi = self.critic_2(state_space=state_space, action=pi_action)
        q_pi = torch.min(q_1_pi, q_2_pi)
        
        loss_pi = (self.alpha * pi_action - q_pi).mean()

        return loss_pi

    # update actor and critic networks
    def update(self, sample_batch):
        state, actions, rewards, next_state, dones = sample_batch

        # convert sample_batch vectors to tensor
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.float32).to(self.device)
        rewards = torch.unsqueeze(torch.tensor(rewards, dtype=torch.float32), dim=1).to(self.device)
        next_state = torch.tensor(next_state, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.uint8).to(self.device)

        # compute and backward q-loss for critic network
        self.critic_optimizer.zero_grad()
        loss_q = self.calculate_loss_q(state_space=state, actions=actions, rewards=rewards, next_state_space=next_state, dones=dones)
        loss_q.backward()
        self.critic_optimizer.step()

        # freezing q-network
        for q_param in self.q_params:
            q_param.requires_grad = False

        # compute and backward q-loss for actor network
        self.actor_optimizer.zero_grad()
        loss_pi = self.calculate_loss_pi(state_space=state)
        loss_pi.backward()
        self.actor_optimizer.step()

        # unfreezing q-network
        for q_param in self.q_params:
            q_param.requires_grad = True
            
        # update target networks
        with torch.no_grad():
            for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

            for target_param, param in zip(self.critic_target_1.parameters(), self.critic_1.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

            for target_param, param in zip(self.critic_target_2.parameters(), self.critic_2.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

        return loss_pi.data.cpu().detach().numpy(), loss_q.data.cpu().detach().numpy()

    def save_models(self, episode_number):
        torch.save(self.actor.state_dict(), os.path.join(self.checkpoint_dir, "actor" + "-ep_" + str(episode_number)))
        torch.save(self.critic_1.state_dict(), os.path.join(self.checkpoint_dir, "critic_1" + "-ep_" + str(episode_number)))
        torch.save(self.critic_2.state_dict(), os.path.join(self.checkpoint_dir, "critic_2" + "-ep_" + str(episode_number)))
