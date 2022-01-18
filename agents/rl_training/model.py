import os
import sys
import itertools
from tensorboardX import SummaryWriter
import torch as T
import numpy as np
import random

seed = 0
T.manual_seed(seed)
np.random.seed(seed)
random.seed(seed) 
# for cuda
T.cuda.manual_seed_all(seed)
T.backends.cudnn.deterministic = True
T.backends.cudnn.benchmark = False

# to add the parent "agents" folder to sys path and import models
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from utils.buffer import ReplayBuffer
from networks.actor_network import ActorNetwork
from networks.critic_network import CriticNetwork
from networks.resnet_backbone import ResNetBackbone

CHECKPOINT_PATH = os.environ.get('CHECKPOINT_PATH', None)

class RLModel():
    def __init__(self, db, evaluate=False):
        self.db = db
        self.evaluate = evaluate

        if self.evaluate: #evaluate
            self.evaluation_id = self.db.get_evaluation_id()

            is_cpu = db.get_evaluation_is_cpu(self.evaluation_id)
            state_size = db.get_evaluation_state_size(self.evaluation_id)
            n_actions = db.get_evaluation_n_actions(self.evaluation_id)
            self.debug = db.get_evaluation_debug(self.evaluation_id)

            load_episode_number = self.db.get_evaluation_model_episode_number(self.evaluation_id)
            self.model_name = self.db.get_evaluation_model_name(self.evaluation_id)
            checkpoint_dir = CHECKPOINT_PATH + 'models/' + self.model_name + "/"
            log_dir = CHECKPOINT_PATH + 'logs/' + self.model_name + "_model_ep_num_" + str(load_episode_number) + "_id_" + str(self.evaluation_id) + "/"
        else: #train
            self.training_id = self.db.get_training_id()

            is_cpu = db.get_is_cpu(self.training_id)
            state_size = db.get_state_size(self.training_id)
            n_actions = db.get_n_actions(self.training_id)
            self.debug = db.get_debug(self.training_id)

            self.model_name = self.db.get_model_name(self.training_id)
            checkpoint_dir = CHECKPOINT_PATH + 'models/' + self.model_name + "/"
            log_dir = CHECKPOINT_PATH + 'logs/' + self.model_name + "/"

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        print(f"models will be in {checkpoint_dir}")
        print(f"logs will be saved to {log_dir}")

        if is_cpu:
            self.device = T.device('cpu')
        else:
            self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')

        print("device: ", self.device)


        self.writer = SummaryWriter(logdir=log_dir, comment="_carla_model")

        # load pretrained ResNet
        self.resnet_backbone = ResNetBackbone(device=self.device)

        self.actor = ActorNetwork(device=self.device, state_size=state_size, n_actions=n_actions, name='actor', checkpoint_dir=checkpoint_dir)
        self.actor_target = ActorNetwork(device=self.device, state_size=state_size, n_actions=n_actions, name='actor_target', checkpoint_dir=checkpoint_dir)

        self.critic_1 = CriticNetwork(device=self.device, state_size=state_size, n_actions=n_actions, name='critic_1', checkpoint_dir=checkpoint_dir)
        self.critic_target_1 = CriticNetwork(device=self.device, state_size=state_size, n_actions=n_actions, name='critic_target_1', checkpoint_dir=checkpoint_dir)

        self.critic_2 = CriticNetwork(device=self.device, state_size=state_size, n_actions=n_actions, name='critic_2', checkpoint_dir=checkpoint_dir)
        self.critic_target_2 = CriticNetwork(device=self.device, state_size=state_size, n_actions=n_actions, name='critic_target_2', checkpoint_dir=checkpoint_dir)
    
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
            self.critic_criterion = T.nn.MSELoss()

            # define actor and critic network optimizers
            self.actor_optimizer = T.optim.Adam(self.actor.parameters(), lr=lrpolicy)
            self.critic_optimizer = T.optim.Adam(self.q_params, lr=lrvalue)

        if self.evaluate: #evaluate
            self.load_models(load_episode_number)
        else: #train
            episode_num = self.db.get_global_episode_number(self.training_id)
            if episode_num != 0 : #load previous models if it is not first episode of training
                self.load_models(episode_num)

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
    def update(self, sample_batch):
        image_features, fused_inputs, actions, rewards, next_image_features, next_fused_inputs, dones = sample_batch

        # convert sample_batch vectors to tensor
        image_features = T.tensor(image_features, dtype=T.float32).to(self.device)
        fused_inputs = T.tensor(fused_inputs, dtype=T.float32).to(self.device)
        actions = T.tensor(actions, dtype=T.float32).to(self.device)
        rewards = T.unsqueeze(T.tensor(rewards, dtype=T.float32), dim=1).to(self.device)
        next_image_features = T.tensor(next_image_features, dtype=T.float32).to(self.device)
        next_fused_inputs = T.tensor(next_fused_inputs, dtype=T.float32).to(self.device)
        dones = T.tensor(dones, dtype=T.uint8).to(self.device)

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

        return loss_pi.data.cpu().detach().numpy(), loss_q.data.cpu().detach().numpy()

    def save_models(self, episode_number):
        print(f'.... saving models episode_number {episode_number} ....')
        self.actor.save_checkpoint(episode_number)
        self.actor_target.save_checkpoint(episode_number)
        self.critic_1.save_checkpoint(episode_number)
        self.critic_target_1.save_checkpoint(episode_number)
        self.critic_2.save_checkpoint(episode_number)
        self.critic_target_2.save_checkpoint(episode_number)

    def load_models(self, episode_number):
        print(f'.... loading models episode_number {episode_number} ....')
        self.actor.load_checkpoint(episode_number)
        self.actor_target.load_checkpoint(episode_number)
        self.critic_1.load_checkpoint(episode_number)
        self.critic_target_1.load_checkpoint(episode_number)
        self.critic_2.load_checkpoint(episode_number)
        self.critic_target_2.load_checkpoint(episode_number)