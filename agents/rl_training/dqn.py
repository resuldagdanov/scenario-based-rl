import os
import sys
import random

import torch as T
import torch.nn as nn
from torchvision import models
from pathlib import Path

from tensorboardX import SummaryWriter

# to add the parent "agents" folder to sys path and import models
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from agent_utils.buffer import ReplayBuffer
from networks.dqn_network import DQNNetwork

CHECKPOINT_PATH = os.environ.get('CHECKPOINT_PATH', None)


class DQNModel():
    def __init__(self, db, evaluate=False):
        self.db = db
        self.evaluate = evaluate

        if self.evaluate: # evaluate
            self.evaluation_id = self.db.get_evaluation_id()

            is_cpu = db.get_evaluation_is_cpu(self.evaluation_id)
            self.state_size = db.get_evaluation_state_size(self.evaluation_id)
            self.n_actions = db.get_evaluation_n_actions(self.evaluation_id)
            self.debug = db.get_evaluation_debug(self.evaluation_id)

            load_episode_number = self.db.get_evaluation_model_episode_number(self.evaluation_id)
            self.model_name = self.db.get_evaluation_model_name(self.evaluation_id)
            self.checkpoint_dir = CHECKPOINT_PATH + 'models/' + self.model_name + "/"
            log_dir = CHECKPOINT_PATH + 'logs/' + self.model_name + "_model_ep_num_" + str(load_episode_number) + "_id_" + str(self.evaluation_id) + "/"

        else: # train
            self.training_id = self.db.get_training_id()

            is_cpu = db.get_is_cpu(self.training_id)
            self.state_size = db.get_state_size(self.training_id)
            self.n_actions = db.get_n_actions(self.training_id)
            self.debug = db.get_debug(self.training_id)

            self.model_name = self.db.get_model_name(self.training_id)
            self.checkpoint_dir = CHECKPOINT_PATH + 'models/' + self.model_name + "/"
            log_dir = CHECKPOINT_PATH + 'logs/' + self.model_name + "/"

            self.epsilon_max = self.db.get_epsilon_max(self.training_id) # 1.0
            self.epsilon_decay = self.db.get_epsilon_decay(self.training_id) # 0.99
            self.epsilon_min = self.db.get_epsilon_min(self.training_id) # 0.02

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        print(f"models will be in {self.checkpoint_dir}")
        print(f"logs will be saved to {log_dir}")

        self.checkpoint_dir_resnet = str(Path(self.checkpoint_dir).parent.absolute())

        if is_cpu:
            self.device = T.device('cpu')
        else:
            self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        print("device: ", self.device)

        self.writer = SummaryWriter(logdir=log_dir, comment="_carla_model")

        # load pretrained ResNet
        self.resnet50 = models.resnet50(pretrained=False)
        resnet_model_path = os.path.join(os.path.join(os.environ.get('BASE_CODE_PATH'), "checkpoint/models/"), "resnet50.zip")
        self.resnet50.load_state_dict(T.load(resnet_model_path))

        # freeze weights
        for param in self.resnet50.parameters():
            param.requires_grad = False

        self.resnet50.eval()
        self.resnet50.to(self.device)

        self.dqn_network = DQNNetwork(state_size=self.state_size, n_actions=self.n_actions, device=self.device)
        self.target_dqn_network = DQNNetwork(state_size=self.state_size, n_actions=self.n_actions, device=self.device)

        self.dqn_network.to(self.device)
        self.target_dqn_network.to(self.device)

        if evaluate:
            self.dqn_network.eval()
            self.target_dqn_network.eval()

        if not self.evaluate:
            self.alpha = db.get_alpha(self.training_id)
            self.gamma = db.get_gamma(self.training_id)
            self.tau = db.get_tau(self.training_id)

            self.batch_size = db.get_batch_size(self.training_id)

            buffer_size = db.get_buffer_size(self.training_id)
            self.random_seed = db.get_random_seed(self.training_id)
            learning_rate = db.get_lrvalue(self.training_id)

            self.memory = ReplayBuffer(self.db, buffer_size=buffer_size, seed=self.random_seed)

            self.optimizer = T.optim.Adam(self.dqn_network.parameters(), lr=learning_rate)
            self.l1 = nn.SmoothL1Loss().to(self.device) #Huber Loss

        if self.evaluate: # evaluate
            self.load_models(load_episode_number)
        else: # train
            episode_num = self.db.get_global_episode_number(self.training_id)
            if episode_num != 0 : # load previous models if it is not first episode of training
                self.load_models(episode_num)

    def select_action(self, image_features, epsilon):
        if random.random() < epsilon: # random action
            action = random.choice(range(0, self.n_actions, 1))
            return action
        else:
            action = self.select_max_action(image_features) # greedy action
            return action

    def select_max_action(self, image_features):
        with T.no_grad():
            net_out = self.dqn_network(image_features)
            max_action = T.argmax(net_out).cpu().detach().numpy()
        return max_action

    def update(self, sample_batch):
        image_features, fused_inputs, actions, rewards, next_image_features, next_fused_inputs, dones = sample_batch

        # convert sample_batch vectors to tensor
        image_features = T.tensor(image_features, dtype=T.float32).to(self.device)
        fused_inputs = T.tensor(fused_inputs, dtype=T.float32).to(self.device)
        actions = T.tensor(actions, dtype=T.int64).to(self.device) # this is discrete for DQN
        rewards = T.unsqueeze(T.tensor(rewards, dtype=T.float32), dim=1).to(self.device)
        next_image_features = T.tensor(next_image_features, dtype=T.float32).to(self.device)
        next_fused_inputs = T.tensor(next_fused_inputs, dtype=T.float32).to(self.device)
        dones = T.tensor(dones, dtype=T.uint8).to(self.device)        

        dones_mask = (dones == 1) # True for terminal states, False for others

        max_next_state_action_value = self.target_dqn_network(next_image_features, next_fused_inputs).max(1).values.unsqueeze(1)
        max_next_state_action_value[dones_mask] = 0.0
        
        target = rewards + self.gamma * max_next_state_action_value
        current = self.dqn_network(image_features, fused_inputs).gather(1, actions)

        # compute and backward loss for dqn network
        loss = self.l1(current, target)
        self.optimizer.zero_grad()
        loss.backward() # compute gradients
        for param in self.dqn_network.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step() # backpropagate errors

        return loss.data.cpu().detach().numpy()

    def target_update(self):
        self.target_dqn_network.load_state_dict(self.dqn_network.state_dict())

    def save_models(self, episode_number):
        print(f'.... saving models episode_number {episode_number} ....')
        checkpoint_file = os.path.join(self.checkpoint_dir, "dqn" + "-ep_" + str(episode_number))
        T.save(self.dqn_network.state_dict(), checkpoint_file)

        checkpoint_file = os.path.join(self.checkpoint_dir, "dqn_target" + "-ep_" + str(episode_number))
        T.save(self.target_dqn_network.state_dict(), checkpoint_file)

    def load_models(self, episode_number):
        print(f'.... loading models episode_number {episode_number} ....')

        checkpoint_file = os.path.join(self.checkpoint_dir,  "dqn" + "-ep_" + str(episode_number))
        self.dqn_network.load_state_dict(T.load(checkpoint_file))

        checkpoint_file = os.path.join(self.checkpoint_dir,  "dqn_target" + "-ep_" + str(episode_number))
        self.target_dqn_network.load_state_dict(T.load(checkpoint_file))

    def load_resnet_weights(self):
        print(f'.... loading resnet50 model ....')
        checkpoint_file = os.path.join(self.checkpoint_dir_resnet, "resnet50")
        self.resnet50.load_state_dict(T.load(checkpoint_file))