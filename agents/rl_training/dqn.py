import os
import sys
import torch as T
T.manual_seed(0)
T.backends.cudnn.benchmark = False
T.use_deterministic_algorithms(True)
import torch.nn as nn
import random
from tensorboardX import SummaryWriter
import copy

# to add the parent "agents" folder to sys path and import models
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from utils.buffer import ReplayBuffer
from networks.dqn_network import DQNNetwork
from networks.resnet_backbone import ResNetBackbone

CHECKPOINT_PATH = os.environ.get('CHECKPOINT_PATH', None)

class DQNModel():
    def __init__(self, db, evaluate=False):
        self.db = db
        self.evaluate = evaluate

        if self.evaluate: #evaluate
            self.evaluation_id = self.db.get_evaluation_id()

            is_cpu = db.get_evaluation_is_cpu(self.evaluation_id)
            self.state_size = db.get_evaluation_state_size(self.evaluation_id)
            self.n_actions = db.get_evaluation_n_actions(self.evaluation_id)
            self.debug = db.get_evaluation_debug(self.evaluation_id)

            load_episode_number = self.db.get_evaluation_model_episode_number(self.evaluation_id)
            self.model_name = self.db.get_evaluation_model_name(self.evaluation_id)
            checkpoint_dir = CHECKPOINT_PATH + 'models/' + self.model_name + "/"
            log_dir = CHECKPOINT_PATH + 'logs/' + self.model_name + "_model_ep_num_" + str(load_episode_number) + "_id_" + str(self.evaluation_id) + "/"

        else: #train
            self.training_id = self.db.get_training_id()

            is_cpu = db.get_is_cpu(self.training_id)
            self.state_size = db.get_state_size(self.training_id)
            self.n_actions = db.get_n_actions(self.training_id)
            self.debug = db.get_debug(self.training_id)

            self.model_name = self.db.get_model_name(self.training_id)
            checkpoint_dir = CHECKPOINT_PATH + 'models/' + self.model_name + "/"
            log_dir = CHECKPOINT_PATH + 'logs/' + self.model_name + "/"

            self.epsilon_max = self.db.get_epsilon_max(self.training_id) # 1.0
            self.epsilon_decay = self.db.get_epsilon_decay(self.training_id) # 0.99
            self.epsilon_min = self.db.get_epsilon_min(self.training_id) # 0.02

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
        self.resnet_backbone = ResNetBackbone(device=self.device) # TODO: save weights to local and load

        self.dqn_network = DQNNetwork(device=self.device, state_size=self.state_size, n_actions=self.n_actions, name='dqn', checkpoint_dir=checkpoint_dir)
        for param in self.dqn_network.parameters():
            print(f"dqn_init dqn_network {param.data}")

        self.target_dqn_network = DQNNetwork(device=self.device, state_size=self.state_size, n_actions=self.n_actions, name='dqn_target', checkpoint_dir=checkpoint_dir)
        self.target_dqn_network.load_state_dict(copy.deepcopy(self.dqn_network.state_dict()))
        for param in self.target_dqn_network.parameters():
            print(f"dqn_init target_dqn_network {param.data}")

        if not self.evaluate:
            self.alpha = db.get_alpha(self.training_id)
            self.gamma = db.get_gamma(self.training_id)
            self.tau = db.get_tau(self.training_id)

            self.batch_size = db.get_batch_size(self.training_id)

            buffer_size = db.get_buffer_size(self.training_id)
            self.random_seed = db.get_random_seed(self.training_id)
            learning_rate = db.get_lrvalue(self.training_id)

            self.memory = ReplayBuffer(self.db, buffer_size=buffer_size, seed=self.random_seed)
            random.seed(self.random_seed)

            self.optimizer = T.optim.Adam(self.dqn_network.parameters(), lr=learning_rate)
            self.l1 = nn.SmoothL1Loss().to(self.device) #Huber Loss

        if self.evaluate: #evaluate
            self.load_models(load_episode_number)
        else: #train
            episode_num = self.db.get_global_episode_number(self.training_id)
            if episode_num != 0 : #load previous models if it is not first episode of training
                self.load_models(episode_num)


    def select_action(self, image_features, fused_input, epsilon):
        if random.random() < epsilon: # random action
            action = random.choice(range(0, self.n_actions, 1))
            return action
        else:
            action = self.select_max_action(image_features, fused_input) # greedy action
            return action

    def select_max_action(self, image_features, fused_input):
        max_action = T.argmax(self.dqn_network(image_features, fused_input)).unsqueeze(0).unsqueeze(0).cpu().detach().numpy()[0]
        return max_action[0]

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
        self.optimizer.step() # backpropagate errors

        return loss.data.cpu().detach().numpy()

    def target_update(self):
        self.target_dqn_network.load_state_dict(copy.deepcopy(self.dqn_network.state_dict()))
        for param in self.target_dqn_network.parameters():
            print(f"target_update target_dqn_network {param.data}")

    def save_models(self, episode_number):
        print(f'.... saving models episode_number {episode_number} ....')
        self.dqn_network.save_checkpoint(episode_number)
        for param in self.dqn_network.parameters():
            print(f"save_models dqn_network {param.data}")

        self.target_dqn_network.save_checkpoint(episode_number)
        for param in self.target_dqn_network.parameters():
            print(f"save_models target_dqn_network {param.data}")

    def load_models(self, episode_number):
        print(f'.... loading models episode_number {episode_number} ....')
        self.dqn_network.load_checkpoint(episode_number)
        for param in self.dqn_network.parameters():
            print(f"load_models dqn_network {param.data}")

        self.target_dqn_network.load_checkpoint(episode_number)
        for param in self.target_dqn_network.parameters():
            print(f"load_models target_dqn_network {param.data}")