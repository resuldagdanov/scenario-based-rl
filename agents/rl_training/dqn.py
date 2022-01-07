import os
import sys
import torch as T
import torch.nn as nn
import random
from tensorboardX import SummaryWriter

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

        self.epsilon = 1.0 #1.0 #TODO: get this from db
        self.epsilon_decay = 0.99 #TODO: get this from db
        self.epsilon_min = 0.02 #TODO: get this from db

        seed = 0 #TODO: get this from DB
        random.seed(seed)

        # load pretrained ResNet
        self.resnet_backbone = ResNetBackbone(device=self.device)

        self.dqn_network = DQNNetwork(device=self.device, state_size=self.state_size, n_actions=self.n_actions, name='dqn', checkpoint_dir=checkpoint_dir)
        
        if not self.evaluate:
            self.alpha = db.get_alpha(self.training_id)
            self.gamma = db.get_gamma(self.training_id)
            self.tau = db.get_tau(self.training_id)

            self.batch_size = db.get_batch_size(self.training_id)

            buffer_size = db.get_buffer_size(self.training_id)
            random_seed = db.get_random_seed(self.training_id)
            learning_rate = db.get_lrvalue(self.training_id)

            self.memory = ReplayBuffer(self.db, buffer_size=buffer_size, seed=random_seed)

            self.optimizer = T.optim.Adam(self.dqn_network.parameters(), lr=learning_rate)
            self.l1 = nn.SmoothL1Loss().to(self.device) #Huber Loss

        if self.evaluate: #evaluate
            self.load_models(load_episode_number)
        else: #train
            episode_num = self.db.get_global_episode_number(self.training_id)
            if episode_num != 0 : #load previous models if it is not first episode of training
                self.load_models(episode_num)

    def select_action(self, image_features, fused_input):
        if random.random() < self.epsilon: # random action
            action = random.choice(range(0, self.n_actions, 1))
            return action
        else:
            action = self.select_max_action(image_features, fused_input)[0] # greedy action
            return action

    def select_max_action(self, image_features, fused_input):
        return T.argmax(self.dqn_network(image_features, fused_input)).unsqueeze(0).unsqueeze(0).cpu().detach().numpy()[0]  #TODO: check the correctness

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
        dones_mask = T.tensor(dones, dtype=T.bool).to(self.device) #TODO: print and see this

        # compute and backward loss for dqn network
        self.optimizer.zero_grad()

        # Q-Learning target is Q*(S, A) <- r + γ max_a Q(S', a) 
        max_next_state_action_value = self.dqn_network(next_image_features, next_fused_inputs).max(1).values.unsqueeze(1)
        max_next_state_action_value[dones_mask] = 0.0 # TODO: check the correctness
        target = rewards + self.gamma * max_next_state_action_value
        current = self.dqn_network(image_features, fused_inputs).gather(1, actions) #TODO: check the correctness

        loss = self.l1(current, target)
        loss.backward() # compute gradients
        self.optimizer.step() # backpropagate errors

        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon, self.epsilon_min)

        return loss.data.cpu().detach().numpy()

    def save_models(self, episode_number):
        print(f'.... saving models episode_number {episode_number} ....')
        self.dqn_network.save_checkpoint(episode_number)

    def load_models(self, episode_number):
        print(f'.... loading models episode_number {episode_number} ....')
        self.dqn_network.load_checkpoint(episode_number)