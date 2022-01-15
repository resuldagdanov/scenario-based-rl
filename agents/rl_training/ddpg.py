
import os
import sys
import torch
torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
from copy import deepcopy

from tensorboardX import SummaryWriter

# to add the parent "agents" folder to sys path and import models
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from utils.buffer import ReplayBuffer
from networks.value_network import ValueNetwork
from networks.offset_network import OffsetNetwork


CHECKPOINT_PATH = os.environ.get('CHECKPOINT_PATH', None)


class DDPG():
    def __init__(self, db, evaluate=True):
        self.db = db
        self.evaluate = evaluate

        # TODO: transfer this to global hyperparameters dictionary
        self.polyak = 0.995 

        # TODO: IMPORTANT! load specific models correctly without defining as None
        trained_qvalue_path = None

        model_name = "offset_model_epoch_45.pth"
        trained_policy_path = os.path.join(os.path.join(os.environ.get('BASE_CODE_PATH'), "checkpoint/models/"), model_name)

        if self.evaluate: # evaluate
            is_value_pretrained = False #True # TODO: Understand this and its usage

            self.evaluation_id = self.db.get_evaluation_id()

            is_cpu = db.get_evaluation_is_cpu(self.evaluation_id)
            state_size = db.get_evaluation_state_size(self.evaluation_id)
            n_actions = db.get_evaluation_n_actions(self.evaluation_id)
            self.debug = db.get_evaluation_debug(self.evaluation_id)

            load_episode_number = self.db.get_evaluation_model_episode_number(self.evaluation_id)
            self.model_name = self.db.get_evaluation_model_name(self.evaluation_id)
            
            self.checkpoint_dir = CHECKPOINT_PATH + 'models/' + self.model_name + "/"
            log_dir = CHECKPOINT_PATH + 'logs/' + self.model_name + "_model_ep_num_" + str(load_episode_number) + "_id_" + str(self.evaluation_id) + "/"
        
        else: # train # TODO: This is unneccessary ?
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
        self.policy = OffsetNetwork()
        self.policy.to(self.device)

        self.policy.load_state_dict(torch.load(trained_policy_path))
        self.policy.eval()

        # freeze weights until brake network and offset network including mlp network
        for param in self.policy.front_rgb_backbone.parameters():
            param.requires_grad = False
        for param in self.policy.waypoint_fuser.parameters():
            param.requires_grad = False
        for param in self.policy.mlp_encoder_network.parameters():
            param.requires_grad = False

        # create value calculator networks
        self.qvalue = ValueNetwork(device=self.device)

        if is_value_pretrained is True:
            self.qvalue.load_state_dict(torch.load(trained_qvalue_path))

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

            # create replay buffer memory
            self.memory = ReplayBuffer(self.db, buffer_size=buffer_size, seed=random_seed)

            # create target networks
            self.policy_target = deepcopy(self.policy)
            self.qvalue_target = deepcopy(self.qvalue)

            # freeze target networks with respect to optimizers (only update via polyak averaging)
            for p in self.policy_target.parameters():
                p.requires_grad = False
            for p in self.qvalue_target.parameters():
                p.requires_grad = False

            # define actor and critic network optimizers
            self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lrpolicy)
            self.qvalue_optimizer = torch.optim.Adam(self.qvalue.parameters(), lr=lrvalue)

    def get_state_space(self, front_image, fused_input):
        with torch.no_grad():
            state_space = self.policy(fronts=front_image, fused_input=fused_input)

        return state_space.cpu().detach().numpy()[0]

    def select_action(self, network, state_space):
        with torch.no_grad():
            actions = network.compute_action(state_space=state_space)

        # brake and offset amount
        actions_np = tuple(act.cpu().detach().numpy()[0] for act in actions)
        return actions_np

    # set up function for computing DDPG Q-loss
    def calculate_loss_q(self, state_space, actions, rewards, next_state_space, dones):
        q = self.qvalue(state_space=state_space, action=actions)

        # bellman backup for Q function
        with torch.no_grad():
            next_action = self.select_action(network=self.policy_target, state_space=next_state_space)
            q_pi_target = self.qvalue_target(state_space=next_state_space, action=next_action)

            # apply q-function
            backup = rewards + self.gamma * (1 - dones) * q_pi_target

        # mse loss against bellman backup
        loss_q = ((q - backup)**2).mean()

        return loss_q

    # compute pi-loss
    def calculate_loss_pi(self, state_space):
        pi_action = self.select_action(network=self.policy, state_space=state_space)
        loss_pi = -pi_action.mean()
        
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

        # run one gradient descent step for Q
        self.qvalue_optimizer.zero_grad()
        loss_q = self.calculate_loss_q(state_space=state, actions=actions, rewards=rewards, next_state_space=next_state, dones=dones)

        # compute and backward q-loss for critic network
        loss_q.backward()
        self.qvalue_optimizer.step()

        # freeze Q-network so you don't waste computational effort 
        # computing gradients for it during the policy learning step
        for q_param in self.qvalue.parameters():
            q_param.requires_grad = False

        # run one gradient descent step for pi
        self.policy_optimizer.zero_grad()
        loss_pi = self.calculate_loss_pi(state_space=state)

        # compute and backward pi-loss for actor network
        loss_pi.backward()
        self.policy_optimizer.step()

        # unfreeze Q-network so you can optimize it at next DDPG step
        for q_param in self.qvalue.parameters():
            q_param.requires_grad = True
            
        # update target networks by polyak averaging
        with torch.no_grad():
            # NB: we use an in-place operations "mul_", "add_" to update target
            # params, as opposed to "mul" and "add", which would make new tensors
            for target_param, param in zip(self.policy_target.parameters(), self.policy.parameters()):
                target_param.data.mul_(self.polyak)
                target_param.data.add_((1 - self.polyak) * param.data)

            for target_param, param in zip(self.qvalue_target.parameters(), self.qvalue.parameters()):
                target_param.data.mul_(self.polyak)
                target_param.data.add_((1 - self.polyak) * param.data)

        return loss_pi.data.cpu().detach().numpy(), loss_q.data.cpu().detach().numpy()

    def save_models(self, episode_number):
        torch.save(self.policy.state_dict(), os.path.join(self.checkpoint_dir, "policy" + "-ep_" + str(episode_number)))
        torch.save(self.qvalue.state_dict(), os.path.join(self.checkpoint_dir, "qvalue" + "-ep_" + str(episode_number)))
