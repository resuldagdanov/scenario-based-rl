import os
import sys
import itertools
import torch as T
import torch.nn.functional as F

# to add the parent "agents" folder to sys path and import models
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from utils.buffer import ReplayBuffer
from networks.actor_network import ActorNetwork
from networks.critic_network import CriticNetwork
from networks.value_network import ValueNetwork


class RLModel():
    def __init__(self, device, max_action):

        lrpolicy=0.0001
        lrvalue = 0.0005
        n_actions = 2
        buffer_size=200000

        self.actor = ActorNetwork(device=device, lrpolicy=lrpolicy, n_actions=n_actions, max_action=max_action, name='actor', checkpoint_dir='tmp/sac')
        self.actor_target = ActorNetwork(device=device, lrpolicy=lrpolicy, n_actions=n_actions, max_action=max_action, name='actor', checkpoint_dir='tmp/sac')

        self.critic_1 = CriticNetwork(device=device, lrvalue=lrvalue, n_actions=n_actions, max_action=max_action, name='critic_1', checkpoint_dir='tmp/sac')
        self.critic_target_1 = CriticNetwork(device=device, lrvalue=lrvalue, n_actions=n_actions, max_action=max_action, name='critic_1', checkpoint_dir='tmp/sac')

        self.critic_2 = CriticNetwork(device=device, lrvalue=lrvalue, n_actions=n_actions, max_action=max_action, name='critic_1', checkpoint_dir='tmp/sac')
        self.critic_target_2 = CriticNetwork(device=device, lrvalue=lrvalue, n_actions=n_actions, max_action=max_action, name='critic_1', checkpoint_dir='tmp/sac')
        
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
        self.actor_optimizer = self.actor.optimizer
        self.critic_optimizer = self.critic_1.optimizer

    def choose_action(self, image, fused_inputs):
        with T.no_grad():
            actions, _ = self.actor.sample_normal(image=image, fused_inputs=fused_inputs, reparameterize=False)

        return actions.cpu().detach().numpy()[0]

    # TODO: left here -> check out state representation
    # compute q-loss
    def calculate_loss_q(self, data):
        states, actions, rewards, next_states, dones = data[0], data[1], data[2], data[3], data[4]

        q_1 = self.critic_1(states, actions)
        q_2 = self.critic_2(states, actions)

        with T.no_grad():
            next_action, logp_next_action = self.actor(next_states)

            q_1_pi_target = self.critic_target_1(next_states, next_action)
            q_2_pi_target = self.critic_target_2(next_states, next_action)
            q_pi_target = torch.min(q_1_pi_target, q_2_pi_target)

            # apply q-function
            backup = rewards + self.gamma * (1 - dones) * (q_pi_target - self.alpha * logp_next_action)

        # get average q-loss from both critic networks
        loss_q_1 = ((q_1 - backup) ** 2).mean()
        loss_q_2 = ((q_2 - backup) ** 2).mean()
        loss_q = loss_q_1 + loss_q_2

        return loss_q

    # compute pi-loss
    def calculate_loss_pi(self, states):
        pi, logp_pi = self.actor(states)

        q_1_pi = self.critic_1(states, pi)
        q_2_pi = self.critic_2(states, pi)

        q_pi = T.min(q_1_pi, q_2_pi)
        loss_pi = (self.alpha * logp_pi - q_pi).mean()

        return loss_pi

    # update actor and critic networks
    def update(self, experience):
        states, actions, rewards, next_states, dones = experience

        # convert experience vectors to tensor
        states = T.FloatTensor(states)
        actions = T.FloatTensor(actions)
        rewards = T.FloatTensor(rewards)
        next_states = T.FloatTensor(next_states)
        dones = T.tensor(dones, dtype=T.uint8)

        # compute and backward q-loss for critic network
        self.critic_optimizer.zero_grad()
        loss_q = self.calculate_loss_q((states, actions, rewards, next_states, dones))
        loss_q.backward()
        self.critic_optimizer.step()

        # freezing q-network
        for q_param in self.q_params:
            q_param.requires_grad = False

        # compute and backward q-loss for actor network
        self.actor_optimizer.zero_grad()
        loss_pi = self.calculate_loss_pi(states)
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

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_value_params = self.target_value.named_parameters()
        value_params = self.value.named_parameters()

        target_value_state_dict = dict(target_value_params)
        value_state_dict = dict(value_params)

        for name in value_state_dict:
            value_state_dict[name] = tau*value_state_dict[name].clone() + \
                    (1-tau)*target_value_state_dict[name].clone()

        self.target_value.load_state_dict(value_state_dict, strict=False)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

        reward = T.tensor(reward, dtype=T.float).to(self.actor.device)
        done = T.tensor(done).to(self.actor.device)
        state_ = T.tensor(new_state, dtype=T.float).to(self.actor.device)
        state = T.tensor(state, dtype=T.float).to(self.actor.device)
        action = T.tensor(action, dtype=T.float).to(self.actor.device)

        value = self.value(state).view(-1)
        value_ = self.target_value(state_).view(-1)
        value_[done] = 0.0

        actions, log_probs = self.actor.sample_normal(state, reparameterize=False)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        self.value.optimizer.zero_grad()
        value_target = critic_value - log_probs
        value_loss = 0.5 * F.mse_loss(value, value_target)
        value_loss.backward(retain_graph=True)
        self.value.optimizer.step()

        self.value_losses.append(value_loss.item())

        actions, log_probs = self.actor.sample_normal(state, reparameterize=True)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)
        
        actor_loss = log_probs - critic_value
        actor_loss = T.mean(actor_loss)
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        self.actor_losses.append(actor_loss.item())

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        q_hat = self.scale*reward + self.gamma*value_
        q1_old_policy = self.critic_1.forward(state, action).view(-1)
        q2_old_policy = self.critic_2.forward(state, action).view(-1)
        critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
        critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)

        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.critic_losses.append(critic_loss.item())

        self.update_network_parameters()

    def save_models(self):
        print('.... saving models ....')
        self.actor.save_checkpoint()
        self.value.save_checkpoint()
        self.target_value.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()

    def load_models(self):
        print('.... loading models ....')
        self.actor.load_checkpoint()
        self.value.load_checkpoint()
        self.target_value.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()