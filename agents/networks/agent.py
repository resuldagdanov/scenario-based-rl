import torch as T
import torch.nn.functional as F
from .buffer import ReplayBuffer
from .networks import ActorNetwork, CriticNetwork, ValueNetwork, RESNET50Model

class Agent():
    def __init__(self, device, max_action, alpha=0.0003, beta=0.0003, gamma=0.99, n_actions=2, max_size=1000000, tau=0.005, batch_size=128, reward_scale=2, checkpoint_dir="tmp/sac"):
        self.state_size = (1000,)
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, self.state_size, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions

        self.actor = ActorNetwork(device, alpha, self.state_size[0], n_actions=n_actions, name='actor', max_action=max_action, checkpoint_dir=checkpoint_dir)
        self.critic_1 = CriticNetwork(device, beta, self.state_size[0], n_actions=n_actions, name='critic_1', checkpoint_dir=checkpoint_dir)
        self.critic_2 = CriticNetwork(device, beta, self.state_size[0], n_actions=n_actions, name='critic_2', checkpoint_dir=checkpoint_dir)
        self.value = ValueNetwork(device, beta, self.state_size[0], name='value', checkpoint_dir=checkpoint_dir)
        self.target_value = ValueNetwork(device, beta, self.state_size[0], name='target_value', checkpoint_dir=checkpoint_dir)

        self.actor_losses = []
        self.critic_losses = []
        self.value_losses = []

        self.scale = reward_scale
        self.update_network_parameters(tau=1)

    def print_losses(self):
        print("actor_losses ", self.actor_losses)
        print("critic_losses ", self.critic_losses)
        print("value_losses ", self.value_losses)

    def choose_action(self, state):
        actions, _ = self.actor.sample_normal(state, reparameterize=False)

        return actions.cpu().detach().numpy()[0]

    def remember(self, state, action, reward, new_state, done):
        if done:
            print(f"remember {done}")
        self.memory.store_transition(state, action, reward, new_state, done)

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

        state, action, reward, new_state, done = \
                self.memory.sample_buffer(self.batch_size)

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