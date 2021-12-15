import numpy as np
import random
from collections import deque


class ReplayBuffer:
    def __init__(self, buffer_size):
        self.memories = deque(maxlen=buffer_size)

    # append experience to the replay memory
    def push(self, state, action, reward, next_state, done):
        self.memories.append((state, action, np.array([reward]), next_state, done))

    # get experience sample of batch size
    def sample(self, batch_size):
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = [], [], [], [], []

        # get one batch randomly from the replay memory
        batch = random.sample(self.memories, batch_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)
            
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def __len__(self):
        return len(self.memories)
