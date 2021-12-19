import numpy as np
import random
from collections import deque


class ReplayBuffer:
    def __init__(self, buffer_size):
        self.memories = deque(maxlen=buffer_size)

    # append experience to the replay memory
    def push(self, image_features, fused_inputs, action, reward, next_image_features, next_fused_inputs, done):
        self.memories.append((image_features, fused_inputs, action, np.array([reward]), next_image_features, next_fused_inputs, done))

    # get experience sample of batch size
    def sample(self, batch_size):
        image_feature_batch, fused_input_batch, action_batch, reward_batch, next_image_feature_batch, next_fused_input_batch, done_batch = [], [], [], [], []

        # get one batch randomly from the replay memory
        batch = random.sample(self.memories, batch_size)

        for experience in batch:
            image_feature, fused_input, action, reward, next_image_feature, next_fused_input, done = experience

            image_feature_batch.append(image_feature)
            fused_input_batch.append(fused_input)
            action_batch.append(action)
            reward_batch.append(reward)
            next_image_feature_batch.append(next_image_feature)
            next_fused_input_batch.append(next_fused_input)
            done_batch.append(done)
            
        return image_feature_batch, fused_input_batch, action_batch, reward_batch, next_image_feature_batch, next_fused_input_batch, done_batch

    def __len__(self):
        return len(self.memories)
