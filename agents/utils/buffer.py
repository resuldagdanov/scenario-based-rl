import random #TODO: add seed for this one
from utils.db import DB

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.db = DB()
        self.buffer_size = buffer_size
        self.id = 0
        self.filled_size = 0

    # append experience to the replay memory
    def push(self, image_features, fused_inputs, action, reward, next_image_features, next_fused_inputs, done):
        if self.id == self.buffer_size:
            self.id = 0 #unique db id

        self.db.insert_data(self.id, image_features, fused_inputs, action, reward, next_image_features, next_fused_inputs, done)
        self.id += 1
        self.filled_size += 1

    # get experience sample of batch size
    def sample(self, batch_size):
        high = self.filled_size if self.filled_size < self.buffer_size else self.buffer_size

        sample_indexes = random.sample(range(0, high), batch_size) #low inclusive, high exclusive
        
        sample_batch = self.db.read_batch_data(tuple(sample_indexes), batch_size)
        
        return sample_batch

    def close(self):
        self.db.close()