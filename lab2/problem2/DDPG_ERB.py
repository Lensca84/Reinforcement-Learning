import numpy as np
import torch
from collections import deque, namedtuple

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class ExperienceReplayBuffer(object):
    def __init__(self, maximum_length, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.buffer = deque(maxlen=maximum_length)
    
    def append(self, experience):
        self.buffer.append(experience)

    def __len__(self):
        return len(self.buffer)
    
    def sample_batch(self, n):
        if n > len(self.buffer):
            raise IndexError('Tried to sample too many elements from the buffer!')

        indices = np.random.choice(len(self.buffer), size=n, replace=False)

        batch = [self.buffer[i] for i in indices]

        # Return a tuple of list
        return zip(*batch)