import numpy as np
import torch
from collections import deque, namedtuple

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class ExperienceReplayBuffer(object):
    def __init__(self, maximum_length, cer_mode, cer_proportion, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.buffer = deque(maxlen=maximum_length)
        self.cer_mode = cer_mode
        self.cer_proportion = cer_proportion
    
    def append(self, experience):
        self.buffer.append(experience)

    def __len__(self):
        return len(self.buffer)
    
    def sample_batch(self, n):
        if n > len(self.buffer):
            raise IndexError('Tried to sample too many elements from the buffer!')

        if self.cer_mode:
            n_random = int((1-self.cer_proportion)*n)
            n_last = n - n_random
            indices_random = np.random.choice(len(self.buffer), size=n_random, replace=False)
            batch = [self.buffer[i] for i in indices_random]

            for k in range(1, n_last+1):
                batch.append(self.buffer[-k])

            # Return a tuple of list
            return zip(*batch)
        else:
            indices = np.random.choice(len(self.buffer), size=n, replace=False)

            batch = [self.buffer[i] for i in indices]

            # Return a tuple of list
            return zip(*batch)