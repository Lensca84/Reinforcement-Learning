import numpy as np
import torch
from collections import deque, namedtuple

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class ExperienceReplayBuffer(object):
    def __init__(self, maximum_length, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.maximum_length = maximum_length
        self.buffer = deque(maxlen=self.maximum_length)
    
    def append(self, experience):
        self.buffer.append(experience)

    def __len__(self):
        return len(self.buffer)
    
    def clear(self):
        self.buffer.clear()
    
    def iterator(self):
        return zip(*[experience for experience in self.buffer])