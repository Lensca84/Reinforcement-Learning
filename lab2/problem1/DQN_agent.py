# Copyright [2020] [KTH Royal Institute of Technology] Licensed under the
# Educational Community License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may
# obtain a copy of the License at http://www.osedu.org/licenses/ECL-2.0
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing
# permissions and limitations under the License.
#
# Course: EL2805 - Reinforcement Learning - Lab 2 Problem 1
# Code author: [Alessio Russo - alessior@kth.se]
# Last update: 20th October 2020, by alessior@kth.se
#

# Load packages
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from DQN_network import DqnNetwork
from DQN_network import DuelingDqnNetwork
from DQN_ERB import ExperienceReplayBuffer
import random

class Agent(object):
    ''' Base agent class, used as a parent class

        Args:
            n_actions (int): number of actions

        Attributes:
            n_actions (int): where we store the number of actions
            last_action (int): last action taken by the agent
    '''
    def __init__(self, n_actions: int):
        self.n_actions = n_actions
        self.last_action = None

    def forward(self, state: np.ndarray):
        ''' Performs a forward computation '''
        pass

    def backward(self):
        ''' Performs a backward pass on the network '''
        pass


class RandomAgent(Agent):
    ''' Agent taking actions uniformly at random, child of the class Agent'''
    def __init__(self, n_actions: int, seed):
        super(RandomAgent, self).__init__(n_actions)
        torch.manual_seed(seed)
        np.random.seed(seed)

    def forward(self, state: np.ndarray) -> int:
        ''' Compute an action uniformly at random across n_actions possible
            choices

            Returns:
                action (int): the random action
        '''
        self.last_action = np.random.randint(0, self.n_actions)
        return self.last_action


class DqnAgent(Agent):
    ''' Agent that will play with the DQN algorithm'''
    def __init__(self, n_actions, size_of_layers, buffer_size, discount_factor, batch_size, alpha, clipping_value, cer_mode, cer_proportion, dueling_mode, double_mode, seed):
        super(DqnAgent, self).__init__(n_actions)
        torch.manual_seed(seed)
        np.random.seed(seed)

        if dueling_mode:
            self.network = DuelingDqnNetwork(size_of_layers, seed)
            self.target_network = DuelingDqnNetwork(size_of_layers, seed)
        else:
            self.network = DqnNetwork(size_of_layers, seed)
            self.target_network = DqnNetwork(size_of_layers, seed)
        self.target_equal_to_main()
        self.buffer = ExperienceReplayBuffer(buffer_size, cer_mode, cer_proportion, seed)
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.optimizer = optim.Adam(self.network.parameters(), lr=alpha)
        self.clipping_value = clipping_value
        self.double_mode = double_mode
    
    def target_equal_to_main(self):
        for target_param, main_param in zip(self.target_network.parameters(), self.network.parameters()):
            target_param.data.copy_(main_param)
        return
    
    def forward(self, state, epsilon):
        ''' Return the best action w.r.t. epsilon policy and the network '''
        if random.random() < epsilon:
            self.last_action = np.random.randint(0, self.n_actions)
            return self.last_action
        else:
            state_tensor = torch.tensor([state], requires_grad=False, dtype=torch.float32)
            self.last_action = self.network(state_tensor).max(1)[1].item()
            return self.last_action
    
    def backward(self):
        # Sample a random batch of experiences
        states, actions, rewards, next_states, dones = self.buffer.sample_batch(self.batch_size)

        states_tensor = torch.tensor(states, requires_grad=False, dtype=torch.float32)
        actions_tensor = torch.tensor(actions, requires_grad=False, dtype=torch.int64).unsqueeze(1)
        rewards_tensor = torch.tensor(rewards, requires_grad=False, dtype=torch.float32).unsqueeze(1)
        next_states_tensor = torch.tensor(next_states, requires_grad=False, dtype=torch.float32)
        dones_tensor = torch.tensor(dones, requires_grad=False, dtype=torch.float32).unsqueeze(1)

        if self.double_mode:
            # Get the best actions for the network on the next states
            actions_main_network = self.network(next_states_tensor).detach().max(1)[1].unsqueeze(1)
            # Get the best values for the target network with the actions of the main network
            target_network_values = self.target_network(next_states_tensor).detach().gather(1, actions_main_network)
            # Compute the target values
            target_values = rewards_tensor + (self.discount_factor * target_network_values * (1-dones_tensor))
        else:
            # Get the target value for the network
            target_values = rewards_tensor + (self.discount_factor * self.forward_target(next_states_tensor).detach().max(1)[0].unsqueeze(1) * (1 - dones_tensor))



        # Get the values of the network
        values = self.network(states_tensor).gather(1, actions_tensor)

        # Compute loss function
        loss = nn.functional.mse_loss(values, target_values)

        # Compute gradient
        loss.backward()

        # Clip gradient norm to clipping value
        nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=self.clipping_value)

        # Perform backward pass (backpropagation)
        self.optimizer.step()

        return