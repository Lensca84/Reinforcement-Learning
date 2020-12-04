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
    def __init__(self, n_actions: int):
        super(RandomAgent, self).__init__(n_actions)

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
    def __init__(self, n_actions, size_of_layers, buffer_size, discount_factor, batch_size, alpha, clipping_value):
        super(DqnAgent, self).__init__(n_actions)

        self.network = DqnNetwork(size_of_layers)
        self.target_network = DqnNetwork(size_of_layers)
        self.target_equal_to_main()
        self.buffer = ExperienceReplayBuffer(buffer_size)
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.optimizer = optim.Adam(self.network.parameters(), lr=alpha)
        self.clipping_value = clipping_value
    
    def target_equal_to_main(self):
        self.target_network.load_state_dict(self.network.state_dict())
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
    
    def forward_target(self, state):
        ''' Return the best value of the target network '''
        state_tensor = torch.tensor([state], requires_grad=False, dtype=torch.float32)
        return self.target_network(state_tensor).max(1)[0].item()

    def backward(self):
        # Sample a random batch of experiences
        states, actions, rewards, next_states, dones = self.buffer.sample_batch(self.batch_size)

        # Get the values of the network
        states_tensor = torch.tensor(states, requires_grad=False, dtype=torch.float32)
        values = self.network(states_tensor)
        # Compute the target values
        #target_values = np.zeros((self.batch_size, self.n_actions))
        target_values = values.clone()
        #target_values = np.zeros((self.batch_size, 1))
        for i in range(self.batch_size):
            if dones[i]:
                target_values[i][actions[i]] = rewards[i]
                #target_values[i][0] = rewards[i]
            else:
                target_values[i][actions[i]] = rewards[i] + self.discount_factor*self.forward_target(next_states[i])
                #target_values[i][0] = rewards[i] + self.discount_factor*self.forward_target(next_states[i])
        #target_values_tensor = torch.tensor(target_values, requires_grad=False, dtype=torch.float32)
        
        # Update the network with a backward pass
        #states_tensor = torch.tensor(states, requires_grad=False, dtype=torch.float32)
        #values = self.network(states_tensor)
        #values_action = np.zeros((self.batch_size, 1))
        #for i in range(self.batch_size):
        #    values_action[i][0] = values[i][actions[i]]
        
        #values_action_tensor = torch.tensor(values_action, requires_grad=False, dtype=torch.float32)
        #print("Values: ", values)
        #print("tensor target values :", target_values_tensor)
        #print("values action tensor: ", values_action_tensor)

        # Compute loss function
        #loss = nn.functional.mse_loss(values, target_values_tensor)
        loss = nn.functional.mse_loss(values, target_values)
        #loss = nn.functional.mse_loss(values_action_tensor, target_values_tensor)

        # Compute gradient
        loss.backward()

        # Clip gradient norm to 1
        nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=self.clipping_value)

        # Perform backward pass (backpropagation)
        self.optimizer.step()

        return