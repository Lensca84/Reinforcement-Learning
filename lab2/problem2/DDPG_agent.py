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
# Course: EL2805 - Reinforcement Learning - Lab 2 Problem 2
# Code author: [Alessio Russo - alessior@kth.se]
# Last update: 26th October 2020, by alessior@kth.se
#

# Load packages
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from DDPG_network import DdpgActorNetwork
from DDPG_network import DdpgCriticNetwork
from DDPG_ERB import ExperienceReplayBuffer
from DDPG_soft_updates import soft_updates


class Agent(object):
    ''' Base agent class

        Args:
            n_actions (int): actions dimensionality

        Attributes:
            n_actions (int): where we store the dimensionality of an action
    '''
    def __init__(self, n_actions: int):
        self.n_actions = n_actions

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

    def forward(self, state: np.ndarray) -> np.ndarray:
        ''' Compute a random action in [-1, 1]

            Returns:
                action (np.ndarray): array of float values containing the
                    action. The dimensionality is equal to self.n_actions from
                    the parent class Agent.
        '''
        return np.clip(-1 + 2 * np.random.rand(self.n_actions), -1, 1)

class DdpgAgent(Agent):

    def __init__(self, n_actions: int, dim_state, buffer_size, discount_factor, batch_size, alpha_actor, alpha_critic, clipping_value, tau, mu, sigma, d, seed):
        super(DdpgAgent, self).__init__(n_actions)
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.buffer = ExperienceReplayBuffer(buffer_size, seed)
        self.discount_factor = discount_factor
        self.batch_size = batch_size

        self.actor_network = DdpgActorNetwork(n_actions, dim_state, seed)
        self.target_actor_network = DdpgActorNetwork(n_actions, dim_state, seed)
        self.critic_network = DdpgCriticNetwork(n_actions, dim_state, seed)
        self.target_critic_network = DdpgCriticNetwork(n_actions, dim_state, seed)
        soft_updates(self.actor_network, self.target_actor_network, 1)
        soft_updates(self.critic_network, self.target_critic_network, 1)
        
        self.optimizer_actor = optim.Adam(self.actor_network.parameters(), lr=alpha_actor)
        self.optimizer_critic = optim.Adam(self.critic_network.parameters(), lr=alpha_critic)
        self.clipping_value = clipping_value
        self.n_t = 0
        self.mu = mu
        self.sigma = sigma
        self.tau = tau
        self.d = d

    
    def forward(self, state):
        state_tensor = torch.tensor([state], requires_grad=False, dtype=torch.float32)
        action = self.actor_network(state_tensor).detach().numpy()[0] + self.n_t
        wt = np.random.normal(0, self.sigma, self.n_actions)
        self.n_t = -self.mu*self.n_t + wt
        return action

    def backward(self, t):
        # Sample a random batch of experiences
        states, actions, rewards, next_states, dones = self.buffer.sample_batch(self.batch_size)

        states_tensor = torch.tensor(states, requires_grad=False, dtype=torch.float32)
        actions_tensor = torch.tensor(actions, requires_grad=False, dtype=torch.float32)
        rewards_tensor = torch.tensor(rewards, requires_grad=False, dtype=torch.float32).unsqueeze(1)
        next_states_tensor = torch.tensor(next_states, requires_grad=False, dtype=torch.float32)
        dones_tensor = torch.tensor(dones, requires_grad=False, dtype=torch.float32).unsqueeze(1)

        ## Backward for the critic function

        # Get the best actions for the target network on the next states
        target_actor_values = self.target_actor_network(next_states_tensor)
        # Get the critic values for the target network with the actions of the target actor network
        target_critic_values = self.target_critic_network(next_states_tensor, target_actor_values)
        # Compute the target values
        target_critic_values = rewards_tensor + (self.discount_factor*target_critic_values*(1-dones_tensor)) 

        # Get the values of the network
        critic_values = self.critic_network(states_tensor, actions_tensor)

        self.optimizer_critic.zero_grad()
        # Compute the critic loss function
        loss_critic = nn.functional.mse_loss(target_critic_values, critic_values)

        # Compute gradient
        loss_critic.backward()

        # Clip gradient norm to clipping value
        nn.utils.clip_grad_norm_(self.critic_network.parameters(), max_norm=self.clipping_value)

        # Perform backward pass (backpropagation)
        self.optimizer_critic.step()

        if t % self.d == 0:
            ## Backward for the actor function

            # Get the action for the network on the states
            actor_values = self.actor_network(states_tensor)
            # Get the best values for the network with the actions of the actor network
            critic_values = self.critic_network(states_tensor, actor_values)

            self.optimizer_actor.zero_grad()
            # Compute the critic loss function
            loss_actor = -torch.mean(critic_values, dim=0, keepdim=True)

            # Compute gradient
            loss_actor.backward()

            # Clip gradient norm to clipping value
            nn.utils.clip_grad_norm_(self.actor_network.parameters(), max_norm=self.clipping_value)

            # Perform backward pass (backpropagation)
            self.optimizer_actor.step()

            ## Soft update of the target networks
            soft_updates(self.critic_network, self.target_critic_network, self.tau)
            soft_updates(self.actor_network, self.target_actor_network, self.tau)

        return