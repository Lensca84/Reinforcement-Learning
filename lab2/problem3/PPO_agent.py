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
# Course: EL2805 - Reinforcement Learning - Lab 2 Problem 3
# Code author: [Alessio Russo - alessior@kth.se]
# Last update: 29th October 2020, by alessior@kth.se
#

# Load packages
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PPO_network import PpoActorNetwork
from PPO_network import PpoCriticNetwork
from PPO_ERB import ExperienceReplayBuffer

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
                    the parent class Agent
        '''
        return np.clip(-1 + 2 * np.random.rand(self.n_actions), -1, 1)

class PpoAgent(Agent):

    def __init__(self, n_actions: int, dim_state, discount_factor, alpha_actor, alpha_critic, epsilon, clipping_value, seed):
        super(PpoAgent, self).__init__(n_actions)
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.buffer = ExperienceReplayBuffer(1001, seed)
        self.discount_factor = discount_factor
        self.epsilon = epsilon

        self.actor_network = PpoActorNetwork(n_actions, dim_state, seed)
        self.old_actor = None
        self.critic_network = PpoCriticNetwork(n_actions, dim_state, seed)
        
        self.optimizer_actor = optim.Adam(self.actor_network.parameters(), lr=alpha_actor)
        self.optimizer_critic = optim.Adam(self.critic_network.parameters(), lr=alpha_critic)

        self.clipping_value = clipping_value
    
    def forward(self, state):
        state_tensor = torch.tensor([state], requires_grad=False, dtype=torch.float32)
        mu, sigma = self.actor_network(state_tensor)
        try:
            action = np.random.normal(mu.detach().numpy()[0], torch.sqrt(sigma).detach().numpy()[0], self.n_actions)
            return action
        except:
            print("WTF les amis !!")
            print("state: ", state)
            print("Sigma: ", sigma)
            print("mu: ", mu)
            print("sqrt sigma: ", torch.sqrt(sigma))
            return [0, -1]


    def backward(self, epoch):
        # Sample a random batch of experiences
        states, actions, rewards, _, _ = self.buffer.iterator()
        t = len(self.buffer)

        states_tensor = torch.tensor(states, requires_grad=False, dtype=torch.float32)
        actions_tensor = torch.tensor(actions, requires_grad=False, dtype=torch.float32)

        ## Backward for the critic function

        # Compute the target values
        target_critic_values = np.zeros(t)
        for i in range(t):
            for n in range(i, t):
                target_critic_values[i] += self.discount_factor**(n-i)*rewards[n]
        target_critic_values_tensor = torch.tensor(target_critic_values, requires_grad=False, dtype=torch.float32).unsqueeze(1)

        # Get the values of the network
        critic_values = self.critic_network(states_tensor)

        self.optimizer_critic.zero_grad()
        # Compute the critic loss function
        loss_critic = nn.functional.mse_loss(target_critic_values_tensor, critic_values)

        # Compute gradient
        loss_critic.backward()

        # Clip gradient norm to clipping value
        nn.utils.clip_grad_norm_(self.critic_network.parameters(), max_norm=self.clipping_value)

        # Perform backward pass (backpropagation)
        self.optimizer_critic.step()

        ## Backward for the actor function

        # Compute phi the advantage function
        advantage_function = (target_critic_values_tensor - critic_values).detach()

        # Compute of the actor values
        mu, sigma = self.actor_network(states_tensor)

        # Compute the policy given the actions and states
        # The actions for one states are independent so we can multiply them together
        policy_action_state = torch.ones(t,requires_grad=False)
        for k in range(self.n_actions):
            policy_action_state *= (1/torch.sqrt(2*np.pi*torch.square(sigma[:,k]))) * torch.exp(-torch.square(actions_tensor[:,k]-mu[:,k])/(2*torch.square(sigma[:,k])))
        policy_action_state = policy_action_state.unsqueeze(1)

        if epoch == 0:
            self.old_actor = policy_action_state.detach()
        r_theta = policy_action_state / self.old_actor

        # Compute the constraint of r_theta
        max_born_epsilon = torch.ones((t,1), requires_grad=False) + self.epsilon
        min_born_epsilon = torch.ones((t,1), requires_grad=False) - self.epsilon
        constraint_max = torch.min(r_theta, max_born_epsilon)
        constraint_r_theta = torch.max(constraint_max, min_born_epsilon)

        min_constraint_r_theta = torch.min(r_theta*advantage_function, constraint_r_theta*advantage_function)

        self.optimizer_actor.zero_grad()
        # Compute the critic loss function
        loss_actor = -torch.mean(min_constraint_r_theta, dim=0, keepdim=True)[0][0]

        # Compute gradient
        loss_actor.backward()
        
        # Clip gradient norm to clipping value
        nn.utils.clip_grad_norm_(self.actor_network.parameters(), max_norm=self.clipping_value)

        # Perform backward pass (backpropagation)
        self.optimizer_actor.step()

        return