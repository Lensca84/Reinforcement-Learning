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
# Last update: 20th November 2020, by alessior@kth.se
#

# Load packages
import numpy as np
import gym
import torch
import matplotlib.pyplot as plt
from tqdm import trange
from DDPG_agent import RandomAgent
from DDPG_agent import DdpgAgent
from DDPG_network import DdpgActorNetwork
from DDPG_network import DdpgCriticNetwork
from DDPG_ERB import Experience
from DDPG_soft_updates import soft_updates
import time


def running_average(x, N):
    ''' Function used to compute the running average
        of the last N elements of a vector x
    '''
    if len(x) >= N:
        y = np.copy(x)
        y[N-1:] = np.convolve(x, np.ones((N, )) / N, mode='valid')
    else:
        y = np.zeros_like(x)
    return y

# Import and initialize Mountain Car Environment
env = gym.make('LunarLanderContinuous-v2')
env.reset()

# Parameters
N_episodes = 300                             # Number of episodes to run for training
discount_factor = 0.99                       # Value of gamma
n_ep_running_average = 50                    # Running average of 50 episodes
m = len(env.action_space.high)               # dimensionality of the action
dim_state = len(env.observation_space.high)  # State dimensionality
buffer_size = 30000                          # The buffer size should be between 5000 and 30000
batch_size = 64                              # The training batch size should be between 4 and 128
d = 2                                        # Target frequence update and actor frequence update
alpha_critic = 5*10**-4                      # The learning rate should be between 10**-3 and 10**-4
alpha_actor = 5*10**-5                       # The learning rate should be between 10**-3 and 10**-4
clipping_value = 1                           # The clipping value should be between 0.5 and 2
tau = batch_size/buffer_size                 # The coefficient of the update for the target
mu = 0.15                                    # The coefficient of the noise
sigma = 0.2                                  # The variance of the noise
seed = 42                                    # Take a seed to do reproducibility

# Set the seed
torch.manual_seed(seed)
np.random.seed(seed)

# Number of images per seconds and frequence
n_images_per_s = 50
frequence_of_images_per_s = 1/n_images_per_s
period_render = N_episodes // 10

# Reward
episode_reward_list = []  # Used to save episodes reward
episode_number_of_steps = []

# DDPG Agent initialization
agent = DdpgAgent(m, dim_state, buffer_size, discount_factor, batch_size, alpha_actor, alpha_critic, clipping_value, tau, mu, sigma, d, seed)

# Fill the buffer with random experiences
fill_value = 10000
r_agent = RandomAgent(m, seed)
percent_fill_value = fill_value // 100

while len(agent.buffer) < fill_value:
    # Reset environment data and initialize variables
    done = False
    state = env.reset()
    t = 0
    while not done:
        if len(agent.buffer) % percent_fill_value == 0:
            print("Buffer initialisation progress: ", len(agent.buffer)//percent_fill_value , "%")
        # Take a random action
        action = r_agent.forward(state)

        # Get next state and reward.  The done variable
        # will be True if you reached the goal position,
        # False otherwise
        next_state, reward, done, _ = env.step(action)

        # Append the new experience to the buffer
        agent.buffer.append(Experience(state, action, reward, next_state, done))

        # Update state for next iteration
        state = next_state
        t+= 1

    # Close environment
    env.close()


# Training process
EPISODES = trange(N_episodes, desc='Episode: ', leave=True)
max_avg_reward = -1000
avg_actor_network = DdpgActorNetwork(m, dim_state, seed)
avg_critic_network = DdpgCriticNetwork(m, dim_state, seed)

soft_updates(agent.critic_network, avg_critic_network, 1)
soft_updates(agent.actor_network, avg_actor_network, 1)

for i in EPISODES:
    # Reset enviroment data
    done = False
    state = env.reset()
    total_episode_reward = 0.
    t = 0
    while not done:
        if i % period_render == 0:
            env.render()
            time.sleep(frequence_of_images_per_s)

        # Take a random action
        action = agent.forward(state)

        # Get next state and reward.  The done variable
        # will be True if you reached the goal position,
        # False otherwise
        next_state, reward, done, _ = env.step(action)

        # Append the new experience to the buffer
        agent.buffer.append(Experience(state, action, reward, next_state, done))

        if len(agent.buffer) >= batch_size:
            # Sample a random batch
            # Compute the target values
            # And update the network with a backward pass
            agent.backward(t)

        # Update episode reward
        total_episode_reward += reward

        # Update state for next iteration
        state = next_state
        t+= 1

    # Append episode reward
    episode_reward_list.append(total_episode_reward)
    episode_number_of_steps.append(t)

    # Save the best average networks
    avg_reward = running_average(episode_reward_list, n_ep_running_average)[-1]
    if max_avg_reward < avg_reward:
        max_avg_reward = avg_reward
        soft_updates(agent.critic_network, avg_critic_network, 1)
        soft_updates(agent.actor_network, avg_actor_network, 1)

    # Close environment
    env.close()

    # Updates the tqdm update bar with fresh information
    # (episode number, total reward of the last episode, total number of Steps
    # of the last episode, average reward, average number of steps)
    EPISODES.set_description(
        "Episode {} - Reward/Steps: {:.1f}/{} - Avg. Reward/Steps: {:.1f}/{}".format(
        i, total_episode_reward, t,
        running_average(episode_reward_list, n_ep_running_average)[-1],
        running_average(episode_number_of_steps, n_ep_running_average)[-1]))


# Plot Rewards and steps
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))
ax[0].plot([i for i in range(1, N_episodes+1)], episode_reward_list, label='Episode reward')
ax[0].plot([i for i in range(1, N_episodes+1)], running_average(
    episode_reward_list, n_ep_running_average), label='Avg. episode reward')
ax[0].set_xlabel('Episodes')
ax[0].set_ylabel('Total reward')
ax[0].set_title('Total Reward vs Episodes')
ax[0].legend()
ax[0].grid(alpha=0.3)

ax[1].plot([i for i in range(1, N_episodes+1)], episode_number_of_steps, label='Steps per episode')
ax[1].plot([i for i in range(1, N_episodes+1)], running_average(
    episode_number_of_steps, n_ep_running_average), label='Avg. number of steps per episode')
ax[1].set_xlabel('Episodes')
ax[1].set_ylabel('Total number of steps')
ax[1].set_title('Total number of steps vs Episodes')
ax[1].legend()
ax[1].grid(alpha=0.3)
plt.show()

# Save the networks
torch.save(agent.actor_network, 'neural-network-2-actor.pth')
torch.save(agent.critic_network, 'neural-network-2-critic.pth')