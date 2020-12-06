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
# Last update: 6th October 2020, by alessior@kth.se
#

# Load packages
import numpy as np
import gym
import torch
import time
import matplotlib.pyplot as plt
from tqdm import trange
from DQN_agent import RandomAgent
from DQN_agent import DqnAgent
from DQN_ERB import Experience
from DQN_network import DqnNetwork
from DQN_network import DuelingDqnNetwork

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

def epsilon_linear(k, epsilon_min, epsilon_max, Z):
    return max(epsilon_min, epsilon_max - (epsilon_max-epsilon_min)*(k-1)/(Z-1))

def epsilon_exp(k, epsilon_min, epsilon_max, Z):
    return max(epsilon_min, epsilon_max*(epsilon_min/epsilon_max)**((k-1)/(Z-1)))

def copy_network(copy, to_copy):
    for copy_param, to_copy_param in zip(copy.parameters(), to_copy.parameters()):
        copy_param.data.copy_(to_copy_param)

# Import and initialize the discrete Lunar Laner Environment
env = gym.make('LunarLander-v2')
env.reset()

# Parameters
N_episodes = 500                             # Number of episodes
discount_factor = 0.99                       # Value of the discount factor
n_ep_running_average = 50                    # Running average of 50 episodes
n_actions = env.action_space.n               # Number of available actions
dim_state = len(env.observation_space.high)  # State dimensionality
buffer_size = 30000                          # The buffer size should be between 5000 and 30000
batch_size = 64                              # The training batch size should be between 4 and 128
target_freq_update = buffer_size//batch_size # The target frequency update should be L/N
#target_freq_update = 300                     # The target frequency update should be L/N
e_min = 0.01                                 # The minimal value of epsilon
e_max = 1.0                                  # The maximal value of epsilon
linear_eps = True                            # Take the value of true if the epsilon is linear and false else
Z = N_episodes*0.8                           # Z should be between 90% and 95% of N_episodes
alpha = 5*10**-4                             # The learning rate should be between 10**-3 and 10**-4
clipping_value = 1                           # The clipping value should be between 0.5 and 2
cer_proportion = 1/8                         # This is the proportion of the latest experiences in the sample
cer_mode = True                              # This enable the mode CER
dueling_mode = True                          # This enable the dueling architecture
double_mode = True                           # This enable the double DQN
seed = 42                                    # The seed for reproducibility
see_result = True                            # This enable the visualization of the last network

# Set the seed on the different libraries
torch.manual_seed(seed)
np.random.seed(seed)


# Number of images per seconds and frequence
n_images_per_s = 50
frequence_of_images_per_s = 1/n_images_per_s
period_render = N_episodes // 10

if see_result:
    # Run some experiences of the last network
    nb_of_experience = 50
    load_agent = torch.load('neural-network-1.pt')

    for k in range(nb_of_experience):
        done = False
        state = env.reset()
        total_episode_reward = 0.
        t = 0
        print("Episode: ", k)
        while not done:
            env.render()
            time.sleep(frequence_of_images_per_s)

            # Take epsilon-greedy action
            state_tensor = torch.tensor([state], requires_grad=False, dtype=torch.float32)
            action = load_agent.forward(state_tensor).max(1)[1].item()

            # Get next state and reward.  The done variable
            # will be True if you reached the goal position,
            # False otherwise
            next_state, reward, done, _ = env.step(action)

            # Update episode reward
            total_episode_reward += reward

            # Update state for next iteration
            state = next_state
            t+= 1
        print("Total episode reward: ", total_episode_reward)
        print("Number of steps: ", t)

# We will use these variables to compute the average episodic reward and
# the average number of steps per episode
episode_reward_list = []       # this list contains the total reward per episode
episode_number_of_steps = []   # this list contains the number of steps per episode

# Random agent initialization
#r_agent = RandomAgent(n_actions)

# DQN agent initialization
hidden_layer_size = 64 # The number of hidden layer should be between 8 and 128
hidden_V_and_A_layer_size = hidden_layer_size//2
size_of_layers = [dim_state, hidden_layer_size, hidden_V_and_A_layer_size, n_actions]
#size_of_layers = [dim_state, hidden_layer_size, n_actions]
agent = DqnAgent(n_actions, size_of_layers, buffer_size, discount_factor, batch_size, alpha, clipping_value, cer_mode, cer_proportion, dueling_mode, double_mode, seed)

# Fill the buffer with random experiences
fill_value = 10000
r_agent = RandomAgent(n_actions, seed)
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


### Training process

# trange is an alternative to range in python, from the tqdm library
# It shows a nice progression bar that you can update with useful information
EPISODES = trange(N_episodes, desc='Episode: ', leave=True)
counter_steps = 0
epsilon_i = e_max
max_avg_reward = -1000
if dueling_mode:
    avg_network = DuelingDqnNetwork(size_of_layers, seed)
else:
    avg_network = DqnNetwork(size_of_layers, seed)

copy_network(avg_network, agent.network)


for i in EPISODES:
    # Reset environment data and initialize variables
    done = False
    state = env.reset()
    total_episode_reward = 0.
    if linear_eps:
        epsilon_i = epsilon_linear(i, e_min, e_max, Z)
    else:
        epsilon_i = epsilon_exp(i, e_min, e_max, Z)
    print("Epsilon: ", epsilon_i)
    t = 0
    while not done:
        if i % period_render == 0:
            env.render()
            time.sleep(frequence_of_images_per_s)

        # Take epsilon-greedy action
        action = agent.forward(state, epsilon_i)

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
            agent.backward()

        # If C steps have passed, the target network will be update
        if counter_steps % target_freq_update == 0:
            agent.target_equal_to_main()

        # Update episode reward
        total_episode_reward += reward

        # Update state for next iteration
        state = next_state
        t+= 1
        counter_steps += 1

    # Append episode reward and total number of steps
    episode_reward_list.append(total_episode_reward)
    episode_number_of_steps.append(t)
    avg_reward = running_average(episode_reward_list, n_ep_running_average)[-1]
    if max_avg_reward < avg_reward:
        max_avg_reward = avg_reward
        copy_network(avg_network, agent.network)

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

# Save the network

torch.save(agent.network, 'neural-network-2.pt')