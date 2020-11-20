#%%
import bank as bk
import numpy as np
import matplotlib.pyplot as plt

# Description of the maze as a numpy array
maze = np.array([
    [1, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 1]
])

# Create an environment maze
env = bk.Bank(maze)
gamma = 0.99
epsilon = 10**-3

# Solve the MDP problem with dynamic programming
V, policy= bk.value_iteration(env, gamma, epsilon);

# Simulate the shortest path starting from position A
path = env.simulate(policy);

#%%
# Show the shortest path
bk.animate_solution(maze, path)