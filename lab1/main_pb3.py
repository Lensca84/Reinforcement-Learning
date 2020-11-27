#%%
import robber as rb
import numpy as np
import matplotlib.pyplot as plt

# Description of the maze as a numpy array
maze = np.array([
    [0, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0]
])

# Create an environment maze
env = rb.Bank(maze)
gamma = 0.99

# Simulate the shortest path starting from position A
duration = 100000

path, V_s0, Q = env.simulate_QLearning(gamma, duration);

print("Q : ", Q)

#%%

time = np.arange(duration)
    
plt.figure()
plt.xlabel("Time t")
plt.ylabel("The value function evaluated at the intial state")
plt.plot(time, V_s0)
plt.show()

#rb.animate_solution(maze, path)

# %%
