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
duration = 100
path = env.simulate(policy, duration);

#%%
#start_int = env.map[env.START]
#n = 100
#gamma_min = 0.8
#gamma_max = 1.0
#gamma_tab = np.linspace(gamma_min, gamma_max, n)
#V_tab = np.zeros(n)
#
#for index, gamma in enumerate(gamma_tab):
#    print("Itération : ", index)
#    V, _= bk.value_iteration(env, gamma, epsilon);
#    V_tab[index] = V[start_int]
#    
#print(gamma_tab)
#print(V_tab)
#plt.figure()
#plt.xlabel("Lambda")
#plt.ylabel("The value function evaluated at the intial state")
#plt.plot(gamma_tab, V_tab)
#plt.show()


#%%
# Show the shortest path
bk.animate_solution(maze, path, 3)

# %%
