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
gamma = 0.8

# Simulate the shortest path starting from position A
duration = 1000000

#%%
path, V_s0, Q = env.simulate_QLearning(gamma, duration);

print("Q : ", Q)

#%%
time = np.arange(duration)
print("Vs0 : ", V_s0)

plt.figure()
plt.xlabel("Time t")
plt.ylabel("The value function evaluated at the intial state V(s0)")
plt.plot(time, V_s0)
plt.show()

#rb.animate_solution(maze, path)

# %%
epsilon = 0.1
path, V_s0, Q = env.simulate_SARSA(gamma, epsilon, duration);

print("Q : ", Q)

#%%
time = np.arange(duration)
print("Vs0 : ", V_s0)

plt.figure()
plt.xlabel("Time t")
plt.ylabel("The value function evaluated at the intial state V(s0)")
plt.plot(time, V_s0)
plt.show()

#%%
start_simu = 500000
rb.animate_solution(maze, path, start_simu)
# %%

plt.figure()
plt.xlabel("Time t")
plt.ylabel("The value function evaluated at the intial state V(s0)")

time = np.arange(duration)
epsilons = [0.1, 0.3, 0.5, 0.7, 0.9]
for epsilon in epsilons:
    _, V_s0, _ = env.simulate_SARSA(gamma, epsilon, duration);
    plt.plot(time, V_s0, label="With epsilon = "+str(epsilon))
    print("Epsilon : ", epsilon)

plt.legend()
plt.show()

# %%

plt.figure()
plt.xlabel("Time t")
plt.ylabel("The value function evaluated at the intial state V(s0)")
epsilon = 0.1
time = np.arange(duration)
_, V_s0, _ = env.simulate_SARSA(gamma, epsilon, duration)
plt.plot(time, V_s0, label="With epsilon = "+str(epsilon))

_, V_s0, _ = env.simulate_SARSA_epsilon(gamma, duration)
plt.plot(time, V_s0, label="With epsilon = 1/t")

_, V_s0, _ = env.simulate_QLearning(gamma, duration);
plt.plot(time, V_s0, label="Q learning")

plt.legend()
plt.show()

# %%
