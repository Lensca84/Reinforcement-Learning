#%%
import maze as mz
import numpy as np
import matplotlib.pyplot as plt

# Description of the maze as a numpy array
maze = np.array([
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 1, 1, 1],
    [0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 1, 2, 0, 0]
])

# Create an environment maze
MINOTAUR_CAN_STAY = True


#env = mz.Maze(maze,MINOTAUR_CAN_STAY)
#env2 = mz.Maze(maze)
#env.show()

# # Finite horizon
# horizon = 20
# # Solve the MDP problem with dynamic programming
# V, policy= mz.dynamic_programming(env,horizon);
#
# # Simulate the shortest path starting from position A
# method = 'DynProg';
# start  = (0,0);
# start_min = (6,5)
# path = env.simulate(start, start_min, policy, method);

# Show the shortest path
#mz.animate_solution(maze, path)

#N = 10000
#method = 'DynProg';
#start  = (0,0);
#start_min = (6,5)
#Tmin = 14
#Tmax = 40
#tab = np.zeros(Tmax-Tmin)
#tab2 = np.zeros(Tmax-Tmin)
#
#for T in range(Tmin,Tmax):
#    count = 0
#    count2 = 0
#    print('Horizon = ', T)
#    V, policy= mz.dynamic_programming(env,T)
#    V2, policy2= mz.dynamic_programming(env2,T)
#    for _ in range(N):
#        path = env.simulate(start, start_min, policy, method);
#        if path[-1].player_pos == start_min:
#            count += 1
#        path2 = env2.simulate(start, start_min, policy2, method);
#        if path2[-1].player_pos == start_min:
#            count2 += 1
#    tab[T-Tmin] = count/N
#    tab2[T-Tmin] = count2/N
#
#plt.figure(1)
#Time = np.arange(Tmin,Tmax)
#plt.plot(Time,tab,label='Minotaur can stay')
#plt.plot(Time,tab2,label='Minotaur cannot stay')
#plt.xlabel('Time horizon')
#plt.ylabel('Maximal probability of exiting the maze')
#plt.grid()
#plt.legend()
#plt.show()


old = True
avg_life = 30

env = mz.Maze(maze,MINOTAUR_CAN_STAY, old, avg_life)

# Finite horizon
horizon = 100
# Solve the MDP problem with dynamic programming
V, policy= mz.dynamic_programming(env,horizon);

#%%
method = 'DynProg';
start  = (0,0);
start_min = (6,5)
path = env.simulate(start, start_min, policy, method);

#%%
# Show the shortest path
mz.animate_solution(maze, path)
# %%
N = 100000
method = 'DynProg';
start  = (0,0);
start_min = (6,5)
counter_eaten = 0
counter_old = 0
counter_out = 0
avg_age_general = 0
avg_age_old = 0
avg_age_out = 0

for k in range(N):
    path = env.simulate(start, start_min, policy, method);
    if path[-1].too_old:
        avg_age_old += len(path)
        counter_old += 1
    elif path[-1].player_pos == start_min:
        counter_out += 1
        avg_age_out += len(path)
    else:
        counter_eaten += 1
    avg_age_general += len(path)

avg_age_general /= N
avg_age_old /= counter_old
avg_age_out /= counter_out

print("Number of death from oldness : ", counter_old)
print("Number of death from being eaten : ", counter_eaten)
print("Number of going out : ", counter_out)
print("Probability of getting out alive : ", counter_out/N)
print("Average age in general : ", avg_age_general)
print("Average age from oldness : ", avg_age_old)
print("Average age when we are out : ", avg_age_out)


# %%
