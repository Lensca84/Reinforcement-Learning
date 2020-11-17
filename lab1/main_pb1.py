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
env = mz.Maze(maze)
env.show()

# Finite horizon
horizon = 20
# Solve the MDP problem with dynamic programming
V, policy= mz.dynamic_programming(env,horizon);

# Simulate the shortest path starting from position A
method = 'DynProg';
start  = (0,0);
start_min = (6,5)
path = env.simulate(start, start_min, policy, method);

# Show the shortest path
mz.animate_solution(maze, path)