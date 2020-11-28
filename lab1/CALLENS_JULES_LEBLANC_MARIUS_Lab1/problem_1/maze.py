#%%
import numpy as np
import matplotlib.pyplot as plt
import time
import random
from IPython import display

# Implemented methods
methods = ['DynProg', 'ValIter'];

# Some colours
LIGHT_RED    = '#FFC4CC';
LIGHT_GREEN  = '#95FD99';
BLACK        = '#000000';
WHITE        = '#FFFFFF';
LIGHT_PURPLE = '#E8D0FF';
LIGHT_ORANGE = '#FAE0C3';
BLUE         = '#0000FF';
RED          = '#FF0000';
CHOCOLATE    = '#D2691E';

class State:

    def __init__(self, player_pos, min_pos, too_old = False):
        self.player_pos = player_pos
        self.min_pos = min_pos
        self.too_old = too_old

    def __hash__(self):
        if self.too_old:
            return -1
        else:
            return hash((self.player_pos, self.min_pos))

    def __eq__(self, other_state):
        if self.too_old:
            return other_state.too_old
        else:
            return self.player_pos == other_state.player_pos and self.min_pos == other_state.min_pos and not(other_state.too_old)

class Maze:

    # Actions
    STAY       = 0
    MOVE_LEFT  = 1
    MOVE_RIGHT = 2
    MOVE_UP    = 3
    MOVE_DOWN  = 4

    # Give names to actions
    actions_names = {
        STAY: "stay",
        MOVE_LEFT: "move left",
        MOVE_RIGHT: "move right",
        MOVE_UP: "move up",
        MOVE_DOWN: "move down"
    }

    # Reward values
    STEP_REWARD = -1
    GOAL_REWARD = 0
    IMPOSSIBLE_REWARD = -100
    EATED_REWARD = -100
    TOO_OLD_REWARD = -100


    def __init__(self, maze, min_stay=False, old=False, avg_life=30, weights=None, random_rewards=False):
        """ Constructor of the environment Maze.
        """
        self.maze                     = maze;
        self.old = old
        self.MINOTAUR_CAN_STAY = min_stay
        self.avg_life = avg_life
        self.actions                  = self.__actions();
        self.states, self.map         = self.__states();
        self.n_actions                = len(self.actions);
        self.n_states                 = len(self.states);
        self.transition_probabilities = self.__transitions();
        self.rewards                  = self.__rewards(weights=weights, random_rewards=random_rewards);



    def __actions(self):
        actions = dict();
        actions[self.STAY]       = (0, 0);
        actions[self.MOVE_LEFT]  = (0,-1);
        actions[self.MOVE_RIGHT] = (0, 1);
        actions[self.MOVE_UP]    = (-1,0);
        actions[self.MOVE_DOWN]  = (1,0);
        return actions;

    def __minotaur_actions(self):
        minotaur_actions = dict();
        if self.MINOTAUR_CAN_STAY:
            minotaur_actions[self.STAY] = (0, 0);

        minotaur_actions[self.MOVE_LEFT]  = (0,-1);
        minotaur_actions[self.MOVE_RIGHT] = (0, 1);
        minotaur_actions[self.MOVE_UP]    = (-1,0);
        minotaur_actions[self.MOVE_DOWN]  = (1,0);
        return minotaur_actions;

    def __states(self):
        states = dict();
        map = dict();
        s = 0;
        if self.old:
            new_state = State((-1, -1), (-1, -1), True)
            states[s] = new_state
            map[new_state] = s
            s += 1
        #Minotaur possible states
        for i1 in range(self.maze.shape[0]):
            for j1 in range(self.maze.shape[1]):
                #Player possible states
                for i in range(self.maze.shape[0]):
                    for j in range(self.maze.shape[1]):
                        if self.maze[i,j] != 1:
                            new_state = State((i,j),(i1,j1))
                            states[s] = new_state
                            map[new_state] = s
                            s += 1
        return states, map

    def __move(self, state, action_player, action_min):
        """ Makes a step in the maze, given a current position and an action.
            If the action STAY or an inadmissible action is used, the agent stays in place.

            :return tuple next_cell: Position (x,y) on the maze that agent transitions to.
        """
        # Compute the future position given current (state, action)
        row = self.states[state].player_pos[0] + self.actions[action_player][0];
        col = self.states[state].player_pos[1] + self.actions[action_player][1];

        row_min = self.states[state].min_pos[0] + self.actions[action_min][0];
        col_min = self.states[state].min_pos[1] + self.actions[action_min][1];
        # Is the future position an impossible one ?
        hitting_maze_walls =  (row == -1) or (row == self.maze.shape[0]) or \
                              (col == -1) or (col == self.maze.shape[1]) or \
                              (self.maze[row,col] == 1);

        min_hiting_border = (row_min == -1) or (row_min == self.maze.shape[0]) or \
                            (col_min == -1) or (col_min == self.maze.shape[1])
        if min_hiting_border:
            return None

        # Based on the impossiblity check return the next state.
        if hitting_maze_walls:
            return state;
        else:
            new_state = State((row, col), (row_min,col_min))
            return self.map[new_state];

    def __possible_moves(self, state, action):
        states = []
        for min_action in self.__minotaur_actions():
            new_state = self.__move(state, action, min_action)
            if new_state != None:
                states.append(new_state)
        return states


    def __transitions(self):
        """ Computes the transition probabilities for every state action pair.
            :return numpy.tensor transition probabilities: tensor of transition
            probabilities of dimension S*S*A
        """
        # Initialize the transition probailities tensor (S,S,A)
        dimensions = (self.n_states,self.n_states,self.n_actions);
        transition_probabilities = np.zeros(dimensions);

        # Compute the transition probabilities. Note that the transitions
        # are deterministic.
        if self.old:
            for s in range(self.n_states):
                for a in range(self.n_actions):
                    if s == 0:
                        transition_probabilities[s, s, a] = 1;
                    else:
                        next_states = self.__possible_moves(s,a);
                        prob_to_be_too_old = 1/self.avg_life
                        p = (1 - prob_to_be_too_old) / len(next_states)
                        # Too old case
                        transition_probabilities[0, s, a] = prob_to_be_too_old
                        for next_s in next_states:
                            transition_probabilities[next_s, s, a] = p;
        else:
            for s in range(self.n_states):
                for a in range(self.n_actions):
                    next_states = self.__possible_moves(s,a);
                    p = 1 / len(next_states)
                    for next_s in next_states:
                        transition_probabilities[next_s, s, a] = p;
        return transition_probabilities;

    def __rewards(self, weights=None, random_rewards=None):

        rewards = np.zeros((self.n_states, self.n_actions));

        # If the rewards are not described by a weight matrix
        if weights is None:
            if self.old:
                for s in range(self.n_states):
                    for a in range(self.n_actions):
                        next_states = self.__possible_moves(s, a);
                        for next_s in next_states:
                            # Reward when too old
                            if self.states[s].too_old:
                                rewards[s,a] = self.TOO_OLD_REWARD;
                            # Reward for hitting a wall
                            elif s == next_s and a != self.STAY:
                                rewards[s,a] = self.IMPOSSIBLE_REWARD;
                            # Reward for being eated by the minotaur
                            elif self.states[next_s].player_pos == self.states[next_s].min_pos:
                                rewards[s,a] = self.EATED_REWARD
                            # Reward for reaching the exit
                            elif self.maze[self.states[next_s].player_pos] == 2:
                                rewards[s,a] = self.GOAL_REWARD;
                            # Reward for taking a step to an empty cell that is not the exit
                            else:
                                rewards[s,a] = self.STEP_REWARD;

            else:
                for s in range(self.n_states):
                    for a in range(self.n_actions):
                        next_states = self.__possible_moves(s, a);
                        for next_s in next_states:
                            # Reward for hitting a wall
                            if s == next_s and a != self.STAY:
                                rewards[s,a] = self.IMPOSSIBLE_REWARD;
                            # Reward for being eated by the minotaur
                            elif self.states[next_s].player_pos == self.states[next_s].min_pos:
                                rewards[s,a] = self.EATED_REWARD
                            # Reward for reaching the exit
                            elif self.maze[self.states[next_s].player_pos] == 2:
                                rewards[s,a] = self.GOAL_REWARD;
                            # Reward for taking a step to an empty cell that is not the exit
                            else:
                                rewards[s,a] = self.STEP_REWARD;

        # If the weights are described by a weight matrix
        else:
            for s in range(self.n_states):
                 for a in range(self.n_actions):
                     next_s = self.__move(s,a);
                     i,j = self.states[next_s];
                     # Simply put the reward as the weights o the next state.
                     rewards[s,a] = weights[i][j];

        return rewards;

    def simulate(self, start, start_min, policy, method):
        if method not in methods:
            error = 'ERROR: the argument method must be in {}'.format(methods);
            raise NameError(error);

        path = list();
        if method == 'DynProg':
            # Deduce the horizon from the policy shape
            horizon = policy.shape[1];
            # Initialize current state and time
            t = 0;
            s = self.map[State(start, start_min)];
            # Add the starting position in the maze to the path
            path.append(State(start, start_min));
            while t < horizon-1 and not(self.states[s].player_pos == start_min) and s != 0:
                # Move to next state given the policy and the current state
                if self.old:
                    if random.random() < 1/self.avg_life:
                        next_s = 0
                        print("Is dead from being old")
                    else:
                        next_s = random.choice(self.__possible_moves(s,policy[s,t]));
                else:
                    next_s = random.choice(self.__possible_moves(s,policy[s,t]));
                # Add the position in the maze corresponding to the next state
                # to the path
                path.append(self.states[next_s])
                # Update time and state for next iteration
                t +=1;
                s = next_s;
        if method == 'ValIter':
            # Initialize current state, next state and time
            t = 1;
            s = self.map[start];
            # Add the starting position in the maze to the path
            path.append(start);
            # Move to next state given the policy and the current state
            next_s = self.__move(s,policy[s]);
            # Add the position in the maze corresponding to the next state
            # to the path
            path.append(self.states[next_s]);
            # Loop while state is not the goal state
            while s != next_s:
                # Update state
                s = next_s;
                # Move to next state given the policy and the current state
                next_s = self.__move(s,policy[s]);
                # Add the position in the maze corresponding to the next state
                # to the path
                path.append(self.states[next_s])
                # Update time and state for next iteration
                t +=1;
        print('Simulation done !')
        return path


    def show(self):
        #print('The states are :')
        #print(self.states)
        #print('The actions are:')
        #print(self.actions)
        #print('The mapping of the states:')
        #print(self.map)
        print('The rewards:')
        print(self.rewards)
        print('Initialisation done !')

def dynamic_programming(env, horizon):
    """ Solves the shortest path problem using dynamic programming
        :input Maze env           : The maze environment in which we seek to
                                    find the shortest path.
        :input int horizon        : The time T up to which we solve the problem.
        :return numpy.array V     : Optimal values for every state at every
                                    time, dimension S*T
        :return numpy.array policy: Optimal time-varying policy at every state,
                                    dimension S*T
    """

    # The dynamic prgramming requires the knowledge of :
    # - Transition probabilities
    # - Rewards
    # - State space
    # - Action space
    # - The finite horizon
    p         = env.transition_probabilities;
    r         = env.rewards;
    n_states  = env.n_states;
    n_actions = env.n_actions;
    T         = horizon;

    # The variables involved in the dynamic programming backwards recursions
    V      = np.zeros((n_states, T+1));
    policy = np.zeros((n_states, T+1));
    Q      = np.zeros((n_states, n_actions));


    # Initialization
    Q            = np.copy(r);
    V[:, T]      = np.max(Q,1);
    policy[:, T] = np.argmax(Q,1);

    # The dynamic programming bakwards recursion
    for t in range(T-1,-1,-1):
        # Update the value function acccording to the bellman equation
        for s in range(n_states):
            for a in range(n_actions):
                # Update of the temporary Q values
                Q[s,a] = r[s,a] + np.dot(p[:,s,a],V[:,t+1])
        # Update by taking the maximum Q value w.r.t the action a
        V[:,t] = np.max(Q,1);
        # The optimal action is the one that maximizes the Q function
        policy[:,t] = np.argmax(Q,1);

    print('Dynamic programming done !')
    return V, policy;

def value_iteration(env, gamma, epsilon):
    """ Solves the shortest path problem using value iteration
        :input Maze env           : The maze environment in which we seek to
                                    find the shortest path.
        :input float gamma        : The discount factor.
        :input float epsilon      : accuracy of the value iteration procedure.
        :return numpy.array V     : Optimal values for every state at every
                                    time, dimension S*T
        :return numpy.array policy: Optimal time-varying policy at every state,
                                    dimension S*T
    """
    # The value itearation algorithm requires the knowledge of :
    # - Transition probabilities
    # - Rewards
    # - State space
    # - Action space
    # - The finite horizon
    p         = env.transition_probabilities;
    r         = env.rewards;
    n_states  = env.n_states;
    n_actions = env.n_actions;

    # Required variables and temporary ones for the VI to run
    V   = np.zeros(n_states);
    Q   = np.zeros((n_states, n_actions));
    BV  = np.zeros(n_states);
    # Iteration counter
    n   = 0;
    # Tolerance error
    tol = (1 - gamma)* epsilon/gamma;

    # Initialization of the VI
    for s in range(n_states):
        for a in range(n_actions):
            Q[s, a] = r[s, a] + gamma*np.dot(p[:,s,a],V);
    BV = np.max(Q, 1);

    # Iterate until convergence
    while np.linalg.norm(V - BV) >= tol and n < 200:
        # Increment by one the numbers of iteration
        n += 1;
        # Update the value function
        V = np.copy(BV);
        # Compute the new BV
        for s in range(n_states):
            for a in range(n_actions):
                Q[s, a] = r[s, a] + gamma*np.dot(p[:,s,a],V);
        BV = np.max(Q, 1);
        # Show error
        #print(np.linalg.norm(V - BV))

    # Compute policy
    policy = np.argmax(Q,1);
    # Return the obtained policy
    print("Value iteration done !")
    return V, policy;

def draw_maze(maze):

    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN};

    # Give a color to each cell
    rows,cols    = maze.shape;
    colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)];

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows));

    # Remove the axis ticks and add title title
    ax = plt.gca();
    ax.set_title('The Maze');
    ax.set_xticks([]);
    ax.set_yticks([]);

    # Create a table to color
    grid = plt.table(cellText=None,
                            cellColours=colored_maze,
                            cellLoc='center',
                            loc=(0,0),
                            edges='closed');
    # Modify the hight and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0/rows);
        cell.set_width(1.0/cols);

def animate_solution(maze, path):

    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN};

    # Size of the maze
    rows,cols = maze.shape;

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows));

    # Remove the axis ticks and add title title
    ax = plt.gca();
    ax.set_title('Policy simulation');
    ax.set_xticks([]);
    ax.set_yticks([]);

    # Give a color to each cell
    colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)];

    # Create a table to color
    grid = plt.table(cellText=None,
                     cellColours=colored_maze,
                     cellLoc='center',
                     loc=(0,0),
                     edges='closed');

    # Modify the hight and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0/rows);
        cell.set_width(1.0/cols);


    # Update the color at each frame
    for i in range(len(path)):
        if i > 0:
            grid.get_celld()[(path[i-1].player_pos)].set_facecolor(col_map[maze[path[i-1].player_pos]])
            grid.get_celld()[(path[i-1].player_pos)].get_text().set_text('')
            grid.get_celld()[(path[i-1].min_pos)].set_facecolor(col_map[maze[path[i-1].min_pos]])
            grid.get_celld()[(path[i-1].min_pos)].get_text().set_text('')


        if i > 0:
            if path[i].too_old:
                grid.get_celld()[(path[i-1].player_pos)].set_facecolor(RED)
                grid.get_celld()[(path[i-1].player_pos)].get_text().set_text('Player is too old')
                grid.get_celld()[(path[i-1].min_pos)].set_facecolor(CHOCOLATE)
                grid.get_celld()[(path[i-1].min_pos)].get_text().set_text('Minotaur')
                break

        grid.get_celld()[(path[i].player_pos)].set_facecolor(BLUE)
        grid.get_celld()[(path[i].player_pos)].get_text().set_text('Player')
        grid.get_celld()[(path[i].min_pos)].set_facecolor(CHOCOLATE)
        grid.get_celld()[(path[i].min_pos)].get_text().set_text('Minotaur')

        if i > 0:
            if path[i].player_pos == path[i].min_pos:
                grid.get_celld()[(path[i].player_pos)].set_facecolor(RED)
                grid.get_celld()[(path[i].player_pos)].get_text().set_text('Player has been eaten')
                break
            elif maze[path[i].player_pos[0], path[i].player_pos[1]] == 2 :
                grid.get_celld()[(path[i].player_pos)].set_facecolor(LIGHT_GREEN)
                grid.get_celld()[(path[i].player_pos)].get_text().set_text('Player is out')
                break

        display.display(fig)
        display.clear_output(wait=True)
        time.sleep(0.1)

# %%
