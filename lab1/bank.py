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
YELLOW       = '#FFFF00';
ORANGE       = '#FFA500';

class State:

    def __init__(self, player_pos, police_pos):
        self.player_pos = player_pos
        self.police_pos = police_pos

    def __hash__(self):
        return hash((self.player_pos, self.police_pos))

    def __eq__(self, other_state):
        return self.player_pos == other_state.player_pos and self.police_pos == other_state.police_pos

class Bank:

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
    STEP_REWARD = 0
    BANK_REWARD = 10
    IMPOSSIBLE_REWARD = -100
    CAUGHT_REWARD = -50

    # Starting places
    START = State((0,0),(1,2))


    def __init__(self, maze):
        """ Constructor of the environment Maze.
        """
        self.maze                     = maze;
        self.actions                  = self.__actions();
        self.states, self.map         = self.__states();
        self.n_actions                = len(self.actions);
        self.n_states                 = len(self.states);
        self.transition_probabilities = self.__transitions();
        self.rewards                  = self.__rewards();

    def __actions(self):
        actions = dict();
        actions[self.STAY]       = (0, 0);
        actions[self.MOVE_LEFT]  = (0,-1);
        actions[self.MOVE_RIGHT] = (0, 1);
        actions[self.MOVE_UP]    = (-1,0);
        actions[self.MOVE_DOWN]  = (1,0);
        return actions;

    def __police_actions(self):
        police_actions = dict();
        police_actions[self.MOVE_LEFT]  = (0,-1);
        police_actions[self.MOVE_RIGHT] = (0, 1);
        police_actions[self.MOVE_UP]    = (-1,0);
        police_actions[self.MOVE_DOWN]  = (1,0);
        return police_actions;

    def __states(self):
        states = dict();
        map = dict();
        s = 0;
        #Police possible states
        for i1 in range(self.maze.shape[0]):
            for j1 in range(self.maze.shape[1]):
                #Player possible states
                for i in range(self.maze.shape[0]):
                    for j in range(self.maze.shape[1]):
                        new_state = State((i,j),(i1,j1))
                        states[s] = new_state
                        map[new_state] = s
                        s += 1
        return states, map

    def __move(self, state, action_player, action_pol):
        """ Makes a step in the maze, given a current position and an action.
            If the action STAY or an inadmissible action is used, the agent stays in place.

            :return tuple next_cell: Position (x,y) on the maze that agent transitions to.
        """
        # Compute the future position given current (state, action)
        row = self.states[state].player_pos[0] + self.actions[action_player][0];
        col = self.states[state].player_pos[1] + self.actions[action_player][1];

        row_pol = self.states[state].police_pos[0] + self.actions[action_pol][0];
        col_pol = self.states[state].police_pos[1] + self.actions[action_pol][1];
        # Is the future position an impossible one ?
        hitting_maze_walls =  (row == -1) or (row == self.maze.shape[0]) or \
                              (col == -1) or (col == self.maze.shape[1]);

        pol_hiting_border = (row_pol == -1) or (row_pol == self.maze.shape[0]) or \
                            (col_pol == -1) or (col_pol == self.maze.shape[1])
        player_row = self.states[state].player_pos[0]
        player_col = self.states[state].player_pos[1]
        pol_row = self.states[state].police_pos[0]
        pol_col = self.states[state].police_pos[1]

        pol_top_player            = (player_col == pol_col and player_row > pol_row) and \
                                    (action_pol == self.MOVE_LEFT or action_pol == self.MOVE_RIGHT or \
                                    action_pol == self.MOVE_DOWN)
        pol_top_right_player      = (player_col < pol_col and player_row > pol_row) and \
                                    (action_pol == self.MOVE_DOWN or action_pol == self.MOVE_LEFT)
        pol_right_player          = (player_col < pol_col and player_row == pol_row) and \
                                    (action_pol == self.MOVE_LEFT or action_pol == self.MOVE_UP or \
                                    action_pol == self.MOVE_DOWN)
        pol_bottom_right_player   = (player_col < pol_col and player_row < pol_row) and \
                                    (action_pol == self.MOVE_UP or action_pol == self.MOVE_LEFT)
        pol_bottom_player         = (player_col == pol_col and player_row < pol_row) and \
                                    (action_pol == self.MOVE_LEFT or action_pol == self.MOVE_UP or \
                                    action_pol == self.MOVE_RIGHT)
        pol_bottom_left_player    = (player_col > pol_col and player_row < pol_row) and \
                                    (action_pol == self.MOVE_UP or action_pol == self.MOVE_RIGHT)
        pol_left_player           = (player_col > pol_col and player_row == pol_row) and \
                                    (action_pol == self.MOVE_DOWN or action_pol == self.MOVE_UP or \
                                    action_pol == self.MOVE_RIGHT)
        pol_top_left_player       = (player_col > pol_col and player_row > pol_row) and \
                                    (action_pol == self.MOVE_DOWN or action_pol == self.MOVE_RIGHT)

        pol_player_direction = pol_top_player or pol_top_right_player or pol_right_player or \
                               pol_bottom_right_player or pol_bottom_player or pol_bottom_left_player or \
                               pol_left_player or pol_top_left_player

        if pol_hiting_border or not(pol_player_direction):
            return None

        # Based on the impossiblity check return the next state.
        if hitting_maze_walls:
            return state;
        else:
            new_state = State((row, col), (row_pol,col_pol))
            return self.map[new_state];

    def __possible_moves(self, state, action):
        states = []
        for pol_action in self.__police_actions():
            new_state = self.__move(state, action, pol_action)
            if new_state != None:
                states.append(new_state)
        return states

    def __is_caught(self, s):
        state = self.states[s]
        return state.player_pos == state.police_pos

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
        for s in range(self.n_states):
            for a in range(self.n_actions):
                if self.__is_caught(s):
                    transition_probabilities[self.map[self.START], s, a] = 1;
                else:
                    next_states = self.__possible_moves(s,a);
                    p = 1 / len(next_states)
                    for next_s in next_states:
                        transition_probabilities[next_s, s, a] = p;
        return transition_probabilities;

    def __rewards(self):

        rewards = np.zeros((self.n_states, self.n_actions));

        for s in range(self.n_states):
            for a in range(self.n_actions):
                next_states = self.__possible_moves(s, a);
                # Reward for being caught by the police
                if self.__is_caught(s):
                    rewards[s,a] = self.CAUGHT_REWARD;
                # Reward for hitting a wall
                elif s == next_states[0] and a != self.STAY:
                    rewards[s,a] = self.IMPOSSIBLE_REWARD;
                # Reward for reaching the exit
                elif self.maze[self.states[s].player_pos] == 1:
                    rewards[s,a] = self.BANK_REWARD;
                # Reward for taking a step to an empty cell that is not the exit
                else:
                    rewards[s,a] = self.STEP_REWARD;

        return rewards;

    def simulate(self, policy, duration):
        path = list();
        # Initialize current state and time
        t = 0;
        s = self.map[self.START];
        # Add the starting position in the maze to the path
        path.append(self.START);
        while t < duration:
            # Move to next state given the policy and the current state
            if self.__is_caught(s):
                next_s = self.START;
            else:
                next_s = random.choice(self.__possible_moves(s,policy[s]));
            # Add the position in the maze corresponding to the next state
            # to the path
            path.append(self.states[next_s])
            # Update time and state for next iteration
            t +=1;
            s = next_s;
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
    #print("Value iteration done !")
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

def animate_solution(maze, path, start_view=0):

    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: YELLOW};

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
    for i in range(start_view,len(path)):
        if i > 0:
            grid.get_celld()[(path[i-1].player_pos)].set_facecolor(col_map[maze[path[i-1].player_pos]])
            grid.get_celld()[(path[i-1].player_pos)].get_text().set_text('')
            grid.get_celld()[(path[i-1].police_pos)].set_facecolor(col_map[maze[path[i-1].police_pos]])
            grid.get_celld()[(path[i-1].police_pos)].get_text().set_text('')

        grid.get_celld()[(path[i].player_pos)].set_facecolor(ORANGE)
        grid.get_celld()[(path[i].player_pos)].get_text().set_text('Player')
        if i%2 == 0:
            grid.get_celld()[(path[i].police_pos)].set_facecolor(BLUE)
            grid.get_celld()[(path[i].police_pos)].get_text().set_text('Police')
        if i%2 != 0:
            grid.get_celld()[(path[i].police_pos)].set_facecolor(BLUE)
            grid.get_celld()[(path[i].police_pos)].get_text().set_text('Police')

        if i > 0:
            if path[i].player_pos == path[i].police_pos:
                grid.get_celld()[(path[i].player_pos)].set_facecolor(LIGHT_RED)
                grid.get_celld()[(path[i].player_pos)].get_text().set_text('Player has been caught')
            elif maze[path[i].player_pos[0], path[i].player_pos[1]] == 1:
                grid.get_celld()[(path[i].player_pos)].set_facecolor(ORANGE)
                grid.get_celld()[(path[i].player_pos)].get_text().set_text('Robbing')

        display.display(fig)
        display.clear_output(wait=True)
        time.sleep(0.1)

# %%
