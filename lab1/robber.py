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
    BANK_REWARD = 1
    IMPOSSIBLE_REWARD = -100
    CAUGHT_REWARD = -10

    # Starting places
    START = State((0,0),(3,3))


    def __init__(self, maze):
        """ Constructor of the environment Maze.
        """
        self.maze                     = maze;
        self.actions                  = self.__actions();
        self.states, self.map         = self.__states();
        self.n_actions                = len(self.actions);
        self.n_states                 = len(self.states);

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

        if pol_hiting_border:
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

    def r(self, s, a):
        next_states = self.__possible_moves(s, a);
        # Reward for being caught by the police
        if self.__is_caught(s):
            return self.CAUGHT_REWARD;
        # Reward for hitting a wall
        elif s == next_states[0] and a != self.STAY:
            return self.IMPOSSIBLE_REWARD;
        # Reward for reaching the exit
        elif self.maze[self.states[s].player_pos] == 1:
            return self.BANK_REWARD;
        # Reward for taking a step to an empty cell that is not the exit
        else:
            return self.STEP_REWARD;

    def caracteristic(self,n,s,a):
        nsa = n[s,a]
        if nsa == 0:
            return 0
        else:
            return 1/(nsa**(2/3))

    def policy(self):
        return random.randint(0,4)

    def simulate_QLearning(self, gamma, duration):
        path = list();
        # Initialize current state and time
        t = 0;
        s0 = self.map[self.START];
        s = s0
        Q = np.zeros((self.n_states,self.n_actions))
        n = np.zeros((self.n_states,self.n_actions))
        V_s0 = np.zeros(duration)
        period = duration//100
        # Add the starting position in the maze to the path
        path.append(self.START);
        while t < duration:
            #2. Observations
            a = self.policy()
            next_s = random.choice(self.__possible_moves(s,a))
            #3. Q-function improvement
            Q[s,a] += self.caracteristic(n,s,a)*(self.r(s,a)+gamma*(max(Q[next_s])-Q[s,a]))
            V_s0[t] = max(Q[s])
            # Add the position in the maze corresponding to the next state
            # to the path
            path.append(self.states[next_s])
            # Update time and state for next iteration
            t +=1;
            s = next_s;
            if t%(period) == 0:
                print('Simulation at ', t//period, "%")
        print('Simulation done !')
        return path, V_s0, Q


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
    for i in range(len(path)):
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
            grid.get_celld()[(path[i].police_pos)].set_facecolor(RED)
            grid.get_celld()[(path[i].police_pos)].get_text().set_text('Police')

        if i > 0:
            if path[i].player_pos == path[i].police_pos:
                grid.get_celld()[(path[i].player_pos)].set_facecolor(LIGHT_RED)
                grid.get_celld()[(path[i].player_pos)].get_text().set_text('Player has been caught')
            elif maze[path[i].player_pos[0], path[i].player_pos[1]] == 1:
                grid.get_celld()[(path[i].player_pos)].set_facecolor(ORANGE)
                grid.get_celld()[(path[i].player_pos)].get_text().set_text('Player is robbing')

        display.display(fig)
        display.clear_output(wait=True)
        time.sleep(1)

# %%
