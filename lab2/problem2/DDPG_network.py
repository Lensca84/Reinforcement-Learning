import torch.nn as nn
import torch
import numpy as np

class DdpgActorNetwork(nn.Module):

    def __init__(self, n_actions, dim_state, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)

        first_layer_size = 400
        second_layer_size = 200

        self.input_layer = nn.Linear(dim_state, first_layer_size)
        self.input_layer_activation = nn.ReLU()

        self.hidden_layer = nn.Linear(first_layer_size, second_layer_size)
        self.hidden_layer_activation = nn.ReLU()
        
        self.output_layer = nn.Linear(second_layer_size, n_actions)
        self.output_layer_activation = nn.Tanh()
    
    def forward(self, state):

        l1 = self.input_layer(state)
        l1 = self.input_layer_activation(l1)

        l2 = self.hidden_layer(l1)
        l2 = self.hidden_layer_activation(l2)

        out = self.output_layer(l2)
        out = self.output_layer_activation(out)

        return out

class DdpgCriticNetwork(nn.Module):

    def __init__(self, n_actions, dim_state, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)

        first_layer_size = 400
        input_second_layer = first_layer_size + n_actions
        second_layer_size = 200

        self.input_layer = nn.Linear(dim_state, first_layer_size)
        self.input_layer_activation = nn.ReLU()

        self.hidden_layer = nn.Linear(input_second_layer, second_layer_size)
        self.hidden_layer_activation = nn.ReLU()
        
        self.output_layer = nn.Linear(second_layer_size, 1)
    
    def forward(self, state, action):

        l1 = self.input_layer(state)
        l1 = self.input_layer_activation(l1)

        concat_l1_action = torch.cat((l1, action), 1)
        l2 = self.hidden_layer(concat_l1_action)
        l2 = self.hidden_layer_activation(l2)

        out = self.output_layer(l2)
        
        return out