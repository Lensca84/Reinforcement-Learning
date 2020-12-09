import torch.nn as nn
import torch
import numpy as np

class PpoActorNetwork(nn.Module):

    def __init__(self, n_actions, dim_state, seed):
        super().__init__()
        torch.manual_seed(seed)
        np.random.seed(seed)

        first_layer_size = 400
        second_layer_size = 200

        self.input_layer_Ls = nn.Linear(dim_state, first_layer_size)
        self.input_layer_Ls_activation = nn.ReLU()

        self.mu_head_hidden_layer = nn.Linear(first_layer_size, second_layer_size)
        self.mu_head_hidden_layer_activation = nn.ReLU()

        self.mu_head_output_layer = nn.Linear(second_layer_size, n_actions)
        self.mu_head_output_layer_activation = nn.Tanh()
        
        self.sigma_head_hidden_layer = nn.Linear(first_layer_size, second_layer_size)
        self.sigma_head_hidden_layer_activation = nn.ReLU()

        self.sigma_head_output_layer = nn.Linear(second_layer_size, n_actions)
        self.sigma_head_output_layer_activation = nn.Sigmoid()
    
    def forward(self, state):

        Ls = self.input_layer_Ls(state)
        Ls = self.input_layer_Ls_activation(Ls)

        l_mu = self.mu_head_hidden_layer(Ls)
        l_mu = self.mu_head_hidden_layer_activation(l_mu)

        out_mu = self.mu_head_output_layer(l_mu)
        out_mu = self.mu_head_output_layer_activation(out_mu)

        l_sigma = self.sigma_head_hidden_layer(Ls)
        l_sigma = self.sigma_head_output_layer_activation(l_sigma)

        out_sigma = self.sigma_head_output_layer(l_sigma)
        out_sigma = self.sigma_head_output_layer_activation(out_sigma)

        return out_mu, out_sigma

class PpoCriticNetwork(nn.Module):

    def __init__(self, n_actions, dim_state, seed):
        super().__init__()
        torch.manual_seed(seed)
        np.random.seed(seed)

        first_layer_size = 400
        second_layer_size = 200

        self.input_layer = nn.Linear(dim_state, first_layer_size)
        self.input_layer_activation = nn.ReLU()

        self.hidden_layer = nn.Linear(first_layer_size, second_layer_size)
        self.hidden_layer_activation = nn.ReLU()
        
        self.output_layer = nn.Linear(second_layer_size, 1)
    
    def forward(self, state):

        l1 = self.input_layer(state)
        l1 = self.input_layer_activation(l1)

        l2 = self.hidden_layer(l1)
        l2 = self.hidden_layer_activation(l2)

        out = self.output_layer(l2)
        
        return out