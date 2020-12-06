import torch.nn as nn
import torch


class DqnNetwork(nn.Module):

    def __init__(self, size_of_layers):
        super().__init__()

        self.nb_h_layer = len(size_of_layers) - 2
        self.input_layer = nn.Linear(size_of_layers[0], size_of_layers[1])
        self.input_layer_activation = nn.ReLU()

        self.hidden_layer = []
        self.hidden_layer_activation = []
        for i in range(1, self.nb_h_layer):
            self.hidden_layer.append(nn.Linear(size_of_layers[i], size_of_layers[i+1]))
            self.hidden_layer_activation.append(nn.ReLU())
        
        self.output_layer = nn.Linear(size_of_layers[-2], size_of_layers[-1])
    
    def forward(self, x):

        l1 = self.input_layer(x)
        l1 = self.input_layer_activation(l1)

        for i in range(self.nb_h_layer-1):
            l1 = self.hidden_layer[i](l1)
            l1 = self.hidden_layer_activation[i](l1)
        
        out = self.output_layer(l1)
        return out

class DuelingDqnNetwork(nn.Module):

    def __init__(self, size_of_layers):
        super().__init__()

        self.nb_h_layer = len(size_of_layers) - 3
        self.input_layer = nn.Linear(size_of_layers[0], size_of_layers[1])
        self.input_layer_activation = nn.ReLU()

        self.hidden_layer = []
        self.hidden_layer_activation = []
        for i in range(1, self.nb_h_layer):
            self.hidden_layer.append(nn.Linear(size_of_layers[i], size_of_layers[i+1]))
            self.hidden_layer_activation.append(nn.ReLU())
        
        self.V_hidden_layer = nn.Linear(size_of_layers[-3], size_of_layers[-2])
        self.V_hidden_layer_activation = nn.ReLU()
        self.A_hidden_layer = nn.Linear(size_of_layers[-3], size_of_layers[-2])
        self.A_hidden_layer_activation = nn.ReLU()
        self.V_layer = nn.Linear(size_of_layers[-2], 1)
        self.A_layer = nn.Linear(size_of_layers[-2], size_of_layers[-1])
    
    def forward(self, x):

        l1 = self.input_layer(x)
        l1 = self.input_layer_activation(l1)

        for i in range(self.nb_h_layer-1):
            l1 = self.hidden_layer[i](l1)
            l1 = self.hidden_layer_activation[i](l1)
        
        V_hidden_value = self.V_hidden_layer(l1)
        V_hidden_value = self.V_hidden_layer_activation(V_hidden_value)

        V_value = self.V_layer(V_hidden_value)

        A_hidden_value = self.A_hidden_layer(l1)
        A_hidden_value = self.A_hidden_layer_activation(A_hidden_value)
        
        A_values = self.A_layer(A_hidden_value)
        
        avg_A_values = torch.mean(A_values, dim=1, keepdim=True)

        out = V_value + A_values - avg_A_values
        return out