import torch.nn as nn


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
