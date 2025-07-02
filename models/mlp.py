import torch
import torch.nn as nn

class Swish(nn.Module):
    """Swish activation function: x * sigmoid(x)"""
    def forward(self, x):
        return x * torch.sigmoid(x)

def get_activation(name):
    if name == 'relu':
        return nn.ReLU()
    elif name == 'tanh':
        return nn.Tanh()
    elif name == 'sigmoid':
        return nn.Sigmoid()
    elif name == 'swish':
        return Swish()
    elif name == 'gelu':
        return nn.GELU()
    elif name == 'leaky_relu':
        return nn.LeakyReLU(0.1)
    else:
        raise ValueError(f"Unsupported activation: {name}")

class MLPRegressor(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, activation):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(hidden_dims)):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(get_activation(activation))
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
