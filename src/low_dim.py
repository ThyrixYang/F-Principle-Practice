import random

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import rfft
from torch.optim import lr_scheduler


class MLP(nn.Module):

    def __init__(self, input_size, output_size, hidden_sizes):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        for i, hidden_size in enumerate(hidden_sizes):
            if i == 0:
                self.layers.append(nn.Linear(input_size, hidden_size))
            else:
                self.layers.append(
                    nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
            # self.layers.append(nn.ReLU())
            self.layers.append(nn.Tanh())
            self.layers.append(nn.BatchNorm1d(hidden_sizes[i]))
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))

    def forward(self, x, _=None):
        for layer in self.layers[:-1]:
            x = layer(x)
        out = self.layers[-1](x)
        return out


def low_dim_data(num=100, period=5):
    """x in range[-10, 10]
    """
    def fn(x):
        res = np.zeros_like(x)
        sin_value = np.sin(x)
        cut = 0.6
        mask1 = sin_value > cut
        maskn1 = sin_value < -cut
        res[mask1] = 1
        res[maskn1] = -1
        return res
    x = np.linspace(-10, 10, num=num)
    _x = x / period * np.pi * 2
    y = fn(_x)
    x = np.reshape(x, (-1, 1))
    y = np.reshape(y, (-1, 1))
    return x, y

def main():
    x, y = low_dim_data(num=100)
    model = MLP(
        input_size=1,
        output_size=1,
        hidden_sizes=[16, 128, 16]
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheculer = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                           factor=0.5,
                                                           patience=50,
                                                           verbose=True)
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    for ep in range(15000):
        _y = model(x)
        loss = torch.mean((y - _y)**2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pred_y = _y.detach().numpy()
        
        #############################################
        #############################################
        
        """
        Plot spacial domain and Fourier domain.
        """
        
        #############################################
        #############################################
        
        if ep % 500 == 0:
            print(loss)


if __name__ == "__main__":
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    main()