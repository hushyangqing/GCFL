import numpy as np

# Pytorch libraries
import torch.nn as nn
import torch.nn.functional as F

class NaiveMLP(nn.Module):
    def __init__(self, dim_in,  dim_out, dim_hidden=128):
        super(NaiveMLP, self).__init__()
        self.predictor = nn.Sequential(
                            nn.Linear(dim_in, dim_hidden),
                            nn.ReLU(),
                            nn.Linear(dim_hidden, dim_out)
                         )

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        return self.predictor(x)


class NaiveCNN(nn.Module):
    def __init__(self, channels=1, dim_out=10, **kwargs):
        super(NaiveCNN, self).__init__()
        self.channels = channels
        if "dim_in" in kwargs:
            self.input_size = int(np.sqrt(kwargs["dim_in"]/channels))
        else:
            self.input_size = 28
        
        kernel_size = 3
        self.fc_input_size = (((((self.input_size - kernel_size)/1 + 1) - kernel_size)/1 + 1) - kernel_size)/2 + 1
        self.fc_input_size = int(self.fc_input_size)**2 * 20

        self.predictor = nn.Sequential(
                    nn.Conv2d(channels, 10, kernel_size=kernel_size),
                    nn.ReLU(),
                    nn.Conv2d(10, 20, kernel_size=kernel_size),
                    nn.MaxPool2d(kernel_size=kernel_size, stride=2),
                    nn.ReLU(),
                    )
        self.fc1 = nn.Linear(self.fc_input_size, 50)
        self.fc2 = nn.Linear(50, dim_out)

    def forward(self, x):
        x = x.view(x.shape[0], self.channels, self.input_size, self.input_size)
        x = self.predictor(x)
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x