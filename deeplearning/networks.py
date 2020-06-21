# Pytorch libraries
import torch.nn as nn
import torch.functional as F

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
    def __init__(self, channels=1, classes=10):
        super(NaiveCNN, self).__init__()
        self.conv1 = nn.Conv2d(channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x