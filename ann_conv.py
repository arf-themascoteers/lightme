import torch.nn as nn


class ANNConv(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.verbose = False
        self.TEST = False
        self.device = device
        self.conv1 = nn.Conv2d(12,36,3, groups=12)
        self.relu1 = nn.LeakyReLU()
        self.flatten = nn.Flatten()

        self.linear1 = nn.Sequential(
            nn.Linear(36, 1),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = x.reshape(x.shape[0],1,-1)
        x = self.flatten(x)
        x = self.linear1(x)
        return x
