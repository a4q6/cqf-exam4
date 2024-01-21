import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(
        self,
        n_channel: int,
        width: int,
    ):
        super(SimpleCNN, self).__init__()
        self.C = n_channel
        self.W = width
        # [B, C, H, W] -> Convolution
        self.conv1 = nn.Conv2d(n_channel, 16, kernel_size=(3, 1), padding="same")
        self.conv2 = nn.Conv2d(16, 16, kernel_size=(1, width))
        self.conv3 = nn.Conv2d(16, 16, kernel_size=(3, 1), padding="same")
        # Fully connection
        self.fc = nn.Linear(16, 1)
        # Binary classification output(0<.<1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        # x = x.view(-1, 12)  # flatten
        x = torch.mean(x, dim=(2, 3))  # GlobalAvgPool[C, H, W] -> [C, 1]
        x = self.fc(x)
        x = self.sigmoid(x)
        return x
