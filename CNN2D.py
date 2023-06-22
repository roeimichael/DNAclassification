# model.py
import torch
from torch import nn

class DNA_CNN(nn.Module):
    def __init__(self):
        super(DNA_CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(4, 1))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 1))
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 10)
        self.max_pool = nn.MaxPool2d((2, 1))
        self.adaptive_pool = nn.AdaptiveMaxPool2d((1, 4))
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
