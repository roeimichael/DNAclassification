import torch
from torch import nn

class DecentCNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(DecentCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, hidden_size, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size // 2, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear((hidden_size // 2) * (input_size // 4), num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = out.view(out.size(0), -1)  # Flattening for the fully connected layer
        out = self.fc1(out)
        return out

