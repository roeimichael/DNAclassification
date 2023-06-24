import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

class ComplexCNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(ComplexCNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv1d(1, hidden_size, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv1d(hidden_size * 2, hidden_size * 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # Calculate the size of the input to the first fully connected layer
        fc_input_size = hidden_size * 4 * (input_size // (2 ** 3))  # 2^3 because of 3 maxpooling layers

        self.fc1 = nn.Linear(fc_input_size, hidden_size * 4)
        self.fc2 = nn.Linear(hidden_size * 4, num_classes)

    def forward(self, x):
        out = checkpoint(self.layer1, x)
        out = checkpoint(self.layer2, out)
        out = checkpoint(self.layer3, out)

        out = out.view(out.size(0), -1)  # Flattening for the fully connected layer

        out = self.fc1(out)
        out = self.fc2(out)

        return out
