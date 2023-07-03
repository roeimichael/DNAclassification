import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

class ComplexCNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(ComplexCNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv1d(1, hidden_size, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size * 2, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(hidden_size * 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(hidden_size * 2, hidden_size * 4, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(hidden_size * 4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv1d(hidden_size * 4, hidden_size * 8, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(hidden_size * 8),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.layer5 = nn.Sequential(
            nn.Conv1d(hidden_size * 8, hidden_size * 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(hidden_size * 16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        # Calculate the size of the input to the first fully connected layer
        fc_input_size = hidden_size * 16 * (input_size // (2 ** 5))  # 2^5 because of 5 maxpooling layers

        self.fc1 = nn.Linear(fc_input_size, hidden_size * 8)
        self.dropout = nn.Dropout(p=0.5)  # Adding dropout layer
        self.fc2 = nn.Linear(hidden_size * 8, num_classes)

    def forward(self, x):
        out = checkpoint(self.layer1, x)
        out = checkpoint(self.layer2, out)
        out = checkpoint(self.layer3, out)
        out = checkpoint(self.layer4, out)
        out = checkpoint(self.layer5, out)

        out = out.view(out.size(0), -1)  # Flattening for the fully connected layer

        out = self.fc1(out)
        out = self.dropout(out)  # Applying dropout
        out = self.fc2(out)

        return out
