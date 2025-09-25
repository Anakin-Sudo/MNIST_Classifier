import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Fully connected (linear) layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # after pooling twice
        self.fc2 = nn.Linear(128, 10)  # 10 digits

        # Pooling layer (reduce size by factor of 2)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        # First conv block: conv -> BN -> ReLU -> pool
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))

        # Second conv block
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))

        # Flatten
        x = x.view(-1, 64 * 7 * 7)

        # Fully connected layers
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)  # logits, no softmax (CrossEntropyLoss expects raw logits)

        return x

