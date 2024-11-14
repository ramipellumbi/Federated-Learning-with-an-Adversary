import torch.nn as nn


class MnistCNN(nn.Module):
    """
    Standard best performing MNIST model
    """

    def __init__(self, dropout=True):
        super(MnistCNN, self).__init__()
        self._should_dropout = dropout
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.dropout(x)
        x = x.view(-1, 9216)
        x = self.relu(self.fc1(x))
        if self._should_dropout:
            x = self.dropout(x)
        x = self.fc2(x)

        return x

    def name(self):
        return "Model"
