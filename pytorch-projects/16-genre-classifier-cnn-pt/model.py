import torch.nn as nn


class Network(nn.Module):

    def __init__(self, dropout_rate):
        super().__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(32),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(32),
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(2, 2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(32),
        )

        self.dense_block = nn.Sequential(
            nn.Linear(in_features=480, out_features=64),
            nn.Dropout(dropout_rate),
            nn.Linear(in_features=64, out_features=10),
        )

    def forward(self, x):
        out = self.conv_block1(x)
        out = self.conv_block2(out)
        out = self.conv_block3(out)
        out = out.view(out.size(0), -1)  # Flatten
        out = self.dense_block(out)
        return out
