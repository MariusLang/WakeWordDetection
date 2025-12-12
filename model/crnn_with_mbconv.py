from torch import nn
import torch


class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion=4, kernel_size=3, stride=1):
        super().__init__()
        hidden_dim = in_channels * expansion
        padding = kernel_size // 2

        self.use_residual = (stride == 1 and in_channels == out_channels)

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=padding, stride=stride),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        out = self.block(x)
        if self.use_residual:
            out += x
        return out


class CRNN_with_MBConv(nn.Module):
    def __init__(self, input_shape=None, num_classes=2):
        super().__init__()
        self.cnn = nn.Sequential(
            MBConvBlock(1, 16, expansion=4),
            MBConvBlock(16, 32, expansion=4, stride=2),
            MBConvBlock(32, 64, expansion=4, stride=2)
        )
        self.rnn = nn.GRU(64, 256, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.mean(dim=2)
        x = x.permute(0, 2, 1)
        _, h = self.rnn(x)
        h = h.permute(1, 0, 2)
        h = h.contiguous().view(x.size(0), -1)
        out = self.fc(h)
        return out