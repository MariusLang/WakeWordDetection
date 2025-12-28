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


class CRNN_Own_GRU(nn.Module):
    """
    Hailo-compatible CRNN model for wake word detection.
    Uses temporal convolutions instead of RNN to ensure Hailo compatibility.
    """

    def __init__(self, input_shape=None, num_classes=2):
        super().__init__()

        self.cnn = nn.Sequential(
            MBConvBlock(1, 16, expansion=4),
            MBConvBlock(16, 32, expansion=4, stride=2),
            MBConvBlock(32, 64, expansion=4, stride=2)
        )

        # Unidirectional GRUs
        self.gru_fwd = nn.GRU(
            input_size=64,
            hidden_size=256,
            batch_first=True
        )

        self.gru_bwd = nn.GRU(
            input_size=64,
            hidden_size=256,
            batch_first=True
        )

        self.fc = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # CNN-Feature extraction
        x = self.cnn(x)  # (B, C=64, F, T)
        x = x.mean(dim=2)  # (B, 64, T)
        x = x.permute(0, 2, 1)  # (B, T, 64)

        # Forward-GRU
        _, h_fwd = self.gru_fwd(x)  # (1, B, 256)

        # Backward-GRU
        x_rev = torch.flip(x, dims=[1])
        _, h_bwd = self.gru_bwd(x_rev)  # (1, B, 256)

        # Summarize Hidden States
        h_fwd = h_fwd.squeeze(0)  # (B, 256)
        h_bwd = h_bwd.squeeze(0)  # (B, 256)
        h = torch.cat([h_fwd, h_bwd], dim=1)  # (B, 512)

        out = self.fc(h)
        return out
