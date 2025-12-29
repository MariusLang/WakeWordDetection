from torch import nn


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


class TemporalBlock2D(nn.Module):
    """
    Temporal convolution block using 2D convolutions for Hailo compatibility.
    Operates on (B, C, 1, T) tensors with kernel (1, k) for temporal processing.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        padding = (0, (kernel_size - 1) * dilation // 2)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, kernel_size),
                      padding=padding, dilation=(1, dilation), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(1, kernel_size),
                      padding=padding, dilation=(1, dilation), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.residual(x)


class CRNN_TemporalConv(nn.Module):
    """
    Hailo-compatible CRNN model for wake word detection.
    Uses dilated temporal convolutions (as 2D ops) instead of bidirectional GRU.
    All operations are 2D to ensure full Hailo compatibility.

    Note: Designed for fixed input shape (1, 40, 100).
    """

    def __init__(self, input_shape=None, num_classes=2):
        super().__init__()

        # Same CNN backbone as CRNN_with_MBConv
        self.cnn = nn.Sequential(
            MBConvBlock(1, 16, expansion=4),
            MBConvBlock(16, 32, expansion=4, stride=2),
            MBConvBlock(32, 64, expansion=4, stride=2)
        )

        # Reduce frequency dimension to 1 using conv instead of mean
        self.freq_reduce = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(10, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Temporal convolutions with increasing dilation (2D ops with kernel height=1)
        # Reduced channels to keep parameter count reasonable
        self.temporal = nn.Sequential(
            TemporalBlock2D(64, 96, kernel_size=3, dilation=1),
            TemporalBlock2D(96, 128, kernel_size=3, dilation=2),
            TemporalBlock2D(128, 128, kernel_size=3, dilation=4),
        )

        # Depthwise separable conv to aggregate temporal dimension (efficient alternative)
        self.final_conv = nn.Sequential(
            # Depthwise: process each channel independently across time
            nn.Conv2d(128, 128, kernel_size=(1, 25), groups=128, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # Pointwise: mix channels
            nn.Conv2d(128, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # Classifier
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # CNN feature extraction: (B, 1, F=40, T=100) -> (B, 64, F'=10, T'=25)
        x = self.cnn(x)

        # Reduce frequency to 1: (B, 64, 10, 25) -> (B, 64, 1, 25)
        x = self.freq_reduce(x)

        # Temporal convolutions: (B, 64, 1, 25) -> (B, 128, 1, 25)
        x = self.temporal(x)

        # Aggregate temporal dimension: (B, 128, 1, 25) -> (B, 256, 1, 1)
        x = self.final_conv(x)

        # Classification: (B, 256, 1, 1) -> (B, num_classes)
        out = self.fc(x)
        return out
