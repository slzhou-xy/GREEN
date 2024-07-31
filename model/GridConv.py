import torch.nn as nn

from config.config import Config


class GridConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.projector = nn.Sequential(
            nn.Linear(out_channels, out_channels * 8),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels * 8, out_channels * 4),
        )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = out.squeeze(0).permute(1, 2, 0).reshape(-1, Config.grid_out_channel)
        out = self.projector(out)
        return out
