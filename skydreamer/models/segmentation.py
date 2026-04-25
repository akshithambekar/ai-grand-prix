"""Lightweight segmentation U-Net."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from skydreamer.config import SegmentationConfig


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class SegmentationUNet(nn.Module):
    def __init__(self, config: SegmentationConfig) -> None:
        super().__init__()
        channels = config.base_channels
        self.output_size = config.output_size
        self.enc1 = DoubleConv(config.input_channels, channels)
        self.enc2 = DownBlock(channels, channels * 2)
        self.enc3 = DownBlock(channels * 2, channels * 4)
        self.bottleneck = DownBlock(channels * 4, channels * 8)
        self.up1 = UpBlock(channels * 8 + channels * 4, channels * 4)
        self.up2 = UpBlock(channels * 4 + channels * 2, channels * 2)
        self.up3 = UpBlock(channels * 2 + channels, channels)
        self.head = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, rgb: Tensor) -> Tensor:
        x1 = self.enc1(rgb)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.bottleneck(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.head(x)
        if logits.shape[-1] != self.output_size or logits.shape[-2] != self.output_size:
            logits = nn.functional.interpolate(
                logits,
                size=(self.output_size, self.output_size),
                mode="bilinear",
                align_corners=False,
            )
        return logits

