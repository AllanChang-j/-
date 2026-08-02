from __future__ import annotations

import torch
from torch import nn


class CNN1D(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_outputs: int,
        channels: list[int] | None = None,
        kernel_size: int = 3,
        dropout: float = 0.25,
        task: str = "binary",
    ):
        super().__init__()
        channels = channels or [32, 64]
        layers: list[nn.Module] = []
        in_channels = n_features
        for out_channels in channels:
            layers.extend(
                [
                    nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2),
                    nn.ReLU(),
                    nn.BatchNorm1d(out_channels),
                    nn.MaxPool1d(kernel_size=2, stride=2),
                    nn.Dropout(dropout),
                ]
            )
            in_channels = out_channels
        self.encoder = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, max(16, in_channels // 2)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(16, in_channels // 2), n_outputs),
        )
        self.task = task

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        return self.head(self.encoder(x))


class CandlestickImageCNN(nn.Module):
    def __init__(self, n_outputs: int, dropout: float = 0.25, task: str = "binary"):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.LayerNorm(32 * 4 * 4),
            nn.Linear(32 * 4 * 4, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_outputs),
        )
        self.task = task

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(1)
        return self.net(x)


def build_cnn(config: dict, n_features: int, n_outputs: int, task: str) -> nn.Module:
    variant = config.get("variant", "cnn1d")
    if variant == "candlestick_image":
        return CandlestickImageCNN(n_outputs=n_outputs, dropout=float(config.get("dropout", 0.25)), task=task)
    return CNN1D(
        n_features=n_features,
        n_outputs=n_outputs,
        channels=[int(value) for value in config.get("channels", [32, 64])],
        kernel_size=int(config.get("kernel_size", 3)),
        dropout=float(config.get("dropout", 0.25)),
        task=task,
    )

