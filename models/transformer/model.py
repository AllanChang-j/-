from __future__ import annotations

import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TimeSeriesTransformer(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_outputs: int,
        d_model: int = 64,
        n_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.2,
        task: str = "binary",
    ):
        super().__init__()
        self.input_projection = nn.Linear(n_features, d_model)
        self.position = PositionalEncoding(d_model=d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, max(16, d_model // 2)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(16, d_model // 2), n_outputs),
        )
        self.task = task

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.input_projection(x)
        encoded = self.position(encoded)
        encoded = self.encoder(encoded)
        pooled = encoded[:, -1, :]
        return self.head(pooled)


def build_transformer(config: dict, n_features: int, n_outputs: int, task: str) -> nn.Module:
    return TimeSeriesTransformer(
        n_features=n_features,
        n_outputs=n_outputs,
        d_model=int(config.get("d_model", 64)),
        n_heads=int(config.get("n_heads", 4)),
        num_layers=int(config.get("num_layers", 2)),
        dim_feedforward=int(config.get("dim_feedforward", 128)),
        dropout=float(config.get("dropout", 0.2)),
        task=task,
    )
