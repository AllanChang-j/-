from __future__ import annotations

import torch
from torch import nn


class AttentionPooling(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights = torch.softmax(self.score(sequence).squeeze(-1), dim=1)
        context = torch.sum(sequence * weights.unsqueeze(-1), dim=1)
        return context, weights


class LSTMModel(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_outputs: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        bidirectional: bool = False,
        attention: bool = True,
        dropout: float = 0.25,
        task: str = "binary",
    ):
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=lstm_dropout,
        )
        output_size = hidden_size * (2 if bidirectional else 1)
        self.attention = AttentionPooling(output_size) if attention else None
        self.head = nn.Sequential(
            nn.LayerNorm(output_size),
            nn.Dropout(dropout),
            nn.Linear(output_size, max(16, output_size // 2)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(16, output_size // 2), n_outputs),
        )
        self.task = task
        self.last_attention_weights: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.lstm(x)
        if self.attention is not None:
            pooled, weights = self.attention(sequence)
            self.last_attention_weights = weights.detach()
        else:
            pooled = sequence[:, -1, :]
            self.last_attention_weights = None
        return self.head(pooled)


def build_lstm(config: dict, n_features: int, n_outputs: int, task: str) -> nn.Module:
    return LSTMModel(
        n_features=n_features,
        n_outputs=n_outputs,
        hidden_size=int(config.get("hidden_size", 64)),
        num_layers=int(config.get("num_layers", 1)),
        bidirectional=bool(config.get("bidirectional", False)),
        attention=bool(config.get("attention", True)),
        dropout=float(config.get("dropout", 0.25)),
        task=task,
    )

