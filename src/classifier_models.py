from __future__ import annotations

import torch
from torch import nn


class MLPClassifier(nn.Module):
    def __init__(self, input_time: int, num_features: int, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_time * num_features, 192),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(192, 96),
            nn.ReLU(),
            nn.Linear(96, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CNN1DClassifier(nn.Module):
    def __init__(self, input_time: int, num_features: int, num_classes: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(num_features, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 96, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        return self.head(self.encoder(x))


class GRUClassifier(nn.Module):
    def __init__(self, input_time: int, num_features: int, num_classes: int) -> None:
        super().__init__()
        self.gru = nn.GRU(num_features, 80, num_layers=2, batch_first=True, dropout=0.10)
        self.head = nn.Sequential(
            nn.Linear(80, 64),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(x)
        return self.head(hidden[-1])


class LSTMClassifier(nn.Module):
    def __init__(self, input_time: int, num_features: int, num_classes: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(num_features, 80, num_layers=2, batch_first=True, dropout=0.10)
        self.head = nn.Sequential(
            nn.Linear(80, 64),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(x)
        return self.head(hidden[-1])


def build_classifier(model_name: str, config, input_time: int) -> nn.Module:
    builders = {
        "MLPClassifier": MLPClassifier,
        "CNN1DClassifier": CNN1DClassifier,
        "GRUClassifier": GRUClassifier,
        "LSTMClassifier": LSTMClassifier,
    }
    if model_name not in builders:
        raise ValueError(f"Unknown classifier model: {model_name}")
    return builders[model_name](
        input_time=input_time,
        num_features=config.num_features,
        num_classes=config.num_classes,
    )
