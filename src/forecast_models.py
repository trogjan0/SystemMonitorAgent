from __future__ import annotations

import torch
from torch import nn


class MLPForecaster(nn.Module):
    def __init__(self, window_size: int, forecast_horizon: int, num_features: int) -> None:
        super().__init__()
        self.forecast_horizon = forecast_horizon
        self.num_features = num_features
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(window_size * num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(128, forecast_horizon * num_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.net(x)
        return output.view(x.size(0), self.forecast_horizon, self.num_features)


class CNN1DForecaster(nn.Module):
    def __init__(self, window_size: int, forecast_horizon: int, num_features: int) -> None:
        super().__init__()
        self.forecast_horizon = forecast_horizon
        self.num_features = num_features
        self.encoder = nn.Sequential(
            nn.Conv1d(num_features, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 96, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(96, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.decoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(128, forecast_horizon * num_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        output = self.decoder(self.encoder(x))
        return output.view(x.size(0), self.forecast_horizon, self.num_features)


class GRUForecaster(nn.Module):
    def __init__(self, window_size: int, forecast_horizon: int, num_features: int) -> None:
        super().__init__()
        self.forecast_horizon = forecast_horizon
        self.num_features = num_features
        self.gru = nn.GRU(num_features, 96, num_layers=2, batch_first=True, dropout=0.10)
        self.decoder = nn.Sequential(
            nn.Linear(96, 128),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(128, forecast_horizon * num_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(x)
        output = self.decoder(hidden[-1])
        return output.view(x.size(0), self.forecast_horizon, self.num_features)


class LSTMForecaster(nn.Module):
    def __init__(self, window_size: int, forecast_horizon: int, num_features: int) -> None:
        super().__init__()
        self.forecast_horizon = forecast_horizon
        self.num_features = num_features
        self.lstm = nn.LSTM(num_features, 96, num_layers=2, batch_first=True, dropout=0.10)
        self.decoder = nn.Sequential(
            nn.Linear(96, 128),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(128, forecast_horizon * num_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(x)
        output = self.decoder(hidden[-1])
        return output.view(x.size(0), self.forecast_horizon, self.num_features)


def build_forecaster(model_name: str, config) -> nn.Module:
    builders = {
        "MLPForecaster": MLPForecaster,
        "CNN1DForecaster": CNN1DForecaster,
        "GRUForecaster": GRUForecaster,
        "LSTMForecaster": LSTMForecaster,
    }
    if model_name not in builders:
        raise ValueError(f"Unknown forecaster model: {model_name}")
    return builders[model_name](
        window_size=config.window_size,
        forecast_horizon=config.forecast_horizon,
        num_features=config.num_features,
    )
