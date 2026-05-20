from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class CurrentClassifierDataset(Dataset):
    def __init__(self, df: pd.DataFrame, feature_names: list[str], window_size: int) -> None:
        self.feature_names = feature_names
        self.window_size = window_size
        self.episodes: list[dict[str, np.ndarray | int]] = []
        self.samples: list[tuple[int, int]] = []

        for episode_id, episode in df.groupby("episode_id", sort=False):
            episode = episode.sort_values("timestep").reset_index(drop=True)
            features = episode[feature_names].to_numpy(dtype=np.float32)
            labels = episode["load_class"].to_numpy(dtype=np.int64)
            episode_idx = len(self.episodes)
            self.episodes.append(
                {
                    "episode_id": int(episode_id),
                    "features": features,
                    "labels": labels,
                }
            )
            for end_pos in range(window_size - 1, len(episode)):
                self.samples.append((episode_idx, end_pos))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        episode_idx, end_pos = self.samples[index]
        episode = self.episodes[episode_idx]
        features = episode["features"]
        labels = episode["labels"]
        assert isinstance(features, np.ndarray)
        assert isinstance(labels, np.ndarray)

        x_window = features[end_pos - self.window_size + 1 : end_pos + 1]
        y_current_class = labels[end_pos]
        return torch.tensor(x_window, dtype=torch.float32), torch.tensor(y_current_class, dtype=torch.long)


class FutureClassifierDataset(Dataset):
    def __init__(self, df: pd.DataFrame, feature_names: list[str], forecast_horizon: int) -> None:
        self.feature_names = feature_names
        self.forecast_horizon = forecast_horizon
        self.episodes: list[dict[str, np.ndarray | int]] = []
        self.samples: list[tuple[int, int]] = []

        for episode_id, episode in df.groupby("episode_id", sort=False):
            episode = episode.sort_values("timestep").reset_index(drop=True)
            features = episode[feature_names].to_numpy(dtype=np.float32)
            labels = episode["future_load_class"].to_numpy(dtype=np.int64)
            episode_idx = len(self.episodes)
            self.episodes.append(
                {
                    "episode_id": int(episode_id),
                    "features": features,
                    "labels": labels,
                }
            )
            for current_pos in range(0, len(episode) - forecast_horizon):
                self.samples.append((episode_idx, current_pos))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        episode_idx, current_pos = self.samples[index]
        episode = self.episodes[episode_idx]
        features = episode["features"]
        labels = episode["labels"]
        assert isinstance(features, np.ndarray)
        assert isinstance(labels, np.ndarray)

        x_future_window = features[current_pos + 1 : current_pos + self.forecast_horizon + 1]
        y_future_class = labels[current_pos]
        return torch.tensor(x_future_window, dtype=torch.float32), torch.tensor(y_future_class, dtype=torch.long)
