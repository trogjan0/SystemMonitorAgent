from __future__ import annotations

import json
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from classifier_models import build_classifier
from config import Config, PROJECT_ROOT, ensure_dirs
from data_generation import generate_synthetic_dataset
from forecast_models import build_forecaster
from metrics_schema import class_id_to_name, recommended_status
from utils import load_json, set_seed


class SystemMonitorAgent:
    def __init__(self, node_id: str = "node-1", device: str | None = None, config: Config | None = None) -> None:
        self.config = config or Config()
        self.node_id = node_id
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.buffer: deque[np.ndarray] = deque(maxlen=self.config.window_size)

        self.best_forecaster = load_json(self.config.models_dir / "best_forecaster.json")
        self.best_current_classifier = load_json(self.config.models_dir / "best_current_classifier.json")
        self.best_future_classifier = load_json(self.config.models_dir / "best_future_classifier.json")

        self.forecaster_scaler = joblib.load(self._resolve_path(self.best_forecaster["scaler_path"]))
        self.classifier_scaler = joblib.load(self._resolve_path(self.best_current_classifier["scaler_path"]))

        self.forecaster = build_forecaster(self.best_forecaster["model_name"], self.config).to(self.device)
        self.current_classifier = build_classifier(
            self.best_current_classifier["model_name"],
            self.config,
            input_time=int(self.best_current_classifier.get("input_time", self.config.window_size)),
        ).to(self.device)
        self.future_classifier = build_classifier(
            self.best_future_classifier["model_name"],
            self.config,
            input_time=int(self.best_future_classifier.get("input_time", self.config.forecast_horizon)),
        ).to(self.device)

        self.forecaster.load_state_dict(self._torch_load(self._resolve_path(self.best_forecaster["model_path"]))["state_dict"])
        self.current_classifier.load_state_dict(
            self._torch_load(self._resolve_path(self.best_current_classifier["model_path"]))["state_dict"]
        )
        self.future_classifier.load_state_dict(
            self._torch_load(self._resolve_path(self.best_future_classifier["model_path"]))["state_dict"]
        )

        self.forecaster.eval()
        self.current_classifier.eval()
        self.future_classifier.eval()

    def _resolve_path(self, path_value: str | Path) -> Path:
        path = Path(path_value)
        if path.is_absolute():
            return path
        return PROJECT_ROOT / path

    def _torch_load(self, path: Path) -> dict:
        try:
            return torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            return torch.load(path, map_location=self.device)

    def add_metrics(self, metrics_dict: dict[str, float]) -> dict | None:
        missing = [feature for feature in self.config.feature_names if feature not in metrics_dict]
        if missing:
            raise ValueError(f"Missing metrics: {missing}")

        record = np.asarray([float(metrics_dict[feature]) for feature in self.config.feature_names], dtype=np.float32)
        self.buffer.append(record)
        if len(self.buffer) < self.config.window_size:
            return None

        current_window = np.stack(self.buffer).astype(np.float32)
        with torch.no_grad():
            current_class_id = self._classify_current(current_window)
            predicted_future_metrics = self._forecast_future(current_window)
            future_class_id = self._classify_future(predicted_future_metrics)

        current_class_name = class_id_to_name(current_class_id)
        future_class_name = class_id_to_name(future_class_id)
        summary = self._predicted_metrics_summary(predicted_future_metrics)
        bottleneck = self._detect_bottleneck(summary)

        return {
            "agent": "SystemMonitorAgent",
            "architecture_mode": "separated",
            "node_id": self.node_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "current_load_class": current_class_name,
            "future_load_class": future_class_name,
            "predicted_metrics_summary": summary,
            "bottleneck": bottleneck,
            "recommended_status": recommended_status(current_class_name, future_class_name),
        }

    def _classify_current(self, current_window: np.ndarray) -> int:
        scaled = self._transform_metrics(self.classifier_scaler, current_window).reshape(
            1,
            self.config.window_size,
            self.config.num_features,
        )
        tensor = torch.tensor(scaled, dtype=torch.float32, device=self.device)
        logits = self.current_classifier(tensor)
        return int(torch.argmax(logits, dim=1).item())

    def _forecast_future(self, current_window: np.ndarray) -> np.ndarray:
        scaled = self._transform_metrics(self.forecaster_scaler, current_window).reshape(
            1,
            self.config.window_size,
            self.config.num_features,
        )
        tensor = torch.tensor(scaled, dtype=torch.float32, device=self.device)
        predicted_scaled = self.forecaster(tensor).cpu().numpy()[0]
        predicted = self.forecaster_scaler.inverse_transform(predicted_scaled)
        return self._clip_metrics(predicted)

    def _classify_future(self, predicted_future_metrics: np.ndarray) -> int:
        scaled = self._transform_metrics(self.classifier_scaler, predicted_future_metrics).reshape(
            1,
            self.config.forecast_horizon,
            self.config.num_features,
        )
        tensor = torch.tensor(scaled, dtype=torch.float32, device=self.device)
        logits = self.future_classifier(tensor)
        return int(torch.argmax(logits, dim=1).item())

    def _transform_metrics(self, scaler, metrics: np.ndarray) -> np.ndarray:
        frame = pd.DataFrame(metrics.reshape(-1, self.config.num_features), columns=self.config.feature_names)
        return scaler.transform(frame).reshape(metrics.shape)

    def _clip_metrics(self, metrics: np.ndarray) -> np.ndarray:
        clipped = metrics.copy()
        index = {name: idx for idx, name in enumerate(self.config.feature_names)}
        for feature in ["cpu_percent", "mem_percent", "swap_percent"]:
            clipped[:, index[feature]] = np.clip(clipped[:, index[feature]], 0, 100)
        for feature in ["psi_cpu", "psi_mem", "psi_io"]:
            clipped[:, index[feature]] = np.clip(clipped[:, index[feature]], 0, 1)
        for feature in ["io_read_mb", "io_write_mb", "net_in_mb", "net_out_mb", "process_count", "blocked_processes"]:
            clipped[:, index[feature]] = np.clip(clipped[:, index[feature]], 0, None)
        return clipped

    def _predicted_metrics_summary(self, predicted_future_metrics: np.ndarray) -> dict[str, float]:
        idx = {name: position for position, name in enumerate(self.config.feature_names)}

        def max_value(feature: str) -> float:
            return round(float(np.max(predicted_future_metrics[:, idx[feature]])), 4)

        return {
            "max_cpu_percent": max_value("cpu_percent"),
            "max_mem_percent": max_value("mem_percent"),
            "max_swap_percent": max_value("swap_percent"),
            "max_psi_cpu": max_value("psi_cpu"),
            "max_psi_mem": max_value("psi_mem"),
            "max_psi_io": max_value("psi_io"),
            "max_io_write_mb": max_value("io_write_mb"),
            "max_blocked_processes": max_value("blocked_processes"),
        }

    def _detect_bottleneck(self, summary: dict[str, float]) -> str:
        conditions = {
            "CPU": summary["max_cpu_percent"] >= 85 or summary["max_psi_cpu"] >= 0.60,
            "MEMORY": summary["max_mem_percent"] >= 85 or summary["max_psi_mem"] >= 0.60,
            "IO": summary["max_psi_io"] >= 0.60 or summary["max_io_write_mb"] >= 100,
        }
        active = [name for name, is_active in conditions.items() if is_active]
        if len(active) > 1:
            return "MIXED"
        if len(active) == 1:
            return active[0]
        return "NONE"


def main() -> None:
    config = Config()
    ensure_dirs(config)
    required = [
        config.models_dir / "best_forecaster.json",
        config.models_dir / "best_current_classifier.json",
        config.models_dir / "best_future_classifier.json",
        config.models_dir / "forecaster_scaler.pkl",
        config.models_dir / "classifier_scaler.pkl",
    ]
    if any(not path.exists() for path in required):
        print("Сначала запустите train_forecasters.py и train_classifiers.py")
        return

    set_seed(config.seed)
    if not config.raw_data_path.exists():
        generate_synthetic_dataset(config)

    agent = SystemMonitorAgent(config=config)
    df = pd.read_csv(config.raw_data_path).sort_values(["episode_id", "timestep"])
    for _, row in df.iterrows():
        metrics = {feature: float(row[feature]) for feature in config.feature_names}
        result = agent.add_metrics(metrics)
        if result is not None:
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return


if __name__ == "__main__":
    main()
