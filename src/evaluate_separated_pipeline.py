from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score

from classifier_models import build_classifier
from config import Config, PROJECT_ROOT, ensure_dirs
from data_generation import generate_synthetic_dataset
from forecast_models import build_forecaster
from metrics_schema import LOAD_CLASS_NAMES
from utils import load_json, plot_confusion_matrix, save_confusion_matrix_csv, set_seed, split_episodes


LABEL_IDS = [0, 1, 2, 3]
LABEL_NAMES = [LOAD_CLASS_NAMES[idx] for idx in LABEL_IDS]


def _resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _torch_load(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _load_or_create_data(config: Config) -> pd.DataFrame:
    if not config.raw_data_path.exists():
        print("synthetic_metrics.csv not found, generating dataset...")
        return generate_synthetic_dataset(config)
    return pd.read_csv(config.raw_data_path)


def _collect_pipeline_samples(df: pd.DataFrame, config: Config) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    windows = []
    current_labels = []
    future_labels = []
    for _, episode in df.groupby("episode_id", sort=False):
        episode = episode.sort_values("timestep").reset_index(drop=True)
        features = episode[config.feature_names].to_numpy(dtype=np.float32)
        current = episode["load_class"].to_numpy(dtype=np.int64)
        future = episode["future_load_class"].to_numpy(dtype=np.int64)
        for end_pos in range(config.window_size - 1, len(episode) - config.forecast_horizon):
            windows.append(features[end_pos - config.window_size + 1 : end_pos + 1])
            current_labels.append(current[end_pos])
            future_labels.append(future[end_pos])

    return (
        np.stack(windows).astype(np.float32),
        np.asarray(current_labels, dtype=np.int64),
        np.asarray(future_labels, dtype=np.int64),
    )


def _metric_row(method: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "method": method,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", labels=LABEL_IDS, zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", labels=LABEL_IDS, zero_division=0),
        "high_recall": recall_score(y_true, y_pred, labels=[2], average="macro", zero_division=0),
        "critical_recall": recall_score(y_true, y_pred, labels=[3], average="macro", zero_division=0),
    }


def _transform_array(scaler, values: np.ndarray, feature_names: list[str]) -> np.ndarray:
    original_shape = values.shape
    frame = pd.DataFrame(values.reshape(-1, len(feature_names)), columns=feature_names)
    return scaler.transform(frame).reshape(original_shape)


def _predict_pipeline(
    windows: np.ndarray,
    forecaster: torch.nn.Module,
    future_classifier: torch.nn.Module,
    forecaster_scaler,
    classifier_scaler,
    config: Config,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    forecaster.eval()
    future_classifier.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            batch = windows[start : start + batch_size]
            batch_scaled = _transform_array(forecaster_scaler, batch, config.feature_names)
            batch_tensor = torch.tensor(batch_scaled, dtype=torch.float32, device=device)
            future_scaled = forecaster(batch_tensor).cpu().numpy()
            future_real = forecaster_scaler.inverse_transform(
                future_scaled.reshape(-1, config.num_features)
            ).reshape(future_scaled.shape)
            classifier_input = _transform_array(classifier_scaler, future_real, config.feature_names)
            logits = future_classifier(torch.tensor(classifier_input, dtype=torch.float32, device=device))
            predictions.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())
    return np.asarray(predictions, dtype=np.int64)


def main() -> None:
    config = Config()
    ensure_dirs(config)
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    required = [
        config.models_dir / "best_forecaster.json",
        config.models_dir / "best_future_classifier.json",
        config.models_dir / "forecaster_scaler.pkl",
        config.models_dir / "classifier_scaler.pkl",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing model artifacts: {', '.join(str(path) for path in missing)}")

    best_forecaster = load_json(config.models_dir / "best_forecaster.json")
    best_future_classifier = load_json(config.models_dir / "best_future_classifier.json")
    forecaster_scaler = joblib.load(_resolve_path(best_forecaster["scaler_path"]))
    classifier_scaler = joblib.load(_resolve_path(best_future_classifier["scaler_path"]))

    forecaster = build_forecaster(best_forecaster["model_name"], config).to(device)
    forecaster_checkpoint = _torch_load(_resolve_path(best_forecaster["model_path"]), device)
    forecaster.load_state_dict(forecaster_checkpoint["state_dict"])

    future_classifier = build_classifier(
        best_future_classifier["model_name"],
        config,
        input_time=config.forecast_horizon,
    ).to(device)
    classifier_checkpoint = _torch_load(_resolve_path(best_future_classifier["model_path"]), device)
    future_classifier.load_state_dict(classifier_checkpoint["state_dict"])

    df = _load_or_create_data(config)
    _, _, test_df = split_episodes(df, seed=config.seed)
    windows, current_labels, true_future_labels = _collect_pipeline_samples(test_df, config)

    separated_predictions = _predict_pipeline(
        windows,
        forecaster,
        future_classifier,
        forecaster_scaler,
        classifier_scaler,
        config,
        device,
        batch_size=config.batch_size,
    )
    baseline_predictions = current_labels

    separated_row = _metric_row("separated_pipeline", true_future_labels, separated_predictions)
    baseline_row = _metric_row("baseline_future_equals_current", true_future_labels, baseline_predictions)
    separated_df = pd.DataFrame([separated_row])
    comparison_df = pd.DataFrame([separated_row, baseline_row])
    separated_df.to_csv(config.reports_dir / "separated_pipeline_evaluation.csv", index=False)
    comparison_df.to_csv(config.reports_dir / "separated_vs_baseline.csv", index=False)

    matrix = confusion_matrix(true_future_labels, separated_predictions, labels=LABEL_IDS)
    save_confusion_matrix_csv(config.reports_dir / "separated_pipeline_confusion_matrix.csv", matrix, LABEL_NAMES)
    plot_confusion_matrix(
        matrix,
        LABEL_NAMES,
        "Матрица ошибок полной разделённой архитектуры",
        config.plots_dir / "separated_pipeline_confusion_matrix.png",
    )

    warning_path = config.reports_dir / "separated_pipeline_warning.txt"
    is_better = separated_row["macro_f1"] > baseline_row["macro_f1"]
    if not is_better:
        warning_path.write_text(
            "WARNING: separated pipeline macro_f1 <= baseline macro_f1.\n",
            encoding="utf-8",
        )
    elif warning_path.exists():
        warning_path.unlink()

    print("\nSeparated pipeline metrics")
    for key, value in separated_row.items():
        if key != "method":
            print(f"  {key}: {value:.4f}")

    print("\nBaseline metrics")
    for key, value in baseline_row.items():
        if key != "method":
            print(f"  {key}: {value:.4f}")

    print(f"\nSeparated pipeline better than baseline by macro_f1: {is_better}")


if __name__ == "__main__":
    main()
