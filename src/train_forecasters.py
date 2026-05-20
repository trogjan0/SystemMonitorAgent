from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from config import Config, ensure_dirs
from data_generation import generate_synthetic_dataset
from forecast_dataset import ForecastDataset
from forecast_models import build_forecaster
from utils import (
    BASELINE_RED,
    BEST_GREEN,
    NORMAL_BLUE,
    count_parameters,
    latency_sla_score,
    measure_latency,
    normalize_inverse,
    save_json,
    save_publication_figure,
    set_seed,
    setup_publication_style,
    split_episodes,
)

_MPLCONFIGDIR = Path(os.environ.get("MPLCONFIGDIR", Path(tempfile.gettempdir()) / "system_monitor_agent_matplotlib"))
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(_MPLCONFIGDIR)

import matplotlib.pyplot as plt


FORECASTER_MODELS = ["MLPForecaster", "CNN1DForecaster", "GRUForecaster", "LSTMForecaster"]

FEATURE_WEIGHTS = {
    "cpu_percent": 1.5,
    "mem_percent": 1.5,
    "swap_percent": 1.0,
    "io_read_mb": 1.0,
    "io_write_mb": 1.0,
    "net_in_mb": 0.5,
    "net_out_mb": 0.5,
    "psi_cpu": 2.0,
    "psi_mem": 2.0,
    "psi_io": 1.5,
    "process_count": 0.8,
    "blocked_processes": 1.5,
}


def _load_or_create_data(config: Config) -> pd.DataFrame:
    if not config.raw_data_path.exists():
        print("synthetic_metrics.csv not found, generating dataset...")
        return generate_synthetic_dataset(config)
    return pd.read_csv(config.raw_data_path)


def _scale_dataframe(df: pd.DataFrame, scaler: StandardScaler, feature_names: list[str]) -> pd.DataFrame:
    scaled = df.copy()
    scaled[feature_names] = scaler.transform(df[feature_names])
    return scaled


def _weighted_mse(prediction: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return torch.mean(weights.view(1, 1, -1) * (prediction - target) ** 2)


def _train_one_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    weights: torch.Tensor,
    config: Config,
    device: torch.device,
) -> torch.nn.Module:
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    best_state = None
    best_val_loss = math.inf
    patience = 5
    epochs_without_improvement = 0

    model.to(device)
    weights = weights.to(device)
    for epoch in range(1, config.epochs + 1):
        model.train()
        train_losses = []
        for x_window, y_future, _, _ in train_loader:
            x_window = x_window.to(device)
            y_future = y_future.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x_window)
            loss = _weighted_mse(prediction, y_future, weights)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x_window, y_future, _, _ in val_loader:
                x_window = x_window.to(device)
                y_future = y_future.to(device)
                prediction = model(x_window)
                loss = _weighted_mse(prediction, y_future, weights)
                val_losses.append(loss.item())

        val_loss = float(np.mean(val_losses))
        print(f"  epoch {epoch:02d}: train_loss={np.mean(train_losses):.5f} val_loss={val_loss:.5f}")
        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print("  early stopping")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def _inverse_3d(values: np.ndarray, scaler: StandardScaler, num_features: int) -> np.ndarray:
    shape = values.shape
    return scaler.inverse_transform(values.reshape(-1, num_features)).reshape(shape)


def _evaluate_model(
    model: torch.nn.Module,
    test_loader: DataLoader,
    scaler: StandardScaler,
    train_feature_min: np.ndarray,
    train_feature_max: np.ndarray,
    feature_names: list[str],
    device: torch.device,
) -> tuple[dict, list[dict], list[dict], np.ndarray, np.ndarray]:
    model.eval()
    predictions_scaled = []
    targets_scaled = []
    with torch.no_grad():
        for x_window, y_future, _, _ in test_loader:
            prediction = model(x_window.to(device)).cpu().numpy()
            predictions_scaled.append(prediction)
            targets_scaled.append(y_future.numpy())

    predictions_scaled_array = np.concatenate(predictions_scaled, axis=0)
    targets_scaled_array = np.concatenate(targets_scaled, axis=0)
    predictions = _inverse_3d(predictions_scaled_array, scaler, len(feature_names))
    targets = _inverse_3d(targets_scaled_array, scaler, len(feature_names))

    errors = predictions - targets
    absolute_errors = np.abs(errors)
    squared_errors = errors**2
    feature_range = np.maximum(train_feature_max - train_feature_min, 1e-6)

    mae = float(np.mean(absolute_errors))
    rmse = float(np.sqrt(np.mean(squared_errors)))
    r2 = float(r2_score(targets.reshape(-1), predictions.reshape(-1)))
    normalized_mae = float(np.mean(absolute_errors / feature_range.reshape(1, 1, -1)))
    normalized_rmse = float(np.mean(np.sqrt(np.mean(squared_errors, axis=(0, 1))) / feature_range))

    metrics = {
        "forecast_mae": mae,
        "forecast_rmse": rmse,
        "forecast_r2": r2,
        "forecast_normalized_mae": normalized_mae,
        "forecast_normalized_rmse": normalized_rmse,
    }

    feature_rows = []
    for idx, feature_name in enumerate(feature_names):
        feature_abs = absolute_errors[:, :, idx]
        feature_sq = squared_errors[:, :, idx]
        feature_mae = float(np.mean(feature_abs))
        feature_rmse = float(np.sqrt(np.mean(feature_sq)))
        feature_rows.append(
            {
                "feature": feature_name,
                "mae": feature_mae,
                "rmse": feature_rmse,
                "normalized_mae": feature_mae / feature_range[idx],
                "normalized_rmse": feature_rmse / feature_range[idx],
            }
        )

    horizon_rows = []
    for horizon_idx in range(targets.shape[1]):
        horizon_abs = absolute_errors[:, horizon_idx, :]
        horizon_sq = squared_errors[:, horizon_idx, :]
        horizon_rows.append(
            {
                "horizon_step": horizon_idx + 1,
                "mae": float(np.mean(horizon_abs)),
                "rmse": float(np.sqrt(np.mean(horizon_sq))),
            }
        )

    return metrics, feature_rows, horizon_rows, predictions, targets


def _minmax(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    min_value = float(array.min())
    max_value = float(array.max())
    if np.isclose(max_value, min_value):
        return np.ones_like(array)
    return (array - min_value) / (max_value - min_value)


def _bar_plot(df: pd.DataFrame, x_col: str, title: str, xlabel: str, path: Path) -> None:
    setup_publication_style()
    fig, ax = plt.subplots(figsize=(8.8, 5.2), dpi=300)
    plot_df = df.sort_values(x_col, ascending=True)
    if x_col == "forecast_r2":
        best_index = plot_df[x_col].idxmax()
    else:
        best_index = plot_df[x_col].idxmin()
    colors = [BEST_GREEN if index == best_index else NORMAL_BLUE for index in plot_df.index]
    bars = ax.barh(plot_df["model_name"], plot_df[x_col], color=colors, edgecolor="black", linewidth=0.6)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Модель")
    ax.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.tick_params(width=0.8)
    for index, value in enumerate(plot_df[x_col]):
        ax.text(
            value,
            index,
            f" {value:.4f}",
            va="center",
            fontsize=11,
            fontstyle="italic",
        )
    for bar in bars:
        bar.set_linewidth(0.6)
    fig.tight_layout()
    save_publication_figure(fig, path, dpi=300, cmyk=True)
    plt.close(fig)


def _feature_metric_plot(
    feature_metrics: pd.DataFrame,
    best_model: str,
    metric: str,
    title: str,
    xlabel: str,
    path: Path,
) -> None:
    setup_publication_style()
    df = feature_metrics[feature_metrics["model_name"] == best_model].sort_values(metric)
    fig, ax = plt.subplots(figsize=(9.6, 6.0), dpi=300)
    ax.barh(df["feature"], df[metric], color=NORMAL_BLUE, edgecolor="black", linewidth=0.6)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Признак")
    ax.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.tick_params(width=0.8)
    for index, value in enumerate(df[metric]):
        ax.text(value, index, f" {value:.4f}", va="center", fontsize=11, fontstyle="italic")
    fig.tight_layout()
    save_publication_figure(fig, path, dpi=300, cmyk=True)
    plt.close(fig)


def _horizon_metric_plot(
    horizon_metrics: pd.DataFrame,
    best_model: str,
    metric: str,
    title: str,
    ylabel: str,
    path: Path,
) -> None:
    setup_publication_style()
    df = horizon_metrics[horizon_metrics["model_name"] == best_model]
    fig, ax = plt.subplots(figsize=(8.8, 5.0), dpi=300)
    ax.plot(
        df["horizon_step"],
        df[metric],
        marker="o",
        color=BASELINE_RED,
        linewidth=1.4,
        markersize=5,
        markeredgewidth=0.8,
    )
    ax.set_title(title)
    ax.set_xlabel("Шаг горизонта прогноза, ед.")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.tick_params(width=0.8)
    fig.tight_layout()
    save_publication_figure(fig, path, dpi=300, cmyk=True)
    plt.close(fig)


def _timeseries_plot(
    predictions: np.ndarray,
    targets: np.ndarray,
    feature_names: list[str],
    feature_name: str,
    path: Path,
) -> None:
    setup_publication_style()
    feature_idx = feature_names.index(feature_name)
    sample_idx = min(4, len(predictions) - 1)
    horizon = np.arange(1, predictions.shape[1] + 1)
    fig, ax = plt.subplots(figsize=(8.4, 4.8), dpi=300)
    ax.plot(
        horizon,
        targets[sample_idx, :, feature_idx],
        marker="o",
        linewidth=1.4,
        markersize=5,
        markeredgewidth=0.8,
        label="Истинное значение",
    )
    ax.plot(
        horizon,
        predictions[sample_idx, :, feature_idx],
        marker="s",
        linewidth=1.4,
        markersize=5,
        markeredgewidth=0.8,
        label="Прогноз",
    )
    ax.set_title(f"Пример прогноза: {feature_name}")
    ax.set_xlabel("Шаг горизонта прогноза, ед.")
    ax.set_ylabel("Значение метрики, отн. ед.")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.tick_params(width=0.8)
    ax.legend()
    fig.tight_layout()
    save_publication_figure(fig, path, dpi=300, cmyk=True)
    plt.close(fig)


def main() -> None:
    config = Config()
    ensure_dirs(config)
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = _load_or_create_data(config)
    train_df, val_df, test_df = split_episodes(df, seed=config.seed)

    scaler = StandardScaler()
    scaler.fit(train_df[config.feature_names])
    scaler_path = config.models_dir / "forecaster_scaler.pkl"
    joblib.dump(scaler, scaler_path)

    train_scaled = _scale_dataframe(train_df, scaler, config.feature_names)
    val_scaled = _scale_dataframe(val_df, scaler, config.feature_names)
    test_scaled = _scale_dataframe(test_df, scaler, config.feature_names)

    train_dataset = ForecastDataset(train_scaled, config.feature_names, config.window_size, config.forecast_horizon)
    val_dataset = ForecastDataset(val_scaled, config.feature_names, config.window_size, config.forecast_horizon)
    test_dataset = ForecastDataset(test_scaled, config.feature_names, config.window_size, config.forecast_horizon)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    feature_weights = torch.tensor([FEATURE_WEIGHTS[name] for name in config.feature_names], dtype=torch.float32)
    train_feature_min = train_df[config.feature_names].min().to_numpy(dtype=float)
    train_feature_max = train_df[config.feature_names].max().to_numpy(dtype=float)

    comparison_rows = []
    feature_metric_rows = []
    horizon_metric_rows = []
    prediction_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for model_name in FORECASTER_MODELS:
        print(f"\nTraining {model_name}")
        model = build_forecaster(model_name, config)
        model = _train_one_model(model, train_loader, val_loader, feature_weights, config, device)

        sample_input = next(iter(test_loader))[0][:1]
        latency_ms = measure_latency(model, sample_input, device)
        metrics, feature_rows, horizon_rows, predictions, targets = _evaluate_model(
            model,
            test_loader,
            scaler,
            train_feature_min,
            train_feature_max,
            config.feature_names,
            device,
        )

        params = count_parameters(model)
        model_path = config.models_dir / f"{model_name}.pt"
        torch.save(
            {
                "model_name": model_name,
                "state_dict": model.cpu().state_dict(),
                "window_size": config.window_size,
                "forecast_horizon": config.forecast_horizon,
                "feature_names": config.feature_names,
            },
            model_path,
        )
        model.to(device)

        row = {
            "model_name": model_name,
            **metrics,
            "latency_ms": latency_ms,
            "latency_sla_score": latency_sla_score(latency_ms),
            "params": params,
            "model_path": str(model_path),
        }
        comparison_rows.append(row)
        prediction_cache[model_name] = (predictions, targets)

        for feature_row in feature_rows:
            feature_metric_rows.append({"model_name": model_name, **feature_row})
        for horizon_row in horizon_rows:
            horizon_metric_rows.append({"model_name": model_name, **horizon_row})

        print(
            f"  test: MAE={metrics['forecast_mae']:.4f} RMSE={metrics['forecast_rmse']:.4f} "
            f"R2={metrics['forecast_r2']:.4f} latency={latency_ms:.3f} ms"
        )

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df["rmse_score"] = normalize_inverse(comparison_df["forecast_rmse"])
    comparison_df["mae_score"] = normalize_inverse(comparison_df["forecast_mae"])
    comparison_df["r2_score"] = _minmax(comparison_df["forecast_r2"].tolist())
    comparison_df["forecast_score"] = (
        0.40 * comparison_df["rmse_score"]
        + 0.35 * comparison_df["mae_score"]
        + 0.15 * comparison_df["r2_score"]
        + 0.10 * comparison_df["latency_sla_score"]
    )
    comparison_df = comparison_df.sort_values("forecast_score", ascending=False)

    feature_metrics_df = pd.DataFrame(feature_metric_rows)
    horizon_metrics_df = pd.DataFrame(horizon_metric_rows)
    comparison_df.to_csv(config.reports_dir / "forecast_model_comparison.csv", index=False)
    feature_metrics_df.to_csv(config.reports_dir / "forecast_feature_metrics.csv", index=False)
    horizon_metrics_df.to_csv(config.reports_dir / "forecast_horizon_metrics.csv", index=False)

    best_row = comparison_df.iloc[0].to_dict()
    best_model_name = str(best_row["model_name"])
    best_json = {
        "model_name": best_model_name,
        "model_path": str(Path("artifacts") / "models" / f"{best_model_name}.pt"),
        "scaler_path": str(Path("artifacts") / "models" / "forecaster_scaler.pkl"),
        "forecast_score": float(best_row["forecast_score"]),
        "forecast_mae": float(best_row["forecast_mae"]),
        "forecast_rmse": float(best_row["forecast_rmse"]),
        "forecast_r2": float(best_row["forecast_r2"]),
        "latency_ms": float(best_row["latency_ms"]),
        "feature_names": config.feature_names,
        "window_size": config.window_size,
        "forecast_horizon": config.forecast_horizon,
    }
    save_json(config.models_dir / "best_forecaster.json", best_json)

    _bar_plot(
        comparison_df,
        "forecast_rmse",
        "Сравнение моделей прогнозирования по RMSE",
        "RMSE, отн. ед.",
        config.plots_dir / "forecast_rmse_comparison.png",
    )
    _bar_plot(
        comparison_df,
        "forecast_mae",
        "Сравнение моделей прогнозирования по MAE",
        "MAE, отн. ед.",
        config.plots_dir / "forecast_mae_comparison.png",
    )
    _bar_plot(
        comparison_df,
        "forecast_r2",
        "Сравнение моделей прогнозирования по R²",
        "R², отн. ед.",
        config.plots_dir / "forecast_r2_comparison.png",
    )
    _feature_metric_plot(
        feature_metrics_df,
        best_model_name,
        "normalized_rmse",
        f"Нормированная RMSE по признакам: {best_model_name}",
        "Нормированная RMSE, отн. ед.",
        config.plots_dir / "forecast_normalized_rmse_by_feature.png",
    )
    _feature_metric_plot(
        feature_metrics_df,
        best_model_name,
        "normalized_mae",
        f"Нормированная MAE по признакам: {best_model_name}",
        "Нормированная MAE, отн. ед.",
        config.plots_dir / "forecast_normalized_mae_by_feature.png",
    )
    _horizon_metric_plot(
        horizon_metrics_df,
        best_model_name,
        "rmse",
        f"Ошибка RMSE по горизонту прогноза: {best_model_name}",
        "RMSE, отн. ед.",
        config.plots_dir / "forecast_rmse_by_horizon.png",
    )
    _horizon_metric_plot(
        horizon_metrics_df,
        best_model_name,
        "mae",
        f"Ошибка MAE по горизонту прогноза: {best_model_name}",
        "MAE, отн. ед.",
        config.plots_dir / "forecast_mae_by_horizon.png",
    )

    best_predictions, best_targets = prediction_cache[best_model_name]
    _timeseries_plot(best_predictions, best_targets, config.feature_names, "cpu_percent", config.plots_dir / "forecast_timeseries_cpu.png")
    _timeseries_plot(best_predictions, best_targets, config.feature_names, "mem_percent", config.plots_dir / "forecast_timeseries_mem.png")
    _timeseries_plot(best_predictions, best_targets, config.feature_names, "psi_cpu", config.plots_dir / "forecast_timeseries_psi_cpu.png")
    _timeseries_plot(best_predictions, best_targets, config.feature_names, "psi_mem", config.plots_dir / "forecast_timeseries_psi_mem.png")

    print("\nBest forecaster")
    print(f"  model: {best_model_name}")
    print(f"  MAE: {best_row['forecast_mae']:.4f}")
    print(f"  RMSE: {best_row['forecast_rmse']:.4f}")
    print(f"  R2: {best_row['forecast_r2']:.4f}")
    print(f"  latency: {best_row['latency_ms']:.3f} ms")


if __name__ == "__main__":
    main()
