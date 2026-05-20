from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, recall_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from classifier_dataset import CurrentClassifierDataset, FutureClassifierDataset
from classifier_models import build_classifier
from config import Config, ensure_dirs
from data_generation import generate_synthetic_dataset
from metrics_schema import LOAD_CLASS_NAMES
from utils import (
    BEST_GREEN,
    NORMAL_BLUE,
    count_parameters,
    latency_sla_score,
    measure_latency,
    plot_confusion_matrix,
    save_confusion_matrix_csv,
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


CLASSIFIER_MODELS = ["MLPClassifier", "CNN1DClassifier", "GRUClassifier", "LSTMClassifier"]
LABEL_IDS = [0, 1, 2, 3]
LABEL_NAMES = [LOAD_CLASS_NAMES[idx] for idx in LABEL_IDS]


def _load_or_create_data(config: Config) -> pd.DataFrame:
    if not config.raw_data_path.exists():
        print("synthetic_metrics.csv not found, generating dataset...")
        return generate_synthetic_dataset(config)
    return pd.read_csv(config.raw_data_path)


def _scale_dataframe(df: pd.DataFrame, scaler: StandardScaler, feature_names: list[str]) -> pd.DataFrame:
    scaled = df.copy()
    scaled[feature_names] = scaler.transform(df[feature_names])
    return scaled


def _dataset_labels(dataset) -> np.ndarray:
    labels = []
    for episode_idx, position in dataset.samples:
        episode = dataset.episodes[episode_idx]
        episode_labels = episode["labels"]
        assert isinstance(episode_labels, np.ndarray)
        labels.append(int(episode_labels[position]))
    return np.asarray(labels, dtype=np.int64)


def _class_weights(dataset, num_classes: int, device: torch.device) -> torch.Tensor:
    labels = _dataset_labels(dataset)
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _train_one_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_weights: torch.Tensor,
    config: Config,
    device: torch.device,
) -> torch.nn.Module:
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    best_state = None
    best_macro_f1 = -math.inf
    patience = 5
    epochs_without_improvement = 0

    model.to(device)
    for epoch in range(1, config.epochs + 1):
        model.train()
        train_losses = []
        for x_window, target in train_loader:
            x_window = x_window.to(device)
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_window)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        y_true, y_pred = _predict(model, val_loader, device)
        macro_f1 = f1_score(y_true, y_pred, average="macro", labels=LABEL_IDS, zero_division=0)
        print(f"  epoch {epoch:02d}: train_loss={np.mean(train_losses):.5f} val_macro_f1={macro_f1:.4f}")

        if macro_f1 > best_macro_f1 + 1e-4:
            best_macro_f1 = macro_f1
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


def _predict(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for x_window, target in loader:
            logits = model(x_window.to(device))
            predictions = torch.argmax(logits, dim=1).cpu().numpy()
            y_pred.extend(predictions.tolist())
            y_true.extend(target.numpy().tolist())
    return np.asarray(y_true, dtype=np.int64), np.asarray(y_pred, dtype=np.int64)


def _evaluate_classifier(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[dict, np.ndarray, pd.DataFrame]:
    y_true, y_pred = _predict(model, loader, device)
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=LABEL_IDS,
        target_names=LABEL_NAMES,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=LABEL_IDS)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", labels=LABEL_IDS, zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", labels=LABEL_IDS, zero_division=0),
        "high_recall": recall_score(y_true, y_pred, labels=[2], average="macro", zero_division=0),
        "critical_recall": recall_score(y_true, y_pred, labels=[3], average="macro", zero_division=0),
    }
    return metrics, matrix, pd.DataFrame(report_dict).T


def _comparison_plot(df: pd.DataFrame, title: str, path: Path) -> None:
    setup_publication_style()
    plot_df = df.sort_values("classification_score", ascending=True)
    colors = [BEST_GREEN if idx == plot_df["classification_score"].idxmax() else NORMAL_BLUE for idx in plot_df.index]
    fig, ax = plt.subplots(figsize=(8.8, 5.2), dpi=300)
    ax.barh(plot_df["model_name"], plot_df["classification_score"], color=colors, edgecolor="black", linewidth=0.6)
    ax.set_title(title)
    ax.set_xlabel("Интегральная оценка, отн. ед.")
    ax.set_ylabel("Модель")
    ax.set_xlim(0, 1.05)
    ax.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.tick_params(width=0.8)
    for index, value in enumerate(plot_df["classification_score"]):
        ax.text(value + 0.01, index, f"{value:.4f}", va="center", fontsize=11, fontstyle="italic")
    fig.tight_layout()
    save_publication_figure(fig, path, dpi=300, cmyk=True)
    plt.close(fig)


def _train_group(
    group_name: str,
    train_dataset,
    val_dataset,
    test_dataset,
    input_time: int,
    config: Config,
    device: torch.device,
) -> pd.DataFrame:
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    weights = _class_weights(train_dataset, config.num_classes, device)

    rows = []
    best_matrix = None
    best_model_name = None
    for model_name in CLASSIFIER_MODELS:
        print(f"\nTraining {model_name} ({group_name})")
        model = build_classifier(model_name, config, input_time=input_time)
        model = _train_one_model(model, train_loader, val_loader, weights, config, device)

        sample_input = next(iter(test_loader))[0][:1]
        latency_ms = measure_latency(model, sample_input, device)
        metrics, matrix, report_df = _evaluate_classifier(model, test_loader, device)
        params = count_parameters(model)

        suffix = "current" if group_name == "current" else "future"
        model_path = config.models_dir / f"{model_name}_{suffix}_classifier.pt"
        torch.save(
            {
                "model_name": model_name,
                "classifier_type": suffix,
                "state_dict": model.cpu().state_dict(),
                "input_time": input_time,
                "feature_names": config.feature_names,
            },
            model_path,
        )
        model.to(device)

        report_df.to_csv(config.reports_dir / f"classification_report_{model_name}_{suffix}.csv")
        save_confusion_matrix_csv(
            config.reports_dir / f"confusion_matrix_{model_name}_{suffix}.csv",
            matrix,
            LABEL_NAMES,
        )

        classification_score = (
            0.45 * metrics["macro_f1"]
            + 0.25 * metrics["high_recall"]
            + 0.20 * metrics["critical_recall"]
            + 0.10 * latency_sla_score(latency_ms)
        )
        row = {
            "model_name": model_name,
            **metrics,
            "latency_ms": latency_ms,
            "latency_sla_score": latency_sla_score(latency_ms),
            "params": params,
            "classification_score": classification_score,
            "model_path": str(model_path),
        }
        rows.append(row)
        print(
            f"  test: accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f} "
            f"HIGH_recall={metrics['high_recall']:.4f} CRITICAL_recall={metrics['critical_recall']:.4f} "
            f"latency={latency_ms:.3f} ms"
        )

        if best_model_name is None or classification_score > max(item["classification_score"] for item in rows[:-1]):
            best_model_name = model_name
            best_matrix = matrix

    comparison_df = pd.DataFrame(rows).sort_values("classification_score", ascending=False)
    suffix = "current" if group_name == "current" else "future"
    comparison_df.to_csv(config.reports_dir / f"{suffix}_classifier_comparison.csv", index=False)

    best_row = comparison_df.iloc[0].to_dict()
    best_json = {
        "model_name": str(best_row["model_name"]),
        "classifier_type": suffix,
        "model_path": str(Path("artifacts") / "models" / f"{best_row['model_name']}_{suffix}_classifier.pt"),
        "scaler_path": str(Path("artifacts") / "models" / "classifier_scaler.pkl"),
        "classification_score": float(best_row["classification_score"]),
        "accuracy": float(best_row["accuracy"]),
        "macro_f1": float(best_row["macro_f1"]),
        "weighted_f1": float(best_row["weighted_f1"]),
        "high_recall": float(best_row["high_recall"]),
        "critical_recall": float(best_row["critical_recall"]),
        "latency_ms": float(best_row["latency_ms"]),
        "feature_names": config.feature_names,
        "input_time": input_time,
        "window_size": config.window_size,
        "forecast_horizon": config.forecast_horizon,
    }
    save_json(config.models_dir / f"best_{suffix}_classifier.json", best_json)

    if best_matrix is not None:
        plot_confusion_matrix(
            best_matrix,
            LABEL_NAMES,
            (
                f"Матрица ошибок текущей классификации: {comparison_df.iloc[0]['model_name']}"
                if suffix == "current"
                else f"Матрица ошибок будущей классификации: {comparison_df.iloc[0]['model_name']}"
            ),
            config.plots_dir / f"{suffix}_confusion_matrix_best.png",
        )
    _comparison_plot(
        comparison_df,
        (
            "Сравнение классификаторов текущей нагрузки"
            if suffix == "current"
            else "Сравнение классификаторов будущей нагрузки"
        ),
        config.plots_dir / f"{suffix}_classifier_f1_comparison.png",
    )
    return comparison_df


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
    scaler_path = config.models_dir / "classifier_scaler.pkl"
    joblib.dump(scaler, scaler_path)

    train_scaled = _scale_dataframe(train_df, scaler, config.feature_names)
    val_scaled = _scale_dataframe(val_df, scaler, config.feature_names)
    test_scaled = _scale_dataframe(test_df, scaler, config.feature_names)

    current_train = CurrentClassifierDataset(train_scaled, config.feature_names, config.window_size)
    current_val = CurrentClassifierDataset(val_scaled, config.feature_names, config.window_size)
    current_test = CurrentClassifierDataset(test_scaled, config.feature_names, config.window_size)

    future_train = FutureClassifierDataset(train_scaled, config.feature_names, config.forecast_horizon)
    future_val = FutureClassifierDataset(val_scaled, config.feature_names, config.forecast_horizon)
    future_test = FutureClassifierDataset(test_scaled, config.feature_names, config.forecast_horizon)

    current_df = _train_group("current", current_train, current_val, current_test, config.window_size, config, device)
    future_df = _train_group("future", future_train, future_val, future_test, config.forecast_horizon, config, device)

    best_current = current_df.iloc[0]
    best_future = future_df.iloc[0]
    print("\nBest current classifier")
    print(f"  model: {best_current['model_name']}")
    print(f"  accuracy: {best_current['accuracy']:.4f}")
    print(f"  macro F1: {best_current['macro_f1']:.4f}")
    print(f"  HIGH recall: {best_current['high_recall']:.4f}")
    print(f"  CRITICAL recall: {best_current['critical_recall']:.4f}")

    print("\nBest future classifier")
    print(f"  model: {best_future['model_name']}")
    print(f"  accuracy: {best_future['accuracy']:.4f}")
    print(f"  macro F1: {best_future['macro_f1']:.4f}")
    print(f"  HIGH recall: {best_future['high_recall']:.4f}")
    print(f"  CRITICAL recall: {best_future['critical_recall']:.4f}")


if __name__ == "__main__":
    main()
