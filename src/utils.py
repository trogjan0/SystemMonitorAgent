from __future__ import annotations

import json
import os
import random
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_MPLCONFIGDIR = Path(os.environ.get("MPLCONFIGDIR", Path(tempfile.gettempdir()) / "system_monitor_agent_matplotlib"))
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(_MPLCONFIGDIR)

import matplotlib.pyplot as plt
from PIL import Image


BEST_GREEN = "#5A9E4B"
NORMAL_BLUE = "#4E79A7"
BASELINE_RED = "#E15759"
NEUTRAL_GRAY = "#D9D9D9"


def setup_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "axes.titlesize": 15,
            "axes.labelsize": 13,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "figure.titlesize": 18,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.6,
            "grid.linestyle": "--",
            "grid.alpha": 0.35,
            "lines.linewidth": 1.4,
            "lines.markersize": 5,
            "patch.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _figure_output_base(output_base_path: Path) -> Path:
    output_base_path = Path(output_base_path)
    if output_base_path.suffix:
        return output_base_path.with_suffix("")
    return output_base_path


def save_publication_figure(fig, output_base_path: Path, dpi: int = 300, cmyk: bool = True) -> None:
    output_base = _figure_output_base(Path(output_base_path))
    output_base.parent.mkdir(parents=True, exist_ok=True)

    png_path = output_base.with_suffix(".png")
    svg_path = output_base.with_suffix(".svg")
    tiff_path = output_base.with_suffix(".tiff")
    temp_png_handle = tempfile.NamedTemporaryFile(
        prefix=f"{output_base.name}__",
        suffix=".png",
        delete=False,
    )
    temp_png_path = Path(temp_png_handle.name)
    temp_png_handle.close()

    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    fig.savefig(temp_png_path, dpi=dpi, bbox_inches="tight")

    with Image.open(temp_png_path) as image:
        output_image = image.convert("CMYK") if cmyk else image.copy()
        output_image.save(tiff_path, compression="tiff_lzw", dpi=(dpi, dpi))
        output_image.close()
    temp_png_path.unlink(missing_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def measure_latency(model: torch.nn.Module, sample_input: torch.Tensor, device: torch.device, repeats: int = 200) -> float:
    model.eval()
    sample_input = sample_input.to(device)
    warmup = min(20, max(3, repeats // 10))
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(sample_input)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(repeats):
            _ = model(sample_input)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
    return elapsed * 1000.0 / repeats


def split_episodes(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must be 1.0")

    episode_ids = np.array(sorted(df["episode_id"].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(episode_ids)

    train_end = int(len(episode_ids) * train_ratio)
    val_end = train_end + int(len(episode_ids) * val_ratio)
    train_ids = set(episode_ids[:train_end])
    val_ids = set(episode_ids[train_end:val_end])
    test_ids = set(episode_ids[val_end:])

    train_df = df[df["episode_id"].isin(train_ids)].copy()
    val_df = df[df["episode_id"].isin(val_ids)].copy()
    test_df = df[df["episode_id"].isin(test_ids)].copy()
    return train_df, val_df, test_df


def save_json(path: Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def load_json(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def save_confusion_matrix_csv(path: Path, matrix: np.ndarray, labels: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(matrix, index=labels, columns=labels)
    df.to_csv(path)


def plot_confusion_matrix(
    matrix: np.ndarray,
    labels: list[str],
    title: str,
    path: Path,
    xlabel: str = "Предсказанный класс",
    ylabel: str = "Истинный класс",
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    setup_publication_style()

    fig, ax = plt.subplots(figsize=(7.2, 6.0), dpi=300)
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title(title, pad=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(len(labels)), labels=labels, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels=labels)
    ax.tick_params(width=0.8)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    max_value = matrix.max() if matrix.size else 0
    threshold = max_value / 2 if max_value else 0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            color = "white" if matrix[row, col] > threshold else "black"
            ax.text(
                col,
                row,
                str(matrix[row, col]),
                ha="center",
                va="center",
                color=color,
                fontsize=12,
                fontstyle="italic",
            )

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Количество объектов, ед.")
    colorbar.ax.tick_params(labelsize=12, width=0.8)
    fig.tight_layout()
    save_publication_figure(fig, path, dpi=300, cmyk=True)
    plt.close(fig)


def normalize_inverse(values) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    min_value = np.min(array)
    max_value = np.max(array)
    if np.isclose(max_value, min_value):
        return np.ones_like(array, dtype=float)
    return (max_value - array) / (max_value - min_value)


def latency_sla_score(latency_ms: float) -> float:
    if latency_ms < 1.0:
        return 1.0
    if latency_ms < 5.0:
        return 0.5
    return 0.0
