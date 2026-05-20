from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import Config, ensure_dirs
from utils import (
    BASELINE_RED,
    BEST_GREEN,
    NORMAL_BLUE,
    save_publication_figure,
    setup_publication_style,
)

import matplotlib.pyplot as plt


def _require_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required report not found: {path}")
    return pd.read_csv(path)


def _style_axes(ax) -> None:
    ax.grid(True, axis="x", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.tick_params(width=0.8)
    for side in ["left", "bottom"]:
        ax.spines[side].set_linewidth(0.8)


def _horizontal_score_chart(
    ax,
    df: pd.DataFrame,
    score_col: str,
    title: str,
    label_col: str = "model_name",
) -> None:
    plot_df = df.sort_values(score_col, ascending=True).reset_index(drop=True)
    best_idx = int(plot_df[score_col].idxmax())
    colors = [BEST_GREEN if idx == best_idx else NORMAL_BLUE for idx in range(len(plot_df))]
    ax.barh(
        plot_df[label_col],
        plot_df[score_col],
        color=colors,
        edgecolor="black",
        linewidth=0.6,
    )
    ax.set_title(title, pad=10)
    ax.set_xlabel("Интегральная оценка, отн. ед.")
    ax.set_ylabel("Модель")
    ax.set_xlim(0, 1.05)
    _style_axes(ax)
    for index, value in enumerate(plot_df[score_col]):
        ax.text(
            min(value + 0.015, 1.015),
            index,
            f"{value:.3f}",
            va="center",
            fontsize=11,
            fontstyle="italic",
        )


def _pipeline_chart(ax, df: pd.DataFrame) -> None:
    label_map = {
        "separated_pipeline": "Разделённая архитектура",
        "baseline_future_equals_current": "Baseline: future=current",
    }
    plot_df = df.copy()
    plot_df["label"] = plot_df["method"].replace(label_map)
    colors = [BEST_GREEN if method == "separated_pipeline" else BASELINE_RED for method in plot_df["method"]]
    ax.bar(
        plot_df["label"],
        plot_df["macro_f1"],
        color=colors,
        edgecolor="black",
        linewidth=0.6,
        width=0.55,
    )
    ax.set_title("Полная архитектура и baseline", pad=10)
    ax.set_ylabel("Macro F1, отн. ед.")
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.tick_params(axis="x", rotation=8, width=0.8)
    ax.tick_params(axis="y", width=0.8)
    for side in ["left", "bottom"]:
        ax.spines[side].set_linewidth(0.8)
    for index, value in enumerate(plot_df["macro_f1"]):
        ax.text(
            index,
            value + 0.025,
            f"{value:.3f}",
            ha="center",
            fontsize=11,
            fontstyle="italic",
        )


def main() -> None:
    config = Config()
    ensure_dirs(config)
    setup_publication_style()

    forecast_df = _require_csv(config.reports_dir / "forecast_model_comparison.csv")
    current_df = _require_csv(config.reports_dir / "current_classifier_comparison.csv")
    future_df = _require_csv(config.reports_dir / "future_classifier_comparison.csv")
    separated_vs_baseline_df = _require_csv(config.reports_dir / "separated_vs_baseline.csv")
    _ = _require_csv(config.reports_dir / "separated_pipeline_evaluation.csv")

    fig, axes = plt.subplots(2, 2, figsize=(16, 9), dpi=300)
    fig.suptitle("Сравнение нейросетевых моделей SystemMonitorAgent", fontsize=20, y=0.98)

    _horizontal_score_chart(
        axes[0, 0],
        forecast_df,
        "forecast_score",
        "Модель прогнозирования метрик",
    )
    _horizontal_score_chart(
        axes[0, 1],
        current_df,
        "classification_score",
        "Классификатор текущей нагрузки",
    )
    _horizontal_score_chart(
        axes[1, 0],
        future_df,
        "classification_score",
        "Классификатор будущей нагрузки",
    )
    _pipeline_chart(axes[1, 1], separated_vs_baseline_df)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    output_base = config.plots_dir / "publication_system_monitor_model_comparison"
    save_publication_figure(fig, output_base, dpi=300, cmyk=True)
    plt.close(fig)

    print(f"Publication figure saved to: {output_base.with_suffix('.png')}")


if __name__ == "__main__":
    main()
