from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Config:
    seed: int = 42
    num_episodes: int = 160
    episode_length: int = 120
    window_size: int = 30
    forecast_horizon: int = 10
    batch_size: int = 64
    epochs: int = 20
    lr: float = 1e-3
    num_classes: int = 4

    feature_names: list[str] = field(
        default_factory=lambda: [
            "cpu_percent",
            "mem_percent",
            "swap_percent",
            "io_read_mb",
            "io_write_mb",
            "net_in_mb",
            "net_out_mb",
            "psi_cpu",
            "psi_mem",
            "psi_io",
            "process_count",
            "blocked_processes",
        ]
    )

    raw_data_path: Path = PROJECT_ROOT / "data" / "raw" / "synthetic_metrics.csv"
    models_dir: Path = PROJECT_ROOT / "artifacts" / "models"
    reports_dir: Path = PROJECT_ROOT / "artifacts" / "reports"
    plots_dir: Path = PROJECT_ROOT / "artifacts" / "plots"

    @property
    def num_features(self) -> int:
        return len(self.feature_names)


def ensure_dirs(config: Config | None = None) -> None:
    cfg = config or Config()
    directories = [
        cfg.raw_data_path.parent,
        PROJECT_ROOT / "data" / "processed",
        cfg.models_dir,
        cfg.reports_dir,
        cfg.plots_dir,
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
