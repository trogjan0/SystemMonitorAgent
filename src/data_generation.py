from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd

from config import Config, ensure_dirs
from metrics_schema import score_to_class
from utils import set_seed


SCENARIO_TYPES = [
    "normal",
    "cpu_spike",
    "memory_leak",
    "io_burst",
    "mixed_overload",
    "recovery",
    "pre_critical_growth",
    "critical_overload",
]


def _clip(values: np.ndarray, low: float, high: float) -> np.ndarray:
    return np.clip(values, low, high)


def _lagged_pressure(source: np.ndarray, threshold: float, scale: float, delay: int, noise: np.ndarray) -> np.ndarray:
    lagged = np.roll(source, delay)
    lagged[:delay] = source[0]
    pressure = np.maximum(lagged - threshold, 0.0) / scale
    return _clip(pressure + noise, 0.0, 1.0)


def _base_episode(length: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    t = np.arange(length)
    daily_wave = np.sin(2 * np.pi * t / max(length, 1))
    short_wave = np.sin(2 * np.pi * t / 24)

    return {
        "cpu_percent": 28 + 6 * daily_wave + 3 * short_wave + rng.normal(0, 2.0, length),
        "mem_percent": 36 + 3 * daily_wave + rng.normal(0, 1.6, length),
        "swap_percent": 5 + rng.normal(0, 0.8, length),
        "io_read_mb": 28 + 8 * np.maximum(short_wave, 0) + rng.normal(0, 4.0, length),
        "io_write_mb": 22 + 7 * np.maximum(-short_wave, 0) + rng.normal(0, 3.5, length),
        "net_in_mb": 18 + 5 * np.maximum(short_wave, 0) + rng.normal(0, 2.5, length),
        "net_out_mb": 15 + 4 * np.maximum(-short_wave, 0) + rng.normal(0, 2.2, length),
        "process_count": 120 + 10 * daily_wave + rng.normal(0, 6.0, length),
        "blocked_processes": 2 + rng.poisson(1.5, length).astype(float),
    }


def _pulse(t: np.ndarray, center: float, width: float, amplitude: float) -> np.ndarray:
    return amplitude * np.exp(-0.5 * ((t - center) / width) ** 2)


def _apply_scenario(base: dict[str, np.ndarray], scenario_type: str, rng: np.random.Generator) -> dict[str, np.ndarray]:
    metrics = {key: value.copy() for key, value in base.items()}
    length = len(next(iter(metrics.values())))
    t = np.arange(length)
    progress = t / max(length - 1, 1)

    if scenario_type == "normal":
        metrics["cpu_percent"] += rng.normal(0, 1.5, length)
        metrics["mem_percent"] += rng.normal(0, 1.0, length)

    elif scenario_type == "cpu_spike":
        centers = rng.choice(np.arange(18, length - 18), size=3, replace=False)
        for center in centers:
            spike = _pulse(t, center, rng.uniform(2.5, 4.5), rng.uniform(48, 65))
            metrics["cpu_percent"] += spike
            metrics["blocked_processes"] += 18 * (spike / max(spike.max(), 1.0))
        metrics["process_count"] += 25 * (metrics["cpu_percent"] / 100)

    elif scenario_type == "memory_leak":
        growth = 48 * progress**1.25
        metrics["mem_percent"] += growth
        metrics["swap_percent"] += np.maximum(progress - 0.42, 0) * 78
        metrics["blocked_processes"] += np.maximum(progress - 0.55, 0) * 55
        metrics["process_count"] += 45 * progress

    elif scenario_type == "io_burst":
        burst = np.zeros(length)
        for center in range(18, length, 24):
            burst += _pulse(t, center, 3.8, rng.uniform(150, 230))
        metrics["io_read_mb"] += burst * rng.uniform(0.65, 0.90)
        metrics["io_write_mb"] += burst * rng.uniform(0.55, 0.85)
        metrics["blocked_processes"] += burst / 8
        metrics["cpu_percent"] += burst / 18

    elif scenario_type == "mixed_overload":
        ramp = np.clip((progress - 0.10) / 0.78, 0, 1)
        metrics["cpu_percent"] += 58 * ramp + 8 * np.sin(2 * np.pi * progress * 4)
        metrics["mem_percent"] += 52 * ramp
        metrics["swap_percent"] += 58 * np.clip((progress - 0.35) / 0.55, 0, 1)
        metrics["io_read_mb"] += 190 * ramp + 35 * np.maximum(np.sin(2 * np.pi * progress * 8), 0)
        metrics["io_write_mb"] += 170 * ramp + 30 * np.maximum(np.sin(2 * np.pi * progress * 7), 0)
        metrics["blocked_processes"] += 85 * ramp
        metrics["process_count"] += 80 * ramp

    elif scenario_type == "recovery":
        decay = 1 - progress
        metrics["cpu_percent"] = 35 + 58 * decay + rng.normal(0, 2.0, length)
        metrics["mem_percent"] = 42 + 48 * decay + rng.normal(0, 1.8, length)
        metrics["swap_percent"] = 12 + 48 * decay + rng.normal(0, 1.5, length)
        metrics["io_read_mb"] = 35 + 190 * decay + rng.normal(0, 8.0, length)
        metrics["io_write_mb"] = 30 + 150 * decay + rng.normal(0, 7.0, length)
        metrics["blocked_processes"] = 5 + 70 * decay + rng.normal(0, 3.0, length)
        metrics["process_count"] = 140 + 75 * decay + rng.normal(0, 5.0, length)

    elif scenario_type == "pre_critical_growth":
        # The final third intentionally crosses the future horizon boundary:
        # current windows still look HIGH, while t+h becomes CRITICAL.
        ramp = np.clip((progress - 0.22) / 0.70, 0, 1)
        metrics["cpu_percent"] += 66 * ramp**1.15
        metrics["mem_percent"] += 60 * ramp**1.20
        metrics["swap_percent"] += 72 * np.clip((progress - 0.44) / 0.45, 0, 1)
        metrics["io_read_mb"] += 210 * ramp**1.35
        metrics["io_write_mb"] += 185 * ramp**1.35
        metrics["blocked_processes"] += 92 * ramp**1.45
        metrics["process_count"] += 95 * ramp

    elif scenario_type == "critical_overload":
        metrics["cpu_percent"] = 88 + 6 * np.sin(2 * np.pi * progress * 5) + rng.normal(0, 2.2, length)
        metrics["mem_percent"] = 90 + 4 * np.sin(2 * np.pi * progress * 3) + rng.normal(0, 1.8, length)
        metrics["swap_percent"] = 76 + 8 * np.maximum(np.sin(2 * np.pi * progress * 4), 0) + rng.normal(0, 2.5, length)
        metrics["io_read_mb"] = 210 + 45 * np.maximum(np.sin(2 * np.pi * progress * 8), 0) + rng.normal(0, 12, length)
        metrics["io_write_mb"] = 185 + 40 * np.maximum(np.sin(2 * np.pi * progress * 7), 0) + rng.normal(0, 10, length)
        metrics["blocked_processes"] = 72 + 18 * np.maximum(np.sin(2 * np.pi * progress * 6), 0) + rng.normal(0, 5, length)
        metrics["process_count"] = 230 + 28 * np.sin(2 * np.pi * progress * 2) + rng.normal(0, 8, length)

    return metrics


def _finalize_metrics(metrics: dict[str, np.ndarray], rng: np.random.Generator) -> dict[str, np.ndarray]:
    length = len(metrics["cpu_percent"])
    io_total = metrics["io_read_mb"] + metrics["io_write_mb"]

    metrics["cpu_percent"] = _clip(metrics["cpu_percent"], 0, 100)
    metrics["mem_percent"] = _clip(metrics["mem_percent"], 0, 100)
    metrics["swap_percent"] = _clip(metrics["swap_percent"], 0, 100)
    metrics["io_read_mb"] = _clip(metrics["io_read_mb"], 0, 320)
    metrics["io_write_mb"] = _clip(metrics["io_write_mb"], 0, 300)
    metrics["net_in_mb"] = _clip(metrics["net_in_mb"] + 0.10 * metrics["io_read_mb"], 0, 260)
    metrics["net_out_mb"] = _clip(metrics["net_out_mb"] + 0.08 * metrics["io_write_mb"], 0, 240)
    metrics["process_count"] = _clip(metrics["process_count"], 0, 500)
    metrics["blocked_processes"] = _clip(metrics["blocked_processes"], 0, 150)

    metrics["psi_cpu"] = _lagged_pressure(
        metrics["cpu_percent"],
        threshold=52,
        scale=48,
        delay=2,
        noise=rng.normal(0, 0.025, length),
    )
    metrics["psi_mem"] = _lagged_pressure(
        metrics["mem_percent"] + 0.45 * metrics["swap_percent"],
        threshold=58,
        scale=70,
        delay=4,
        noise=rng.normal(0, 0.025, length),
    )
    metrics["psi_io"] = _lagged_pressure(
        io_total,
        threshold=85,
        scale=320,
        delay=3,
        noise=rng.normal(0, 0.030, length),
    )

    return metrics


def compute_load_score(df: pd.DataFrame) -> pd.Series:
    cpu_score = df["cpu_percent"] / 100.0
    mem_score = df["mem_percent"] / 100.0
    swap_score = df["swap_percent"] / 100.0
    io_score = np.clip((df["io_read_mb"] + df["io_write_mb"]) / 400.0, 0.0, 1.0)
    psi_score = 0.35 * df["psi_cpu"] + 0.40 * df["psi_mem"] + 0.25 * df["psi_io"]
    blocked_score = np.clip(df["blocked_processes"] / 100.0, 0.0, 1.0)

    return (
        0.25 * cpu_score
        + 0.25 * mem_score
        + 0.10 * swap_score
        + 0.10 * io_score
        + 0.20 * psi_score
        + 0.10 * blocked_score
    ).clip(0.0, 1.0)


def generate_synthetic_dataset(config: Config | None = None) -> pd.DataFrame:
    cfg = config or Config()
    ensure_dirs(cfg)
    set_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    rows: list[dict[str, float | int | str]] = []
    for episode_id in range(cfg.num_episodes):
        scenario_type = SCENARIO_TYPES[episode_id % len(SCENARIO_TYPES)]
        base = _base_episode(cfg.episode_length, rng)
        metrics = _apply_scenario(base, scenario_type, rng)
        metrics = _finalize_metrics(metrics, rng)

        for timestep in range(cfg.episode_length):
            row: dict[str, float | int | str] = {
                "episode_id": episode_id,
                "scenario_type": scenario_type,
                "timestep": timestep,
            }
            for feature in cfg.feature_names:
                value = metrics[feature][timestep]
                if feature in {"process_count", "blocked_processes"}:
                    value = round(float(value))
                row[feature] = float(value)
            rows.append(row)

    df = pd.DataFrame(rows)
    df["load_score"] = compute_load_score(df)
    df["load_class"] = df["load_score"].map(score_to_class).astype(int)

    future_scores: list[float] = []
    for _, episode in df.groupby("episode_id", sort=False):
        scores = episode["load_score"].to_numpy()
        for idx in range(len(episode)):
            start = idx + 1
            end = min(idx + cfg.forecast_horizon + 1, len(episode))
            if start >= end:
                future_scores.append(float(scores[idx]))
            else:
                future_scores.append(float(np.max(scores[start:end])))

    df["future_load_score"] = future_scores
    df["future_load_class"] = df["future_load_score"].map(score_to_class).astype(int)

    df.to_csv(cfg.raw_data_path, index=False)
    return df


def main() -> None:
    cfg = Config()
    df = generate_synthetic_dataset(cfg)
    current_counts = df["load_class"].value_counts().sort_index().to_dict()
    future_counts = df["future_load_class"].value_counts().sort_index().to_dict()
    print(f"Synthetic dataset saved to: {cfg.raw_data_path}")
    print(f"Rows: {len(df)}")
    print(f"Config: {asdict(cfg) | {'raw_data_path': str(cfg.raw_data_path)}}")
    print(f"Current class distribution: {current_counts}")
    print(f"Future class distribution: {future_counts}")


if __name__ == "__main__":
    main()
