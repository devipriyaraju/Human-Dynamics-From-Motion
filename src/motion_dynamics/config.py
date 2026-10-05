from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Paths:
    force_dir: Path
    motion_dir: Path
    smplh_dir: Path
    processed_cache: Path
    checkpoints_dir: Path
    results_dir: Path


@dataclass(frozen=True)
class Config:
    seed: int
    fps: int
    window: int
    stride: int
    lowpass_hz: float
    gravity: float
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    hidden_dim: int
    force_balance_weight: float
    valid_force_residual_fraction_g: float
    contact_threshold_bw: float
    paths: Paths
    validation_subject: str
    test_subject: str


def load_config(path: str | Path) -> Config:
    path = Path(path)
    raw: dict[str, Any] = yaml.safe_load(path.read_text())
    root = path.parent.parent.resolve()

    def resolve(value: str) -> Path:
        p = Path(value)
        return p if p.is_absolute() else root / p

    paths_raw = raw["paths"]
    paths = Paths(
        force_dir=resolve(paths_raw["force_dir"]),
        motion_dir=resolve(paths_raw["motion_dir"]),
        smplh_dir=resolve(paths_raw["smplh_dir"]),
        processed_cache=resolve(paths_raw["processed_cache"]),
        checkpoints_dir=resolve(paths_raw["checkpoints_dir"]),
        results_dir=resolve(paths_raw["results_dir"]),
    )
    split = raw["split"]
    return Config(
        seed=int(raw["seed"]),
        fps=int(raw["fps"]),
        window=int(raw["window"]),
        stride=int(raw["stride"]),
        lowpass_hz=float(raw["lowpass_hz"]),
        gravity=float(raw["gravity"]),
        batch_size=int(raw["batch_size"]),
        epochs=int(raw["epochs"]),
        learning_rate=float(raw["learning_rate"]),
        weight_decay=float(raw["weight_decay"]),
        hidden_dim=int(raw["hidden_dim"]),
        force_balance_weight=float(raw["force_balance_weight"]),
        valid_force_residual_fraction_g=float(raw["valid_force_residual_fraction_g"]),
        contact_threshold_bw=float(raw["contact_threshold_bw"]),
        paths=paths,
        validation_subject=str(split["validation_subject"]),
        test_subject=str(split["test_subject"]),
    )
