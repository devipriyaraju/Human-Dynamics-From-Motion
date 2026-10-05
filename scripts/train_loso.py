from __future__ import annotations

import argparse

import torch

from motion_dynamics.config import load_config
from motion_dynamics.data.windows import group_windows_by_subject
from motion_dynamics.evaluation.loso import run_loso
from motion_dynamics.training.engine import TrainOptions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(cfg.paths.processed_cache, map_location="cpu", weights_only=False)
    grouped = group_windows_by_subject(
        payload["trials"],
        cfg.window,
        cfg.stride,
        device,
    )
    options = TrainOptions(
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        seed=cfg.seed,
        gravity=cfg.gravity,
    )
    results = run_loso(grouped, cfg.hidden_dim, cfg.fps, options)
    output = cfg.paths.results_dir / "tables" / "loso_all.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output)
    print(output)


if __name__ == "__main__":
    main()
