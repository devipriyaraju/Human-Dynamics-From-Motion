from __future__ import annotations

import argparse

import torch

from motion_dynamics.config import load_config
from motion_dynamics.data.windows import group_windows_by_subject
from motion_dynamics.evaluation.loso import evaluate_prediction
from motion_dynamics.models.tcn import DynamicsTCN


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--subject", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = torch.load(cfg.paths.processed_cache, map_location="cpu", weights_only=False)
    grouped = group_windows_by_subject(cache["trials"], cfg.window, cfg.stride, device)
    subject = args.subject or cfg.test_subject
    data = grouped[subject]
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = DynamicsTCN(
        checkpoint["mean"].to(device),
        checkpoint["std"].to(device),
        hidden_dim=checkpoint.get("metadata", {}).get("hidden_dim", cfg.hidden_dim),
        temporal=checkpoint.get("metadata", {}).get("temporal", True),
        fps=cfg.fps,
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with torch.no_grad():
        metrics = evaluate_prediction(model(data["X"]), data, cfg.gravity)
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
