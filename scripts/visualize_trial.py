from __future__ import annotations

import argparse

import torch

from motion_dynamics.config import load_config
from motion_dynamics.models.tcn import DynamicsTCN
from motion_dynamics.visualization.trial import plot_prediction_trial


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--subject", default=None)
    parser.add_argument("--output", default="results/figures/prediction_trial_new.png")
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = torch.load(cfg.paths.processed_cache, map_location="cpu", weights_only=False)
    subject = args.subject or cfg.test_subject
    trial = next(
        item
        for item in cache["trials"]
        if item["subj"] == subject and len(item["rel"]) >= cfg.window
    )
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
    x = torch.tensor(trial["rel"], device=device)[None]
    with torch.no_grad():
        prediction = model(x)[0].cpu().numpy()
    plot_prediction_trial(trial, prediction, cfg.fps, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
