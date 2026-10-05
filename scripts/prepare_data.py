from __future__ import annotations

import argparse

import torch

from motion_dynamics.config import load_config
from motion_dynamics.data.groundlink import prepare_groundlink
from motion_dynamics.models.smplh import load_models


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = load_models(cfg.paths.smplh_dir, device)
    trials, calibrations = prepare_groundlink(
        cfg.paths.force_dir,
        cfg.paths.motion_dir,
        models,
        device,
        target_fps=cfg.fps,
        cutoff_hz=cfg.lowpass_hz,
        gravity=cfg.gravity,
        residual_fraction_g=cfg.valid_force_residual_fraction_g,
    )
    cfg.paths.processed_cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"trials": trials, "calibrations": calibrations},
        cfg.paths.processed_cache,
    )
    valid = sum(int(trial["valid"].sum()) for trial in trials)
    total = sum(len(trial["valid"]) for trial in trials)
    print(f"saved {len(trials)} trials from {len(calibrations)} subjects")
    print(f"valid inverse-dynamics frames: {100 * valid / total:.1f}%")
    print(cfg.paths.processed_cache)


if __name__ == "__main__":
    main()
