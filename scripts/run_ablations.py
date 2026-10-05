from __future__ import annotations

import argparse

import pandas as pd
import torch

from motion_dynamics.config import load_config
from motion_dynamics.data.windows import combine_subjects, group_windows_by_subject, target_statistics
from motion_dynamics.evaluation.metrics import dynamics_metrics
from motion_dynamics.models.tcn import DesignTCN
from motion_dynamics.training.augment import random_yaw, rotate
from motion_dynamics.training.losses import supervised_dynamics_loss


def fit_design(model, data, std, cfg):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    steps = (len(data["X"]) + cfg.batch_size - 1) // cfg.batch_size
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        cfg.learning_rate,
        total_steps=cfg.epochs * steps,
    )
    for _ in range(cfg.epochs):
        permutation = torch.randperm(len(data["X"]), device=data["X"].device)
        for start in range(0, len(permutation), cfg.batch_size):
            batch = permutation[start : start + cfg.batch_size]
            yaw = random_yaw(len(batch), data["X"].device)
            x = rotate(yaw, data["X"][batch])
            target = rotate(
                yaw,
                data["Y"][batch].view(len(batch), cfg.window, 8, 3),
            ).reshape(len(batch), cfg.window, 24)
            prediction = model(x, target[..., :6])
            if model.design in {"torque only", "measured force upper bound"}:
                loss = supervised_dynamics_loss(
                    prediction[..., 6:],
                    target[..., 6:],
                    data["V"][batch],
                    std[6:],
                )
            else:
                loss = supervised_dynamics_loss(
                    prediction,
                    target,
                    data["V"][batch],
                    std,
                )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
    return model.eval()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = torch.load(cfg.paths.processed_cache, map_location="cpu", weights_only=False)
    grouped = group_windows_by_subject(cache["trials"], cfg.window, cfg.stride, device)
    subjects = sorted(grouped)
    designs = ["torque only", "shared heads", "cascade", "measured force upper bound"]
    rows = {}

    for held_out in subjects:
        train = combine_subjects(grouped, [s for s in subjects if s != held_out])
        test = grouped[held_out]
        mean, std = target_statistics(train)
        for design in designs:
            torch.manual_seed(cfg.seed)
            model = DesignTCN(mean, std, design, cfg.hidden_dim, cfg.fps).to(device)
            model = fit_design(model, train, std, cfg)
            with torch.no_grad():
                prediction = model(test["X"], test["Y"][..., :6])
            metrics = dynamics_metrics(prediction, test["Y"], test["V"], test["ML"])
            rows[(held_out, design)] = metrics

    frame = pd.DataFrame(rows).T.astype(float)
    frame.index.names = ["subject", "design"]
    output = cfg.paths.results_dir / "tables" / "design_comparison.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output)
    print(output)


if __name__ == "__main__":
    main()
