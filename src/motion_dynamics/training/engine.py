from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from motion_dynamics.models.tcn import DynamicsTCN
from motion_dynamics.training.augment import corrupt_motion, random_yaw, rotate
from motion_dynamics.training.losses import physics_aware_loss


@dataclass
class TrainOptions:
    epochs: int = 60
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-3
    force_balance_weight: float = 0.0
    corruption_training: bool = False
    gradient_clip: float = 1.0
    seed: int = 0
    gravity: float = 9.81


def train_model(
    model: DynamicsTCN,
    data: dict[str, torch.Tensor],
    std: torch.Tensor,
    options: TrainOptions,
) -> DynamicsTCN:
    torch.manual_seed(options.seed)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=options.learning_rate,
        weight_decay=options.weight_decay,
    )
    steps_per_epoch = (len(data["X"]) + options.batch_size - 1) // options.batch_size
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        options.learning_rate,
        total_steps=options.epochs * steps_per_epoch,
    )

    for _ in range(options.epochs):
        model.train()
        permutation = torch.randperm(len(data["X"]), device=data["X"].device)
        for start in range(0, len(permutation), options.batch_size):
            batch = permutation[start : start + options.batch_size]
            rotation = random_yaw(len(batch), data["X"].device)
            x = rotate(rotation, data["X"][batch])
            target = data["Y"][batch].view(len(batch), data["Y"].shape[1], 8, 3)
            target = rotate(rotation, target).reshape(len(batch), data["Y"].shape[1], 24)
            acceleration = rotate(rotation, data["A"][batch])
            mask = None
            if options.corruption_training:
                x, mask = corrupt_motion(x)
            prediction = model(x, mask)
            loss = physics_aware_loss(
                prediction,
                target,
                data["V"][batch],
                std,
                acceleration,
                options.gravity,
                options.force_balance_weight,
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), options.gradient_clip)
            optimizer.step()
            scheduler.step()
    return model.eval()


def save_checkpoint(
    path: str | Path,
    model: DynamicsTCN,
    metadata: dict,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "mean": model.mean.detach().cpu(),
            "std": model.std.detach().cpu(),
            "metadata": metadata,
        },
        path,
    )
