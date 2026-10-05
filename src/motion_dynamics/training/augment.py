from __future__ import annotations

import math

import torch

OCCLUSION_SETS = {
    "both feet": [7, 8, 10, 11],
    "lower legs": [4, 5, 7, 8, 10, 11],
}


def random_yaw(batch_size: int, device: torch.device) -> torch.Tensor:
    angle = torch.rand(batch_size, device=device) * 2 * math.pi
    cos = angle.cos()
    sin = angle.sin()
    zero = torch.zeros_like(cos)
    one = torch.ones_like(cos)
    return torch.stack(
        [cos, -sin, zero, sin, cos, zero, zero, zero, one],
        dim=-1,
    ).view(batch_size, 3, 3)


def rotate(rotation: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bij,bt...j->bt...i", rotation, values)


def corrupt_motion(
    x: torch.Tensor,
    kind: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, frames = x.shape[:2]
    mask = torch.ones(batch, frames, 22, device=x.device)
    if kind is None:
        add_noise = torch.rand(batch, 1, 1, 1, device=x.device) < 0.5
        noise_scale = torch.rand(batch, 1, 1, 1, device=x.device) * 0.05
        x = x + torch.randn_like(x) * noise_scale * add_noise
        random_missing = torch.rand(batch, 1, 1, device=x.device) * 0.5
        activate_missing = torch.rand(batch, 1, 1, device=x.device) < 0.5
        threshold = random_missing * activate_missing
        mask = (torch.rand(batch, frames, 22, device=x.device) >= threshold).float()
        for joints in OCCLUSION_SETS.values():
            keep = (torch.rand(batch, 1, 1, device=x.device) >= 0.25).float()
            mask[:, :, joints] *= keep
    elif kind in OCCLUSION_SETS:
        mask[:, :, OCCLUSION_SETS[kind]] = 0
    elif kind == "30% joints missing":
        mask = (torch.rand(batch, frames, 22, device=x.device) >= 0.3).float()
    elif kind == "25 mm noise":
        x = x + torch.randn_like(x) * 0.025
    else:
        raise ValueError(f"Unknown corruption kind {kind}")
    return x, mask
