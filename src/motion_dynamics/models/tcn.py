from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def motion_features(
    x: torch.Tensor,
    mask: torch.Tensor | None = None,
    fps: int = 30,
) -> torch.Tensor:
    """Build per-frame root-relative position and velocity features."""
    velocity = torch.zeros_like(x)
    velocity[:, 1:] = (x[:, 1:] - x[:, :-1]) * fps
    velocity[:, 0] = velocity[:, 1]

    if mask is not None:
        velocity_mask = mask.clone()
        velocity_mask[:, 1:] = mask[:, 1:] * mask[:, :-1]
        x = x * mask[..., None]
        velocity = velocity * velocity_mask[..., None]

    return torch.cat([x, velocity], dim=-1).flatten(2)


class TemporalBlock(nn.Module):
    """Residual temporal block used by the original training runs."""

    def __init__(self, channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.c = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=dilation * (kernel_size // 2),
            dilation=dilation,
        )
        self.n = nn.GroupNorm(8, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + F.gelu(self.n(self.c(x)))


class DynamicsTCN(nn.Module):
    """Predict bilateral GRF and lower-limb joint moments from 3D motion."""

    def __init__(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        hidden_dim: int = 256,
        temporal: bool = True,
        fps: int = 30,
    ):
        super().__init__()
        self.fps = fps
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.inp = nn.Conv1d(22 * 6, hidden_dim, 1)
        self.body = nn.Sequential(
            *[
                TemporalBlock(
                    hidden_dim,
                    3 if temporal else 1,
                    dilation if temporal else 1,
                )
                for dilation in (1, 2, 4, 8)
            ]
        )
        self.head = nn.Conv1d(hidden_dim, 24, 1)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        features = motion_features(x, mask, fps=self.fps).transpose(1, 2)
        standardized = self.head(self.body(self.inp(features))).transpose(1, 2)
        return standardized * self.std + self.mean


class ContactTCN(nn.Module):
    """Four-channel temporal contact predictor used in the contact experiment."""

    def __init__(self, hidden_dim: int = 256, fps: int = 30):
        super().__init__()
        self.fps = fps
        self.inp = nn.Conv1d(22 * 6, hidden_dim, 1)
        self.body = nn.Sequential(
            *[TemporalBlock(hidden_dim, 3, dilation) for dilation in (1, 2, 4, 8)]
        )
        self.head = nn.Conv1d(hidden_dim, 4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = motion_features(x, fps=self.fps).transpose(1, 2)
        return self.head(self.body(self.inp(features))).transpose(1, 2)


class DesignTCN(nn.Module):
    """Architecture variants used for force and torque coupling ablations."""

    def __init__(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        design: str,
        hidden_dim: int = 256,
        fps: int = 30,
    ):
        super().__init__()
        valid = {
            "torque only",
            "shared heads",
            "cascade",
            "measured force upper bound",
        }
        if design not in valid:
            raise ValueError(f"Unknown design {design}")

        self.design = design
        self.fps = fps
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.inp = nn.Conv1d(22 * 6, hidden_dim, 1)
        self.body = nn.Sequential(
            *[TemporalBlock(hidden_dim, 3, dilation) for dilation in (1, 2, 4, 8)]
        )
        self.grf = nn.Conv1d(hidden_dim, 6, 1)

        torque_input = hidden_dim
        if design in {"cascade", "measured force upper bound"}:
            torque_input += 6
        self.torque = nn.Sequential(
            nn.Conv1d(torque_input, hidden_dim, 1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, 18, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        grf_true: torch.Tensor | None = None,
    ) -> torch.Tensor:
        features = motion_features(x, fps=self.fps).transpose(1, 2)
        hidden = self.body(self.inp(features))
        force_standardized = self.grf(hidden)

        if self.design == "measured force upper bound":
            if grf_true is None:
                raise ValueError("grf_true is required for measured-force upper bound")
            force_standardized = (
                (grf_true - self.mean[:6]) / self.std[:6]
            ).transpose(1, 2)

        if self.design in {"cascade", "measured force upper bound"}:
            torque_input = torch.cat([hidden, force_standardized], dim=1)
        else:
            torque_input = hidden

        torque_standardized = self.torque(torque_input)
        standardized = torch.cat(
            [force_standardized, torque_standardized],
            dim=1,
        ).transpose(1, 2)
        return standardized * self.std + self.mean
