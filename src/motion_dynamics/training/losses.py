from __future__ import annotations

import torch
import torch.nn.functional as F


def supervised_dynamics_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    per_channel = F.smooth_l1_loss(
        prediction / std,
        target / std,
        reduction="none",
    )
    per_frame = per_channel.mean(dim=-1)
    return (per_frame * valid).sum() / valid.sum().clamp_min(1)


def force_balance_residual(
    prediction: torch.Tensor,
    com_acceleration: torch.Tensor,
    gravity: float = 9.81,
) -> torch.Tensor:
    gravity_vector = torch.tensor(
        [0.0, 0.0, -gravity],
        device=prediction.device,
        dtype=prediction.dtype,
    )
    predicted_total_force = prediction[..., :6].reshape(*prediction.shape[:-1], 2, 3).sum(dim=-2)
    target_total_force_bw = (com_acceleration - gravity_vector) / gravity
    return predicted_total_force - target_total_force_bw


def physics_aware_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    std: torch.Tensor,
    com_acceleration: torch.Tensor,
    gravity: float,
    force_balance_weight: float,
) -> torch.Tensor:
    loss = supervised_dynamics_loss(prediction, target, valid, std)
    if force_balance_weight > 0:
        residual = force_balance_residual(prediction, com_acceleration, gravity)
        loss = loss + force_balance_weight * residual.pow(2).sum(dim=-1).mean()
    return loss
