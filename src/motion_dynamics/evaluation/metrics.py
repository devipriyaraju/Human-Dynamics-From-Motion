from __future__ import annotations

import math

import torch


def dynamics_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    mediolateral_axis: torch.Tensor,
) -> dict[str, float]:
    keep = valid > 0.5
    pred = prediction[keep]
    truth = target[keep]
    axis = mediolateral_axis[keep]
    result: dict[str, float] = {}

    pred_force = pred[:, :6].view(-1, 2, 3)
    true_force = truth[:, :6].view(-1, 2, 3)
    result["vertical GRF RMSE (BW)"] = (
        (pred_force[..., 2] - true_force[..., 2]).pow(2).mean().sqrt().item()
    )
    result["3D GRF RMSE (BW)"] = (
        (pred_force - true_force).pow(2).sum(dim=-1).mean().sqrt().item()
    )

    pred_torque = pred[:, 6:].view(-1, 2, 3, 3)
    true_torque = truth[:, 6:].view(-1, 2, 3, 3)
    pred_sagittal = (pred_torque * axis[:, None, None]).sum(dim=-1)
    true_sagittal = (true_torque * axis[:, None, None]).sum(dim=-1)

    for joint_index, joint_name in enumerate(["ankle", "knee", "hip"]):
        a = pred_sagittal[:, :, joint_index].flatten()
        b = true_sagittal[:, :, joint_index].flatten()
        rmse = (a - b).pow(2).mean().sqrt().item()
        value_range = (
            torch.quantile(b, 0.99) - torch.quantile(b, 0.01)
        ).item()
        if a.std() > 0 and b.std() > 0:
            corr = torch.corrcoef(torch.stack([a, b]))[0, 1].item()
        else:
            corr = math.nan
        result[f"{joint_name} RMSE (N m/kg)"] = rmse
        result[f"{joint_name} error / range (%)"] = 100 * rmse / value_range
        result[f"{joint_name} correlation"] = corr
    return result
