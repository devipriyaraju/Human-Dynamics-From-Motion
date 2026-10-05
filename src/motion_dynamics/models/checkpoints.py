from __future__ import annotations

from pathlib import Path

import torch

from motion_dynamics.models.tcn import ContactTCN, DynamicsTCN


def load_raw_dynamics_weights(
    path: str | Path,
    mean: torch.Tensor,
    std: torch.Tensor,
    hidden_dim: int = 256,
    fps: int = 30,
    temporal: bool = True,
    device: torch.device | str = "cpu",
) -> DynamicsTCN:
    """Load an original raw state_dict using externally computed target scaling."""
    state = torch.load(path, map_location=device, weights_only=True)
    model = DynamicsTCN(
        mean.to(device),
        std.to(device),
        hidden_dim=hidden_dim,
        temporal=temporal,
        fps=fps,
    ).to(device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed_missing = {"mean", "std"}
    if set(missing) - allowed_missing or unexpected:
        raise RuntimeError(
            f"Checkpoint mismatch, missing={missing}, unexpected={unexpected}"
        )
    return model.eval()


def load_contact_weights(
    path: str | Path,
    hidden_dim: int = 256,
    fps: int = 30,
    device: torch.device | str = "cpu",
) -> ContactTCN:
    """Load the temporal contact model state_dict."""
    state = torch.load(path, map_location=device, weights_only=True)
    model = ContactTCN(hidden_dim=hidden_dim, fps=fps).to(device)
    model.load_state_dict(state)
    return model.eval()
