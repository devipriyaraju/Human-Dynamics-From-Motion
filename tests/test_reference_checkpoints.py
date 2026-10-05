from pathlib import Path

import torch

from motion_dynamics.models.checkpoints import load_contact_weights, load_raw_dynamics_weights


ROOT = Path(__file__).resolve().parents[1]


def test_reference_dynamics_weights_load() -> None:
    path = ROOT / "checkpoints" / "reference" / "dyn_tcn.pt"
    mean = torch.zeros(24)
    std = torch.ones(24)
    model = load_raw_dynamics_weights(path, mean, std)
    assert model.head.out_channels == 24


def test_reference_contact_weights_load() -> None:
    path = ROOT / "checkpoints" / "reference" / "tcn_full_novis.pt"
    model = load_contact_weights(path)
    assert model.head.out_channels == 4
