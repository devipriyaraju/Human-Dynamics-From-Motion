import torch

from motion_dynamics.training.losses import force_balance_residual


def test_force_balance_static_body():
    prediction = torch.zeros(1, 1, 24)
    prediction[..., 2] = 0.5
    prediction[..., 5] = 0.5
    acceleration = torch.zeros(1, 1, 3)
    residual = force_balance_residual(prediction, acceleration, gravity=9.81)
    assert torch.allclose(residual, torch.zeros_like(residual), atol=1e-6)
