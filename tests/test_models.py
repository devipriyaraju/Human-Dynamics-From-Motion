import torch

from motion_dynamics.models.tcn import DynamicsTCN


def test_tcn_output_shape():
    mean = torch.zeros(24)
    std = torch.ones(24)
    model = DynamicsTCN(mean, std, hidden_dim=32, temporal=True, fps=30)
    x = torch.randn(2, 64, 22, 3)
    y = model(x)
    assert y.shape == (2, 64, 24)
