import torch

from motion_dynamics.models.smplh import axis_angle_to_matrix


def test_zero_axis_angle_is_identity():
    rotation = axis_angle_to_matrix(torch.zeros(4, 3))
    expected = torch.eye(3).expand(4, 3, 3)
    assert torch.allclose(rotation, expected, atol=1e-6)
