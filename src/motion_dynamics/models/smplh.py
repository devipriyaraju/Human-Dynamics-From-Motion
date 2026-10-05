from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    angle = axis_angle.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    x, y, z = (axis_angle / angle).unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack(
        [zero, -z, y, z, zero, -x, -y, x, zero],
        dim=-1,
    ).view(*axis_angle.shape[:-1], 3, 3)
    sin = angle.sin()[..., None]
    cos = angle.cos()[..., None]
    eye = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype)
    return eye + sin * skew + (1 - cos) * (skew @ skew)


class SMPLH:
    def __init__(self, path: str | Path, device: torch.device, n_betas: int = 16):
        model = np.load(path, allow_pickle=True)

        def dense(value: np.ndarray) -> np.ndarray:
            if value.dtype == object:
                return np.asarray(value.item().todense())
            return np.asarray(value)

        def tensor(value: np.ndarray) -> torch.Tensor:
            return torch.tensor(dense(value), dtype=torch.float32, device=device)

        self.device = device
        self.v_template = tensor(model["v_template"])
        self.shapedirs = tensor(model["shapedirs"])[..., :n_betas]
        self.posedirs = tensor(model["posedirs"])
        self.j_regressor = tensor(model["J_regressor"])
        self.weights = tensor(model["weights"])
        parents = model["kintree_table"][0].astype(np.int64)
        parents[0] = -1
        self.parents = parents.tolist()
        self.num_joints = len(self.parents)

        assert self.posedirs.shape[-1] == (self.num_joints - 1) * 9
        assert self.j_regressor.shape == (self.num_joints, len(self.v_template))
        assert self.weights.shape == (len(self.v_template), self.num_joints)
        assert all(parent < idx for idx, parent in enumerate(self.parents))

    def __call__(
        self,
        poses: torch.Tensor,
        betas: torch.Tensor,
        translation: torch.Tensor,
        vertex_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = len(poses)
        num_betas = min(len(betas), self.shapedirs.shape[-1])
        v_shaped = self.v_template + self.shapedirs[..., :num_betas] @ betas[:num_betas]
        joints_rest = self.j_regressor @ v_shaped
        rotations = axis_angle_to_matrix(poses.view(batch, self.num_joints, 3))

        global_rotations = [rotations[:, 0]]
        joint_positions = [joints_rest[0].expand(batch, 3)]
        for joint in range(1, self.num_joints):
            parent = self.parents[joint]
            global_rotations.append(global_rotations[parent] @ rotations[:, joint])
            offset = joints_rest[joint] - joints_rest[parent]
            position = joint_positions[parent] + (
                global_rotations[parent] @ offset[:, None]
            ).squeeze(-1)
            joint_positions.append(position)

        global_rotations = torch.stack(global_rotations, dim=1)
        joint_positions = torch.stack(joint_positions, dim=1)
        if vertex_indices is None:
            vertex_indices = torch.arange(len(self.v_template), device=self.device)

        eye = torch.eye(3, device=self.device, dtype=poses.dtype)
        pose_features = (rotations[:, 1:] - eye).reshape(batch, -1)
        v_posed = v_shaped[vertex_indices]
        v_posed = v_posed + (
            self.posedirs[vertex_indices] @ pose_features.T
        ).permute(2, 0, 1)
        translations = joint_positions - (
            global_rotations @ joints_rest[None, :, :, None]
        ).squeeze(-1)
        weights = self.weights[vertex_indices]
        vertex_rotations = torch.einsum("vj,bjmn->bvmn", weights, global_rotations)
        vertex_translations = torch.einsum("vj,bjm->bvm", weights, translations)
        vertices = (
            vertex_rotations @ v_posed[..., None]
        ).squeeze(-1) + vertex_translations
        return joint_positions + translation[:, None], vertices + translation[:, None]


def load_models(directory: str | Path, device: torch.device) -> dict[str, SMPLH]:
    directory = Path(directory)
    paths = sorted(directory.glob("**/model.npz"))
    if not paths:
        raise FileNotFoundError(f"No SMPL-H model.npz files found under {directory}")
    return {path.parent.name: SMPLH(path, device=device) for path in paths}


def sole_patch_indices(model: SMPLH, points_per_patch: int = 20) -> torch.Tensor:
    joints = model.j_regressor @ model.v_template
    dominant_joint = model.weights.argmax(dim=1)
    vertical = model.v_template[:, 1]
    forward = model.v_template[:, 2]
    patches: list[torch.Tensor] = []

    for ankle, toe in [(7, 10), (8, 11)]:
        heel = torch.where((dominant_joint == ankle) & (forward < joints[ankle, 2]))[0]
        forefoot = torch.where(dominant_joint == toe)[0]
        for candidates in (heel, forefoot):
            if len(candidates) < points_per_patch:
                raise ValueError("Too few candidate vertices for a sole patch")
            patches.append(candidates[vertical[candidates].argsort()[:points_per_patch]])
    return torch.cat(patches)
