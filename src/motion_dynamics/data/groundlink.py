from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch

from motion_dynamics.models.smplh import SMPLH, axis_angle_to_matrix, sole_patch_indices
from motion_dynamics.physics.inverse_dynamics import inverse_dynamics


def parse_gender(metadata: np.lib.npyio.NpzFile) -> str:
    value = metadata["gender"]
    value = value.item() if value.shape == () else value
    if isinstance(value, bytes):
        return value.decode()
    return str(value)


def pair_groundlink_trials(force_dir: str | Path, motion_dir: str | Path) -> list[tuple[Path, Path]]:
    force_dir = Path(force_dir)
    motion_dir = Path(motion_dir)
    force_files = sorted(force_dir.glob("**/*.npy"))
    motion_files = {
        path.name.removesuffix("_stageii.npz"): path
        for path in motion_dir.glob("**/*_stageii.npz")
    }
    pairs = []
    for force_path in force_files:
        key = force_path.stem
        if key in motion_files:
            pairs.append((force_path, motion_files[key]))
    return pairs


def _to_numpy(value) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


@torch.no_grad()
def smplh_kinematics(
    metadata: np.lib.npyio.NpzFile,
    models: dict[str, SMPLH],
    patch: torch.Tensor,
    device: torch.device,
    points_per_patch: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model = models.get(parse_gender(metadata), next(iter(models.values())))
    frames = len(metadata["poses"])
    pose = torch.zeros(frames, model.num_joints * 3, device=device)
    pose[:, :66] = torch.tensor(metadata["poses"][:, :66], dtype=torch.float32, device=device)
    rest_joints = model.j_regressor @ model.v_template
    joint_chunks = []
    contact_chunks = []
    rotation_chunks = []

    for start in range(0, frames, 4096):
        local_pose = pose[start : start + 4096]
        rotations = axis_angle_to_matrix(local_pose.view(-1, model.num_joints, 3))
        global_rotations = [rotations[:, 0]]
        positions = [torch.zeros(len(rotations), 3, device=device)]
        for joint in range(1, 22):
            parent = model.parents[joint]
            global_rotations.append(global_rotations[parent] @ rotations[:, joint])
            offset = rest_joints[joint] - rest_joints[parent]
            positions.append(
                positions[parent]
                + (global_rotations[parent] @ offset[:, None]).squeeze(-1)
            )
        joints, vertices = model(
            local_pose,
            torch.zeros(16, device=device),
            torch.zeros(len(local_pose), 3, device=device),
            patch,
        )
        joint_chunks.append(torch.stack(positions, dim=1))
        contact_chunks.append(
            vertices.view(-1, 4, points_per_patch, 3).mean(dim=2) - joints[:, :1]
        )
        rotation_chunks.append(
            torch.stack(
                [global_rotations[index] for index in (1, 4, 7, 2, 5, 8)],
                dim=1,
            )
        )

    return tuple(
        torch.cat(chunks).double().cpu().numpy()
        for chunks in (joint_chunks, contact_chunks, rotation_chunks)
    )


def load_raw_trial(
    force_path: Path,
    motion_path: Path,
    models: dict[str, SMPLH],
    patch: torch.Tensor,
    device: torch.device,
) -> dict:
    name = force_path.stem
    force_data = np.load(force_path, allow_pickle=True).item()
    motion_data = np.load(motion_path, allow_pickle=True)
    grf = np.nan_to_num(_to_numpy(force_data["GRF"]).astype(np.float64))
    cop = np.nan_to_num(_to_numpy(force_data["CoP"]).astype(np.float64))
    if grf.ndim != 3 or grf.shape[1:] != (2, 3):
        raise ValueError(f"Unexpected GRF layout {grf.shape}")
    if cop.shape[1:] != (2, 3):
        raise ValueError(f"Unexpected CoP layout {cop.shape}")
    joints, contact, rotations = smplh_kinematics(
        motion_data,
        models,
        patch,
        device,
    )
    subject_match = re.search(r"s\d{3}", name)
    if subject_match is None:
        raise ValueError(f"Could not parse subject from {name}")
    return {
        "name": name,
        "subj": subject_match.group(),
        "motion": name[14:-2],
        "fps": float(motion_data["mocap_framerate"]),
        "trans": motion_data["trans"].astype(np.float64),
        "joints": joints,
        "contact_points": contact,
        "segment_rotations": rotations,
        "grf_full": grf,
        "cop_full": cop,
    }


def cut_trial(trial: dict, offset: int = 0) -> dict:
    length = min(len(trial["trans"]), len(trial["grf_full"]) - offset)
    return {
        **trial,
        "trans": trial["trans"][:length],
        "joints": trial["joints"][:length],
        "contact_points": trial["contact_points"][:length],
        "segment_rotations": trial["segment_rotations"][:length],
        "grf": trial["grf_full"][offset : offset + length],
        "cop": trial["cop_full"][offset : offset + length],
    }


def calibrate_subject(trials: list[dict]) -> dict:
    total_force = np.concatenate([trial["grf"][:, :, 2].sum(axis=1) for trial in trials])
    threshold = 0.2 * np.percentile(total_force, 75)
    body_weight = np.median(total_force[total_force > threshold])
    best = None

    for assignment in ([0, 1], [1, 0]):
        foot_height_terms = []
        translation_z = []
        plate_z = []
        for trial in trials:
            for foot in range(2):
                loaded = trial["grf"][:, assignment[foot], 2] > 0.3 * body_weight
                contact_points = trial["contact_points"][loaded, 2 * foot : 2 * foot + 2]
                foot_height_terms.append(-contact_points[:, :, 2].min(axis=1))
                translation_z.append(trial["trans"][loaded, 2])
                plate_z.append(trial["cop"][loaded, assignment[foot], 2])

        foot_height_terms = np.concatenate(foot_height_terms)
        translation_z = np.concatenate(translation_z)
        plate_z_value = np.median(np.concatenate(plate_z))
        if foot_height_terms.std() > 0.02:
            scale, _ = np.polyfit(foot_height_terms, translation_z, 1)
        else:
            scale = 1.0
        if not 0.85 < scale < 1.2:
            scale = 1.0
        offset_z = plate_z_value - np.median(translation_z - scale * foot_height_terms)

        horizontal_residuals = []
        for trial in trials:
            for foot in range(2):
                loaded = trial["grf"][:, assignment[foot], 2] > 0.3 * body_weight
                foot_center = trial["contact_points"][loaded, 2 * foot : 2 * foot + 2, :2].mean(axis=1)
                predicted = trial["trans"][loaded, :2] + scale * foot_center
                horizontal_residuals.append(
                    trial["cop"][loaded, assignment[foot], :2] - predicted
                )
        horizontal_residuals = np.concatenate(horizontal_residuals)
        offset_xy = np.median(horizontal_residuals, axis=0)
        cop_error = np.median(
            np.linalg.norm(horizontal_residuals - offset_xy, axis=1)
        )
        candidate = {
            "bw": body_weight,
            "assign": assignment,
            "scale": float(scale),
            "off": np.array([offset_xy[0], offset_xy[1], offset_z]),
            "cop_err": float(cop_error),
        }
        if best is None or candidate["cop_err"] < best["cop_err"]:
            best = candidate
    return best


def cop_residual(trial: dict, calibration: dict, offset: int) -> tuple[float, list[int]]:
    aligned = cut_trial(trial, offset)
    best_error = np.inf
    best_assignment = calibration["assign"]
    if len(aligned["trans"]) < trial["fps"]:
        return best_error, best_assignment

    for assignment in ([0, 1], [1, 0]):
        distances = []
        for foot in range(2):
            loaded = aligned["grf"][:, assignment[foot], 2] > 0.3 * calibration["bw"]
            foot_center = aligned["contact_points"][
                loaded,
                2 * foot : 2 * foot + 2,
                :2,
            ].mean(axis=1)
            predicted = (
                aligned["trans"][loaded, :2]
                + calibration["off"][:2]
                + calibration["scale"] * foot_center
            )
            distances.append(
                np.linalg.norm(
                    aligned["cop"][loaded, assignment[foot], :2] - predicted,
                    axis=1,
                )
            )
        distances = np.concatenate(distances)
        if len(distances) > 10 and np.median(distances) < best_error:
            best_error = float(np.median(distances))
            best_assignment = assignment
    return best_error, best_assignment


def align_trial(trial: dict, calibration: dict, step: int = 2) -> tuple[int, float, list[int]]:
    error_zero, assignment_zero = cop_residual(trial, calibration, 0)
    if error_zero <= 0.10:
        return 0, error_zero, assignment_zero
    max_offset = max(1, len(trial["grf_full"]) - len(trial["trans"]) + 1)
    candidates = []
    for offset in range(0, max_offset, step):
        error, assignment = cop_residual(trial, calibration, offset)
        candidates.append((error, assignment, offset))
    error, assignment, offset = min(candidates, key=lambda item: item[0])
    return offset, error, assignment


def prepare_groundlink(
    force_dir: str | Path,
    motion_dir: str | Path,
    models: dict[str, SMPLH],
    device: torch.device,
    target_fps: int,
    cutoff_hz: float,
    gravity: float,
    residual_fraction_g: float,
) -> tuple[list[dict], dict[str, dict]]:
    reference = models.get("neutral", next(iter(models.values())))
    patch = sole_patch_indices(reference)
    raw_by_subject: dict[str, list[dict]] = {}

    for force_path, motion_path in pair_groundlink_trials(force_dir, motion_dir):
        motion_name = force_path.stem[14:-2]
        if "ballethigh" in motion_name or "ballet_high" in motion_name:
            continue
        if force_path.stem.startswith("s001") and motion_name.startswith("idling"):
            continue
        trial = load_raw_trial(force_path, motion_path, models, patch, device)
        raw_by_subject.setdefault(trial["subj"], []).append(trial)

    cop_values = np.concatenate(
        [
            trial["cop_full"][trial["grf_full"][:, :, 2] > 0][:, :2]
            for trials in raw_by_subject.values()
            for trial in trials
        ]
    )
    if np.median(np.abs(cop_values)) > 20:
        for trials in raw_by_subject.values():
            for trial in trials:
                trial["cop_full"] = trial["cop_full"] / 1000

    calibrations = {}
    for subject, trials in raw_by_subject.items():
        stationary = [
            cut_trial(trial)
            for trial in trials
            if "walk" not in trial["motion"] and "soccer" not in trial["motion"]
        ]
        calibrations[subject] = calibrate_subject(stationary or [cut_trial(t) for t in trials])

    processed = []
    for subject, trials in raw_by_subject.items():
        for trial in trials:
            offset, residual, assignment = align_trial(trial, calibrations[subject])
            if residual > 0.10:
                continue
            calibration = {**calibrations[subject], "assign": assignment}
            processed.append(
                inverse_dynamics(
                    cut_trial(trial, offset),
                    calibration,
                    target_fps=target_fps,
                    cutoff_hz=cutoff_hz,
                    gravity=gravity,
                    residual_fraction_g=residual_fraction_g,
                )
            )
    return processed, calibrations
