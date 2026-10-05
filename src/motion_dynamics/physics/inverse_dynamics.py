from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt

SEGMENTS = [
    ((15, 12), 0.081),
    ((0, 12), 0.497),
    ((16, 18), 0.028),
    ((17, 19), 0.028),
    ((18, 20), 0.016),
    ((19, 21), 0.016),
    ((20, 20), 0.006),
    ((21, 21), 0.006),
    ((1, 4), 0.100),
    ((2, 5), 0.100),
    ((4, 7), 0.0465),
    ((5, 8), 0.0465),
    ((7, 10), 0.0145),
    ((8, 11), 0.0145),
]
LEGS = [(1, 4, 7, 10), (2, 5, 8, 11)]
MASS_FRACTIONS = (0.100, 0.0465, 0.0145)
COM_FRACTIONS = (0.433, 0.433, 0.5)
RADIUS_GYRATION = (0.323, 0.302, 0.475)


def lowpass(values: np.ndarray, fps: float, cutoff_hz: float) -> np.ndarray:
    b, a = butter(4, cutoff_hz / (fps / 2))
    return filtfilt(b, a, values, axis=0)


def derivative(values: np.ndarray, fps: float) -> np.ndarray:
    return np.gradient(values, 1.0 / fps, axis=0)


def angular_velocity(rotation: np.ndarray, fps: float) -> np.ndarray:
    omega_matrix = derivative(rotation, fps) @ rotation.transpose(0, 2, 1)
    return np.stack(
        [
            omega_matrix[:, 2, 1] - omega_matrix[:, 1, 2],
            omega_matrix[:, 0, 2] - omega_matrix[:, 2, 0],
            omega_matrix[:, 1, 0] - omega_matrix[:, 0, 1],
        ],
        axis=1,
    ) / 2


def sagittal_component(torque: np.ndarray, mediolateral_axis: np.ndarray) -> np.ndarray:
    return (torque * mediolateral_axis[:, None, None]).sum(axis=-1)


def inverse_dynamics(
    trial: dict,
    calibration: dict,
    target_fps: int = 30,
    cutoff_hz: float = 10.0,
    gravity: float = 9.81,
    residual_fraction_g: float = 0.25,
    contact_threshold_bw: float = 0.05,
) -> dict:
    fps = float(trial["fps"])
    scale = float(calibration["scale"])
    gravity_vector = np.array([0.0, 0.0, -gravity])
    pelvis = lowpass(trial["trans"] + calibration["off"], fps, cutoff_hz)
    joints_world = pelvis[:, None] + scale * lowpass(trial["joints"], fps, cutoff_hz)
    force_bw = lowpass(
        trial["grf"][:, calibration["assign"]] / calibration["bw"],
        fps,
        cutoff_hz,
    )
    force_accel = force_bw * gravity
    center_of_pressure = trial["cop"][:, calibration["assign"]]

    com_world = sum(
        mass * 0.5 * (joints_world[:, a] + joints_world[:, b])
        for (a, b), mass in SEGMENTS
    )
    com_accel = derivative(derivative(com_world, fps), fps)
    residual = np.linalg.norm(
        force_accel.sum(axis=1) - (com_accel - gravity_vector),
        axis=1,
    )
    valid = residual < residual_fraction_g * gravity

    torque = np.zeros((len(pelvis), 2, 3, 3), dtype=np.float64)
    rotations = trial["segment_rotations"]
    for leg_index, (hip, knee, ankle, toe) in enumerate(LEGS):
        proximal = [joints_world[:, hip], joints_world[:, knee], joints_world[:, ankle]]
        distal = [joints_world[:, knee], joints_world[:, ankle], joints_world[:, toe]]
        foot_contact = force_bw[:, leg_index, 2:3] > contact_threshold_bw
        contact_point = np.where(foot_contact, center_of_pressure[:, leg_index], joints_world[:, ankle])
        distal_force = -force_accel[:, leg_index]
        distal_moment = np.zeros((len(pelvis), 3), dtype=np.float64)
        distal_point = contact_point

        for segment_index in (2, 1, 0):
            length = np.median(
                np.linalg.norm(distal[segment_index] - proximal[segment_index], axis=1)
            )
            center = proximal[segment_index] + COM_FRACTIONS[segment_index] * (
                distal[segment_index] - proximal[segment_index]
            )
            omega = angular_velocity(
                rotations[:, 3 * leg_index + segment_index],
                fps,
            )
            alpha = derivative(lowpass(omega, fps, cutoff_hz), fps)
            inertia = MASS_FRACTIONS[segment_index] * (
                RADIUS_GYRATION[segment_index] * length
            ) ** 2
            proximal_force = MASS_FRACTIONS[segment_index] * (
                derivative(derivative(center, fps), fps) - gravity_vector
            ) + distal_force
            proximal_moment = (
                inertia * alpha
                + distal_moment
                + np.cross(distal_point - center, distal_force)
                - np.cross(proximal[segment_index] - center, proximal_force)
            )
            torque[:, leg_index, 2 - segment_index] = proximal_moment
            distal_force = proximal_force
            distal_moment = proximal_moment
            distal_point = proximal[segment_index]

    mediolateral = joints_world[:, 1] - joints_world[:, 2]
    mediolateral[:, 2] = 0
    mediolateral /= np.linalg.norm(mediolateral, axis=1, keepdims=True) + 1e-9
    indices = np.round(np.arange(0, len(pelvis) - 1, fps / target_fps)).astype(int)
    return {
        "rel": (joints_world - joints_world[:, :1])[indices].astype(np.float32),
        "grf": force_bw[indices].astype(np.float32),
        "tau": torque[indices].astype(np.float32),
        "valid": valid[indices],
        "ml": mediolateral[indices].astype(np.float32),
        "resid": residual[indices],
        "acom": com_accel[indices].astype(np.float32),
        "name": trial["name"],
        "subj": trial["subj"],
        "motion": trial["motion"],
    }
