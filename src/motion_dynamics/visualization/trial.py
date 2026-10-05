from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from motion_dynamics.physics.inverse_dynamics import sagittal_component


def plot_prediction_trial(
    trial: dict,
    prediction: np.ndarray,
    fps: int,
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    truth_torque = sagittal_component(trial["tau"], trial["ml"])
    pred_torque = sagittal_component(
        prediction[:, 6:].reshape(-1, 2, 3, 3),
        trial["ml"],
    )
    time = np.arange(len(trial["ml"])) / fps
    figure, axes = plt.subplots(1, 4, figsize=(18, 3.4))

    for leg in range(2):
        axes[0].plot(time, trial["grf"][:, leg, 2], label=f"{'LR'[leg]} measured")
        axes[0].plot(
            time,
            prediction[:, :6].reshape(-1, 2, 3)[:, leg, 2],
            linestyle="--",
            label=f"{'LR'[leg]} predicted",
        )
    axes[0].set_title("vertical GRF, body weights")

    for index, joint in enumerate(["ankle", "knee", "hip"]):
        axes[index + 1].plot(time, truth_torque[:, 0, index], label="inverse dynamics")
        axes[index + 1].plot(
            time,
            pred_torque[:, 0, index],
            linestyle="--",
            label="predicted",
        )
        axes[index + 1].set_title(f"left {joint} sagittal torque, N m/kg")

    for axis in axes:
        low, high = axis.get_ylim()
        axis.fill_between(time, low, high, where=~trial["valid"], alpha=0.12)
        axis.set_xlabel("time, s")
        axis.legend(fontsize=7)

    figure.suptitle(f"{trial['name']} held-out subject")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
