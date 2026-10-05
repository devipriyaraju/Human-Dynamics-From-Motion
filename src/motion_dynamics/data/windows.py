from __future__ import annotations

import numpy as np
import torch


def make_windows(trial: dict, window: int, stride: int) -> dict[str, torch.Tensor] | None:
    frames = len(trial["rel"])
    starts = list(range(0, frames - window + 1, stride))
    if not starts:
        return None
    targets = np.concatenate(
        [trial["grf"].reshape(frames, 6), trial["tau"].reshape(frames, 18)],
        axis=1,
    )

    def stack(values: np.ndarray) -> torch.Tensor:
        return torch.tensor(np.stack([values[start : start + window] for start in starts]))

    return {
        "X": stack(trial["rel"]),
        "Y": stack(targets),
        "V": stack(trial["valid"]).float(),
        "ML": stack(trial["ml"]),
        "A": stack(trial["acom"]),
    }


def group_windows_by_subject(
    trials: list[dict],
    window: int,
    stride: int,
    device: torch.device,
    exclude_motion: str | None = "dog",
) -> dict[str, dict[str, torch.Tensor]]:
    subjects = sorted({trial["subj"] for trial in trials})
    output = {}
    for subject in subjects:
        windows = []
        for trial in trials:
            if trial["subj"] != subject:
                continue
            if exclude_motion and trial["motion"] == exclude_motion:
                continue
            item = make_windows(trial, window, stride)
            if item is not None:
                windows.append(item)
        if not windows:
            continue
        output[subject] = {
            key: torch.cat([item[key] for item in windows]).to(device)
            for key in windows[0]
        }
    return output


def combine_subjects(
    grouped: dict[str, dict[str, torch.Tensor]],
    subjects: list[str],
) -> dict[str, torch.Tensor]:
    keys = next(iter(grouped.values())).keys()
    return {key: torch.cat([grouped[subject][key] for subject in subjects]) for key in keys}


def target_statistics(data: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    mask = data["V"][..., None]
    mean = (data["Y"] * mask).sum(dim=(0, 1)) / mask.sum()
    variance = (((data["Y"] - mean) ** 2) * mask).sum(dim=(0, 1)) / mask.sum()
    return mean, variance.sqrt().clamp_min(1e-6)
