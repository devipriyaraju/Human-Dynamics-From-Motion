from __future__ import annotations

import pandas as pd
import torch

from motion_dynamics.data.windows import combine_subjects, target_statistics
from motion_dynamics.evaluation.metrics import dynamics_metrics
from motion_dynamics.models.tcn import DynamicsTCN
from motion_dynamics.training.augment import corrupt_motion
from motion_dynamics.training.engine import TrainOptions, train_model
from motion_dynamics.training.losses import force_balance_residual


def evaluate_prediction(
    prediction: torch.Tensor,
    data: dict[str, torch.Tensor],
    gravity: float,
) -> dict[str, float]:
    result = dynamics_metrics(prediction, data["Y"], data["V"], data["ML"])
    result["force-balance error (BW)"] = (
        force_balance_residual(prediction, data["A"], gravity)
        .pow(2)
        .sum(dim=-1)
        .mean()
        .sqrt()
        .item()
    )
    return result


def run_loso(
    grouped: dict[str, dict[str, torch.Tensor]],
    hidden_dim: int,
    fps: int,
    base_options: TrainOptions,
) -> pd.DataFrame:
    subjects = sorted(grouped)
    rows = {}
    variants = {
        "single frame": {"temporal": False, "force_balance_weight": 0.0, "corruption": False},
        "temporal": {"temporal": True, "force_balance_weight": 0.0, "corruption": False},
        "temporal + force-balance loss": {"temporal": True, "force_balance_weight": 1.0, "corruption": False},
        "temporal, corruption-trained": {"temporal": True, "force_balance_weight": 0.0, "corruption": True},
    }

    for held_out in subjects:
        train_subjects = [subject for subject in subjects if subject != held_out]
        train_data = combine_subjects(grouped, train_subjects)
        test_data = grouped[held_out]
        mean, std = target_statistics(train_data)
        rows[(held_out, "clean", "training mean")] = evaluate_prediction(
            mean.expand_as(test_data["Y"]),
            test_data,
            base_options.gravity,
        )

        for name, variant in variants.items():
            model = DynamicsTCN(
                mean,
                std,
                hidden_dim=hidden_dim,
                temporal=variant["temporal"],
                fps=fps,
            ).to(train_data["X"].device)
            options = TrainOptions(
                **{
                    **base_options.__dict__,
                    "force_balance_weight": variant["force_balance_weight"],
                    "corruption_training": variant["corruption"],
                }
            )
            model = train_model(model, train_data, std, options)
            with torch.no_grad():
                rows[(held_out, "clean", name)] = evaluate_prediction(
                    model(test_data["X"]),
                    test_data,
                    base_options.gravity,
                )
                if name in {"temporal", "temporal, corruption-trained"}:
                    for condition in ("both feet", "lower legs", "25 mm noise"):
                        torch.manual_seed(123)
                        x, mask = corrupt_motion(test_data["X"], condition)
                        rows[(held_out, condition, name)] = evaluate_prediction(
                            model(x, mask),
                            test_data,
                            base_options.gravity,
                        )

    frame = pd.DataFrame(rows).T.astype(float)
    frame.index.names = ["subject", "condition", "model"]
    return frame
