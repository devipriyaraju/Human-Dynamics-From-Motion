from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise TypeError("Expected a PyTorch state_dict")

    print(f"checkpoint: {args.checkpoint}")
    print(f"tensors: {len(state)}")
    for name, value in state.items():
        if torch.is_tensor(value):
            print(f"{name:24s} {tuple(value.shape)} {value.dtype}")


if __name__ == "__main__":
    main()
