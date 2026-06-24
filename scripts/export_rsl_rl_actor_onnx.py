#!/usr/bin/env python3
"""Export the deterministic actor network from an RSL-RL checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


class Actor(torch.nn.Module):
    def __init__(self, state_dict: dict[str, torch.Tensor]) -> None:
        super().__init__()
        linear_indices = sorted(
            {
                int(key.split(".")[1])
                for key in state_dict
                if key.startswith("actor.") and key.endswith(".weight")
            }
        )
        layers: list[torch.nn.Module] = []
        for layer_number, state_index in enumerate(linear_indices):
            weight = state_dict[f"actor.{state_index}.weight"]
            bias = state_dict[f"actor.{state_index}.bias"]
            linear = torch.nn.Linear(weight.shape[1], weight.shape[0])
            linear.weight.data.copy_(weight)
            linear.bias.data.copy_(bias)
            layers.append(linear)
            if layer_number != len(linear_indices) - 1:
                layers.append(torch.nn.ELU())
        self.actor = torch.nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(obs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    actor = Actor(state_dict).eval()
    input_dim = state_dict["actor.0.weight"].shape[1]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        actor,
        torch.zeros(1, input_dim),
        args.output,
        input_names=["obs"],
        output_names=["actions"],
        opset_version=11,
    )
    print(f"Exported {args.checkpoint} -> {args.output}")


if __name__ == "__main__":
    main()
