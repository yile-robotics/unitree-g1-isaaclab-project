#!/usr/bin/env python3
from __future__ import annotations

"""Convert the official full-module iPlanner checkpoint to a safe state dict.

The official RSS 2023 checkpoint stores ``(PlannerNet, validation_loss)``.
Uni-LaViRA's G1 server constructs PlannerNet itself and loads a state dict with
``strict=True``.  This one-time converter bridges those two formats without
changing either upstream source tree.
"""

import argparse
import hashlib
from pathlib import Path
import sys


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--module-dir",
        type=Path,
        required=True,
        help="Directory containing Uni-LaViRA planner_net.py/percept_net.py.",
    )
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    module_dir = args.module_dir.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Official checkpoint does not exist: {source}")
    if not (module_dir / "planner_net.py").is_file():
        raise FileNotFoundError(
            f"module-dir does not contain planner_net.py: {module_dir}"
        )
    if source == output:
        raise ValueError("Source and output checkpoint paths must differ.")

    sys.path.insert(0, str(module_dir))
    import torch

    # The official file contains a pickled PlannerNet object.  Only run this
    # conversion on a checkpoint obtained from the official iPlanner link.
    loaded = torch.load(source, map_location="cpu", weights_only=False)
    validation_loss = None
    candidate = loaded
    if isinstance(loaded, (tuple, list)):
        if not loaded:
            raise TypeError("Official checkpoint tuple/list is empty.")
        candidate = loaded[0]
        if len(loaded) > 1:
            try:
                validation_loss = float(loaded[1])
            except (TypeError, ValueError):
                validation_loss = None

    if isinstance(candidate, torch.nn.Module):
        state_dict = candidate.state_dict()
    elif isinstance(candidate, dict):
        state_dict = candidate
    else:
        raise TypeError(
            "Unsupported official checkpoint payload: "
            f"{type(candidate).__name__}"
        )

    if not state_dict or not all(isinstance(key, str) for key in state_dict):
        raise ValueError("Converted iPlanner state dict is empty or malformed.")
    if not all(torch.is_tensor(value) for value in state_dict.values()):
        raise ValueError("Converted iPlanner state dict contains non-tensor values.")

    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "source_name": source.name,
        "source_sha256": _sha256(source),
        "validation_loss": validation_loss,
        "format": "uni_lavira_iplanner_state_dict_v1",
    }
    torch.save((state_dict, metadata), output)
    print(f"SOURCE_SHA256={metadata['source_sha256']}")
    print(f"PARAMETER_TENSORS={len(state_dict)}")
    print(f"VALIDATION_LOSS={validation_loss}")
    print(f"OUTPUT={output}")
    print(f"OUTPUT_SHA256={_sha256(output)}")


if __name__ == "__main__":
    main()
