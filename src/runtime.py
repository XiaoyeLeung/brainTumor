from __future__ import annotations

import argparse
import json
import random
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

_RUN_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


@dataclass(frozen=True)
class RunPaths:
    run_id: str
    output_dir: Path
    checkpoint_dir: Path
    checkpoint_path: Path


def validate_run_name(value: str) -> str:
    if not _RUN_NAME_PATTERN.fullmatch(value):
        raise argparse.ArgumentTypeError(
            "run name must contain only letters, numbers, dots, underscores, "
            "and hyphens"
        )
    return value


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(name: str = "auto") -> torch.device:
    if name != "auto":
        if name == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable")
        if name == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS was requested but is unavailable")
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def prepare_run(
    args: argparse.Namespace, description: str
) -> tuple[RunPaths, dict[str, object]]:
    if not _RUN_NAME_PATTERN.fullmatch(description):
        raise ValueError(f"Invalid generated run description: {description!r}")

    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S-%f")
    run_id = args.run_name or f"{timestamp}_{description}"
    run_output_dir = args.output_dir / run_id
    run_checkpoint_dir = args.checkpoint_dir / run_id

    run_output_dir.mkdir(parents=True, exist_ok=True)
    if run_checkpoint_dir != run_output_dir:
        run_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    paths = RunPaths(
        run_id=run_id,
        output_dir=run_output_dir,
        checkpoint_dir=run_checkpoint_dir,
        checkpoint_path=run_checkpoint_dir / "best.pt",
    )
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config["run_id"] = run_id
    return paths, config


def save_run_outputs(
    history: Sequence[Mapping[str, object]],
    summary: Mapping[str, object],
    output_dir: Path,
    predictions: Sequence[Mapping[str, object]] | None = None,
) -> None:
    report = {
        "summary": dict(summary),
        "history": [dict(row) for row in history],
    }
    if predictions is not None:
        report["predictions"] = [dict(row) for row in predictions]
    with (output_dir / "run.json").open("w") as file:
        json.dump(report, file, indent=2)
