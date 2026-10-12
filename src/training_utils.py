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
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
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
            "run name must contain only letters, numbers, dots, underscores, and hyphens"
        )
    return value


def add_common_training_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_image_size: int,
    default_batch_size: int,
) -> None:
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "brisc2025")
    parser.add_argument("--split-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=PROJECT_ROOT / "checkpoints"
    )
    parser.add_argument(
        "--run-name",
        type=validate_run_name,
    )
    parser.add_argument("--image-size", type=int, default=default_image_size)
    parser.add_argument("--batch-size", type=int, default=default_batch_size)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_data_loaders(
    train_dataset: Dataset,
    validation_dataset: Dataset,
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    loader_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": num_workers > 0,
    }
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=generator,
        **loader_options,
    )
    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        **loader_options,
    )
    return train_loader, validation_loader


def validate_split_columns(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    required_columns: set[str],
) -> None:
    if missing := required_columns - set(train_frame.columns):
        raise ValueError(f"Training CSV is missing columns: {sorted(missing)}")
    if missing := required_columns - set(validation_frame.columns):
        raise ValueError(f"Validation CSV is missing columns: {sorted(missing)}")


def serializable_config(args: argparse.Namespace) -> dict[str, object]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def prepare_run(
    args: argparse.Namespace, description: str
) -> tuple[RunPaths, dict[str, object]]:
    if not _RUN_NAME_PATTERN.fullmatch(description):
        raise ValueError(f"Invalid generated run description: {description!r}")

    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S-%f")
    run_id = args.run_name or f"{timestamp}_{description}"
    run_output_dir = args.output_dir / run_id
    run_checkpoint_dir = args.checkpoint_dir / run_id

    run_output_dir.mkdir(parents=True)
    if run_checkpoint_dir != run_output_dir:
        run_checkpoint_dir.mkdir(parents=True)

    paths = RunPaths(
        run_id=run_id,
        output_dir=run_output_dir,
        checkpoint_dir=run_checkpoint_dir,
        checkpoint_path=run_checkpoint_dir / "best.pt",
    )
    config = serializable_config(args)
    config["run_id"] = run_id
    return paths, config


def save_run_outputs(
    history: Sequence[Mapping[str, object]],
    summary: Mapping[str, object],
    output_dir: Path,
) -> None:
    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
    with (output_dir / "summary.json").open("w") as file:
        json.dump(summary, file, indent=2)
