from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from data import (
    ReconstructionDataset,
    build_reconstruction_transforms,
    build_train_validation_loaders,
    load_classification_frame,
)
from models import UNet
from runtime import (
    prepare_run,
    save_run_outputs,
    seed_everything,
    select_device,
    validate_run_name,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def split_pretraining_frame(
    frame: pd.DataFrame,
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split downstream-training rows for reconstruction without evaluation leakage."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    if len(frame) < 2:
        raise ValueError("Masked pretraining requires at least two training images")
    if frame["filename"].duplicated().any():
        raise ValueError("Pretraining filenames must be unique")
    if "experiment_split" in frame and set(frame["experiment_split"]) != {"train"}:
        raise ValueError(
            "Masked pretraining accepts experiment_split='train' rows only"
        )

    indices = np.arange(len(frame))
    np.random.default_rng(seed).shuffle(indices)
    validation_count = min(
        len(frame) - 1,
        max(1, round(len(frame) * validation_fraction)),
    )
    validation_indices = indices[:validation_count]
    train_indices = indices[validation_count:]
    return (
        frame.iloc[train_indices].reset_index(drop=True),
        frame.iloc[validation_indices].reset_index(drop=True),
    )


def build_patch_mask(
    batch_size: int,
    height: int,
    width: int,
    *,
    patch_size: int,
    mask_ratio: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Create an exact-ratio random block mask with shape (B, 1, H, W)."""
    if patch_size < 1:
        raise ValueError("patch_size must be positive")
    if height % patch_size or width % patch_size:
        raise ValueError("Image dimensions must be divisible by patch_size")
    if not 0.0 < mask_ratio < 1.0:
        raise ValueError("mask_ratio must be between 0 and 1")

    patch_rows = height // patch_size
    patch_columns = width // patch_size
    patch_count = patch_rows * patch_columns
    if patch_count < 2:
        raise ValueError("At least two patches are required")
    masked_patch_count = max(1, min(patch_count - 1, round(patch_count * mask_ratio)))
    selected = torch.rand(
        batch_size,
        patch_count,
        generator=generator,
    ).topk(masked_patch_count, dim=1).indices
    patch_mask = torch.zeros(batch_size, patch_count, dtype=torch.bool)
    patch_mask.scatter_(1, selected, True)
    return (
        patch_mask.reshape(batch_size, 1, patch_rows, patch_columns)
        .repeat_interleave(patch_size, dim=2)
        .repeat_interleave(patch_size, dim=3)
    )


def corrupt_images(
    images: torch.Tensor,
    *,
    patch_size: int,
    mask_ratio: float,
    noise_std: float,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Add light Gaussian noise and replace selected patches with normalized zero."""
    mask = build_patch_mask(
        len(images),
        images.shape[-2],
        images.shape[-1],
        patch_size=patch_size,
        mask_ratio=mask_ratio,
        generator=generator,
    ).to(images.device)
    if noise_std:
        noise = torch.randn(
            images.shape,
            generator=generator,
            dtype=images.dtype,
        ).to(images.device)
        corrupted = images + noise_std * noise
    else:
        corrupted = images.clone()
    return corrupted.masked_fill(mask, 0.0), mask


def masked_reconstruction_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Mean absolute reconstruction error over masked pixels only."""
    if reconstruction.shape != target.shape or mask.shape != target.shape:
        raise ValueError("reconstruction, target, and mask shapes must match")
    masked_pixels = mask.sum()
    if masked_pixels.item() == 0:
        raise ValueError("Reconstruction mask must contain masked pixels")
    return ((reconstruction - target).abs() * mask).sum() / masked_pixels


def run_reconstruction_epoch(
    model: UNet,
    loader: DataLoader,
    device: torch.device,
    *,
    patch_size: int,
    mask_ratio: float,
    noise_std: float,
    optimizer: AdamW | None = None,
    validation_mask_seed: int | None = None,
) -> float:
    training = optimizer is not None
    model.train(training)
    generator = (
        torch.Generator().manual_seed(validation_mask_seed)
        if validation_mask_seed is not None
        else None
    )
    total_loss = 0.0

    with torch.set_grad_enabled(training):
        for images in loader:
            images = images.to(device, non_blocking=True)
            corrupted, mask = corrupt_images(
                images,
                patch_size=patch_size,
                mask_ratio=mask_ratio,
                noise_std=noise_std,
                generator=generator,
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
            reconstruction = model(corrupted)
            loss = masked_reconstruction_loss(reconstruction, images, mask)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(images)

    return total_loss / len(loader.dataset)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pretrain U-Net using train-only masked image reconstruction."
    )
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "brisc2025")
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--run-name", type=validate_run_name, default="pretraining")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--mask-ratio", type=float, default=0.35)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument(
        "--bilinear",
        action="store_true",
        help="Use bilinear upsampling instead of transposed convolutions.",
    )
    parser.add_argument(
        "--earlyStopping",
        dest="early_stopping",
        nargs="?",
        const=5,
        type=int,
        metavar="PATIENCE",
        help="Stop after PATIENCE non-improving validation epochs (default: 5).",
    )
    args = parser.parse_args()
    if args.patch_size < 1:
        parser.error("--patch-size must be positive")
    if args.image_size <= args.patch_size or args.image_size % args.patch_size:
        parser.error(
            "--image-size must be divisible by and larger than --patch-size"
        )
    if not 0.0 < args.mask_ratio < 1.0:
        parser.error("--mask-ratio must be between 0 and 1")
    if args.noise_std < 0.0:
        parser.error("--noise-std must be non-negative")
    if not 0.0 < args.validation_fraction < 1.0:
        parser.error("--validation-fraction must be between 0 and 1")
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.early_stopping is not None and args.early_stopping < 1:
        parser.error("--earlyStopping PATIENCE must be at least 1")
    return args


def _value_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in frame:
        return {}
    return {str(key): int(value) for key, value in frame[column].value_counts().items()}


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = select_device(args.device)

    source_frame = load_classification_frame(args.split_root, "train")
    train_frame, validation_frame = split_pretraining_frame(
        source_frame,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    train_transform, validation_transform = build_reconstruction_transforms(
        args.image_size
    )
    train_dataset = ReconstructionDataset(train_frame, args.data_root, train_transform)
    validation_dataset = ReconstructionDataset(
        validation_frame,
        args.data_root,
        validation_transform,
    )
    train_loader, validation_loader = build_train_validation_loaders(
        train_dataset,
        validation_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        seed=args.seed,
    )

    model = UNet(n_channels=1, n_classes=1, bilinear=args.bilinear).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    description = (
        f"masked-unet_img{args.image_size}_patch{args.patch_size}_"
        f"mask{args.mask_ratio:g}_seed{args.seed}"
    )
    run, config = prepare_run(args, description)

    manifest_columns = [
        column
        for column in (
            "filename",
            "relative_path",
            "tumor_label",
            "plane_label",
            "experiment_split",
        )
        if column in source_frame
    ]
    manifest = pd.concat(
        [
            train_frame[manifest_columns].assign(pretraining_split="train"),
            validation_frame[manifest_columns].assign(
                pretraining_split="validation"
            ),
        ],
        ignore_index=True,
    )
    manifest.to_csv(run.output_dir / "pretraining_manifest.csv", index=False)

    print(f"Run: {run.run_id}")
    print(
        f"Device: {device}; train-only source={len(source_frame)}, "
        f"pretrain={len(train_dataset)}, validation={len(validation_dataset)}"
    )
    history: list[dict[str, float | int]] = []
    best_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    for epoch in range(1, args.epochs + 1):
        train_loss = run_reconstruction_epoch(
            model,
            train_loader,
            device,
            patch_size=args.patch_size,
            mask_ratio=args.mask_ratio,
            noise_std=args.noise_std,
            optimizer=optimizer,
        )
        validation_loss = run_reconstruction_epoch(
            model,
            validation_loader,
            device,
            patch_size=args.patch_size,
            mask_ratio=args.mask_ratio,
            noise_std=args.noise_std,
            validation_mask_seed=args.seed,
        )
        history.append(
            {
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train_masked_l1": train_loss,
                "val_masked_l1": validation_loss,
            }
        )
        print(
            f"Epoch {epoch:02d}: train masked L1={train_loss:.4f}; "
            f"val masked L1={validation_loss:.4f}"
        )

        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_masked_l1": validation_loss,
                    "config": config,
                    "pretraining_source_split": "train",
                },
                run.checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
        scheduler.step()

        if (
            args.early_stopping is not None
            and epochs_without_improvement >= args.early_stopping
        ):
            print(
                f"Early stopping after {args.early_stopping} epochs without "
                "validation masked-L1 improvement."
            )
            break

    summary = {
        "run_id": run.run_id,
        "best_epoch": best_epoch,
        "best_val_masked_l1": best_loss,
        "epochs_completed": len(history),
        "checkpoint": str(run.checkpoint_path),
        "config": config,
        "data_provenance": {
            "source_experiment_split": "train",
            "excluded_experiment_splits": ["validation", "test"],
            "source_count": len(source_frame),
            "pretraining_train_count": len(train_frame),
            "pretraining_validation_count": len(validation_frame),
            "source_planes": _value_counts(source_frame, "plane_label"),
            "source_classes": _value_counts(source_frame, "tumor_label"),
            "manifest": str(run.output_dir / "pretraining_manifest.csv"),
        },
    }
    save_run_outputs(history, summary, run.output_dir)
    print(f"Best validation masked L1={best_loss:.4f} at epoch {best_epoch}")
    print(f"Saved pretraining checkpoint to {run.checkpoint_path}")


if __name__ == "__main__":
    main()
