from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from data import (
    SegmentationDataset,
    build_segmentation_transforms,
    build_train_validation_loaders,
    load_segmentation_frames,
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


def soft_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    smooth: float = 1e-6,
) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    dimensions = tuple(range(1, targets.ndim))
    intersection = (probabilities * targets).sum(dim=dimensions)
    total = probabilities.sum(dim=dimensions) + targets.sum(dim=dimensions)
    dice = (2.0 * intersection + smooth) / (total + smooth)
    return 1.0 - dice.mean()


def segmentation_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    bce_weight: float = 0.5,
) -> torch.Tensor:
    """Combine pixel-wise BCE and soft Dice losses."""
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    return bce_weight * bce + (1.0 - bce_weight) * soft_dice_loss(logits, targets)


def _binary_overlap(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float,
    metric: str,
    smooth: float = 1e-6,
) -> torch.Tensor:
    predictions = (torch.sigmoid(logits.detach()) >= threshold).float()
    targets = (targets.detach() >= 0.5).float()
    dimensions = tuple(range(1, targets.ndim))
    intersection = (predictions * targets).sum(dim=dimensions)

    if metric == "dice":
        denominator = predictions.sum(dim=dimensions) + targets.sum(dim=dimensions)
        scores = (2.0 * intersection + smooth) / (denominator + smooth)
    else:
        union = (
            predictions.sum(dim=dimensions) + targets.sum(dim=dimensions) - intersection
        )
        scores = (intersection + smooth) / (union + smooth)
    return scores.mean()


def dice_coefficient(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Mean hard Dice score for a batch of binary masks."""
    return _binary_overlap(logits, targets, threshold, metric="dice")


def iou_coefficient(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Mean intersection-over-union score for binary masks."""
    return _binary_overlap(logits, targets, threshold, metric="iou")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "brisc2025")
    parser.add_argument("--split-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=PROJECT_ROOT / "checkpoints"
    )
    parser.add_argument("--run-name", type=validate_run_name)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--bilinear",
        action="store_true",
        help="Use bilinear upsampling instead of transposed convolutions.",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def run_segmentation_epoch(
    model: UNet,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
    optimizer: AdamW | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "dice": 0.0, "iou": 0.0, "positive_dice": 0.0}
    positive_count = 0

    with torch.set_grad_enabled(training):
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = segmentation_loss(logits, masks)
            if training:
                loss.backward()
                optimizer.step()

            batch_size = images.size(0)
            totals["loss"] += loss.item() * batch_size
            totals["dice"] += (
                dice_coefficient(logits, masks, threshold).item() * batch_size
            )
            totals["iou"] += (
                iou_coefficient(logits, masks, threshold).item() * batch_size
            )
            positive = masks.flatten(1).any(dim=1)
            if positive.any():
                count = int(positive.sum())
                totals["positive_dice"] += (
                    dice_coefficient(logits[positive], masks[positive], threshold).item()
                    * count
                )
                positive_count += count

    metrics = {
        name: total / len(loader.dataset)
        for name, total in totals.items()
        if name != "positive_dice"
    }
    metrics["positive_dice"] = (
        totals["positive_dice"] / positive_count if positive_count else 0.0
    )
    return metrics


def collect_segmentation_predictions(
    model: UNet,
    loader: DataLoader,
    frame: pd.DataFrame,
    device: torch.device,
    threshold: float,
) -> list[dict[str, object]]:
    model.eval()
    predictions: list[dict[str, object]] = []
    offset = 0

    with torch.no_grad():
        for images, _ in loader:
            probabilities = torch.sigmoid(model(images.to(device)))
            areas = (probabilities >= threshold).float().mean(dim=(1, 2, 3)).cpu()
            for index, area in enumerate(areas):
                row = frame.iloc[offset + index]
                prediction = {
                    "filename": row["filename"],
                    "relative_path": row["relative_path"],
                    "tumor_label": row["tumor_label"],
                    "predicted_mask_area_fraction": float(area),
                }
                if "plane_label" in frame:
                    prediction["plane_label"] = row["plane_label"]
                predictions.append(prediction)
            offset += len(images)
    return predictions


def main() -> None:
    args = parse_args()

    seed_everything(args.seed)
    device = select_device(args.device)
    train_frame, validation_frame = load_segmentation_frames(args.split_root)

    train_transform, validation_transform = build_segmentation_transforms(
        args.image_size
    )

    train_dataset = SegmentationDataset(train_frame, args.data_root, train_transform)

    validation_dataset = SegmentationDataset(
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
    history: list[dict[str, float | int]] = []
    best_dice = -1.0
    best_epoch = 0
    upsampling = "bilinear" if args.bilinear else "transposed"
    description = (
        f"unet-{upsampling}_img{args.image_size}_"
        f"bs{args.batch_size}_lr{args.lr:g}_seed{args.seed}"
    )
    run, config = prepare_run(args, description)

    print(f"Run: {run.run_id}")
    print(
        f"Device: {device}; train={len(train_dataset)}, validation={len(validation_dataset)}"
    )
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_segmentation_epoch(
            model,
            train_loader,
            device,
            args.threshold,
            optimizer=optimizer,
        )

        validation_metrics = run_segmentation_epoch(
            model,
            validation_loader,
            device,
            args.threshold,
        )

        history.append(
            {
                "epoch": epoch,
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"val_{key}": value for key, value in validation_metrics.items()},
            }
        )
        print(
            f"Epoch {epoch:02d}: train loss={train_metrics['loss']:.4f}, "
            f"positive Dice={train_metrics['positive_dice']:.4f}; "
            f"val loss={validation_metrics['loss']:.4f}, "
            f"positive Dice={validation_metrics['positive_dice']:.4f}, "
            f"all-case Dice={validation_metrics['dice']:.4f}"
        )

        if validation_metrics["positive_dice"] > best_dice:
            best_dice = validation_metrics["positive_dice"]
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_positive_dice": best_dice,
                    "config": config,
                },
                run.checkpoint_path,
            )

    summary = {
        "run_id": run.run_id,
        "best_epoch": best_epoch,
        "best_val_positive_dice": best_dice,
        "checkpoint": str(run.checkpoint_path),
        "config": config,
    }
    checkpoint = torch.load(
        run.checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    predictions = collect_segmentation_predictions(
        model,
        validation_loader,
        validation_dataset.frame,
        device,
        args.threshold,
    )
    summary["prediction_split"] = "validation"
    summary["prediction_count"] = len(predictions)
    save_run_outputs(history, summary, run.output_dir, predictions)
    print(f"Best validation positive Dice={best_dice:.4f} at epoch {best_epoch}")
    print(f"Saved best checkpoint to {run.checkpoint_path}")


if __name__ == "__main__":
    main()
