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
from segmentation_postprocessing import (
    SegmentationPostprocessConfig,
    postprocess_masks,
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


def soft_tversky_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    false_positive_weight: float = 0.7,
    smooth: float = 1e-6,
) -> torch.Tensor:
    """Asymmetric overlap loss with extra weight on false-positive pixels."""
    if not 0.0 <= false_positive_weight <= 1.0:
        raise ValueError("false_positive_weight must be in [0, 1]")
    probabilities = torch.sigmoid(logits)
    dimensions = tuple(range(1, targets.ndim))
    true_positive = (probabilities * targets).sum(dim=dimensions)
    false_positive = (probabilities * (1.0 - targets)).sum(dim=dimensions)
    false_negative = ((1.0 - probabilities) * targets).sum(dim=dimensions)
    false_negative_weight = 1.0 - false_positive_weight
    score = (true_positive + smooth) / (
        true_positive
        + false_positive_weight * false_positive
        + false_negative_weight * false_negative
        + smooth
    )
    return 1.0 - score.mean()


def segmentation_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    bce_weight: float = 0.5,
    false_positive_weight: float = 0.7,
    empty_mask_weight: float = 2.0,
) -> torch.Tensor:
    """Combine empty-mask-weighted BCE and false-positive-aware Tversky loss."""
    if not 0.0 <= bce_weight <= 1.0:
        raise ValueError("bce_weight must be in [0, 1]")
    if empty_mask_weight < 1.0:
        raise ValueError("empty_mask_weight must be at least 1")

    pixel_bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    sample_bce = pixel_bce.flatten(1).mean(dim=1)
    empty_masks = ~targets.flatten(1).bool().any(dim=1)
    sample_weights = torch.where(
        empty_masks,
        torch.full_like(sample_bce, empty_mask_weight),
        torch.ones_like(sample_bce),
    )
    weighted_bce = (sample_bce * sample_weights).mean()
    overlap = soft_tversky_loss(logits, targets, false_positive_weight)
    return bce_weight * weighted_bce + (1.0 - bce_weight) * overlap


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
    parser.add_argument(
        "--bce-weight",
        type=float,
        default=0.5,
        help="Contribution of BCE to the combined segmentation loss.",
    )
    parser.add_argument(
        "--false-positive-weight",
        type=float,
        default=0.7,
        help="Tversky weight for false positives; values above 0.5 penalize them more.",
    )
    parser.add_argument(
        "--empty-mask-weight",
        type=float,
        default=2.0,
        help="BCE multiplier for no-tumor training images.",
    )
    parser.add_argument(
        "--selection-metric",
        choices=("positive_dice", "balanced_dice"),
        default="balanced_dice",
        help="Validation metric used for checkpoint selection and early stopping.",
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
    if args.early_stopping is not None and args.early_stopping < 1:
        parser.error("--earlyStopping PATIENCE must be at least 1")
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be in [0, 1]")
    if not 0.0 <= args.bce_weight <= 1.0:
        parser.error("--bce-weight must be in [0, 1]")
    if not 0.0 <= args.false_positive_weight <= 1.0:
        parser.error("--false-positive-weight must be in [0, 1]")
    if args.empty_mask_weight < 1.0:
        parser.error("--empty-mask-weight must be at least 1")
    return args


def run_segmentation_epoch(
    model: UNet,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
    optimizer: AdamW | None = None,
    *,
    bce_weight: float = 0.5,
    false_positive_weight: float = 0.7,
    empty_mask_weight: float = 2.0,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {
        "loss": 0.0,
        "dice": 0.0,
        "iou": 0.0,
        "positive_dice": 0.0,
        "empty_specificity": 0.0,
        "empty_predicted_area": 0.0,
    }
    positive_count = 0
    empty_count = 0

    with torch.set_grad_enabled(training):
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = segmentation_loss(
                logits,
                masks,
                bce_weight=bce_weight,
                false_positive_weight=false_positive_weight,
                empty_mask_weight=empty_mask_weight,
            )
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
            empty = ~positive
            if empty.any():
                empty_predictions = (
                    torch.sigmoid(logits.detach()[empty]) >= threshold
                ).flatten(1)
                count = int(empty.sum())
                totals["empty_specificity"] += float(
                    (~empty_predictions.any(dim=1)).sum()
                )
                totals["empty_predicted_area"] += float(
                    empty_predictions.float().mean(dim=1).sum()
                )
                empty_count += count

    metrics = {
        name: total / len(loader.dataset)
        for name, total in totals.items()
        if name not in {"positive_dice", "empty_specificity", "empty_predicted_area"}
    }
    metrics["positive_dice"] = (
        totals["positive_dice"] / positive_count if positive_count else 0.0
    )
    metrics["empty_specificity"] = (
        totals["empty_specificity"] / empty_count if empty_count else 1.0
    )
    metrics["empty_predicted_area"] = (
        totals["empty_predicted_area"] / empty_count if empty_count else 0.0
    )
    metrics["balanced_dice"] = (
        metrics["positive_dice"] + metrics["empty_specificity"]
    ) / 2.0
    return metrics


def collect_segmentation_predictions(
    model: UNet,
    loader: DataLoader,
    frame: pd.DataFrame,
    device: torch.device,
    threshold: float,
    postprocess_config: SegmentationPostprocessConfig | None = None,
    no_tumor_probabilities: list[float] | None = None,
) -> list[dict[str, object]]:
    model.eval()
    predictions: list[dict[str, object]] = []
    offset = 0
    config = postprocess_config or SegmentationPostprocessConfig()

    with torch.no_grad():
        for images, masks in loader:
            probabilities = torch.sigmoid(model(images.to(device)))
            raw_masks = probabilities >= threshold
            batch_probabilities = (
                no_tumor_probabilities[offset : offset + len(images)]
                if no_tumor_probabilities is not None
                else None
            )
            predicted_masks, postprocess_metadata = postprocess_masks(
                raw_masks,
                config,
                batch_probabilities,
            )
            target_masks = masks.to(device) >= 0.5
            dimensions = (1, 2, 3)
            intersection = (predicted_masks & target_masks).sum(dim=dimensions).float()
            predicted_area = predicted_masks.sum(dim=dimensions).float()
            target_area = target_masks.sum(dim=dimensions).float()
            union = predicted_area + target_area - intersection
            dice = (2.0 * intersection + 1e-6) / (
                predicted_area + target_area + 1e-6
            )
            iou = (intersection + 1e-6) / (union + 1e-6)
            areas = predicted_masks.float().mean(dim=dimensions)
            target_areas = target_masks.float().mean(dim=dimensions)
            batch_metrics = zip(
                areas.cpu(),
                target_areas.cpu(),
                dice.cpu(),
                iou.cpu(),
            )
            for index, metrics in enumerate(batch_metrics):
                area, target_area_fraction, sample_dice, sample_iou = metrics
                row = frame.iloc[offset + index]
                prediction = {
                    "filename": row["filename"],
                    "relative_path": row["relative_path"],
                    "tumor_label": row["tumor_label"],
                    "predicted_mask_area_fraction": float(area),
                    "ground_truth_mask_area_fraction": float(target_area_fraction),
                    "mask_dice": float(sample_dice),
                    "mask_iou": float(sample_iou),
                    **postprocess_metadata[index],
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
    best_score = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
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
            bce_weight=args.bce_weight,
            false_positive_weight=args.false_positive_weight,
            empty_mask_weight=args.empty_mask_weight,
        )

        validation_metrics = run_segmentation_epoch(
            model,
            validation_loader,
            device,
            args.threshold,
            bce_weight=args.bce_weight,
            false_positive_weight=args.false_positive_weight,
            empty_mask_weight=args.empty_mask_weight,
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
            f"empty specificity={validation_metrics['empty_specificity']:.4f}, "
            f"balanced Dice={validation_metrics['balanced_dice']:.4f}"
        )

        selection_score = validation_metrics[args.selection_metric]
        if selection_score > best_score:
            best_score = selection_score
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_positive_dice": validation_metrics["positive_dice"],
                    "val_metrics": validation_metrics,
                    "selection_metric": args.selection_metric,
                    "selection_score": selection_score,
                    "config": config,
                },
                run.checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        if (
            args.early_stopping is not None
            and epochs_without_improvement >= args.early_stopping
        ):
            print(
                f"Early stopping after {args.early_stopping} epochs without "
                f"validation {args.selection_metric} improvement."
            )
            break

    checkpoint = torch.load(
        run.checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    best_metrics = checkpoint["val_metrics"]
    summary = {
        "run_id": run.run_id,
        "best_epoch": best_epoch,
        "selection_metric": args.selection_metric,
        "best_selection_score": best_score,
        "best_val_positive_dice": best_metrics["positive_dice"],
        "best_val_empty_specificity": best_metrics["empty_specificity"],
        "best_val_balanced_dice": best_metrics["balanced_dice"],
        "epochs_completed": len(history),
        "early_stopping_patience": args.early_stopping,
        "checkpoint": str(run.checkpoint_path),
        "config": config,
    }
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
    print(
        f"Best validation {args.selection_metric}={best_score:.4f} "
        f"at epoch {best_epoch}"
    )
    print(f"Saved best checkpoint to {run.checkpoint_path}")


if __name__ == "__main__":
    main()
