from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from metrics import dice_coefficient, iou_coefficient, segmentation_loss
from models import UNet
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from training_utils import (
    add_common_training_arguments,
    build_data_loaders,
    prepare_run,
    save_run_outputs,
    seed_everything,
    select_device,
    validate_split_columns,
)


class SegmentationDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        data_root: Path,
        transform: A.Compose,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.data_root = data_root
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.frame.iloc[index]
        image_path = self.data_root / Path(str(row["relative_path"]).replace("\\", "/"))
        mask_path = self.data_root / Path(
            str(row["mask_relative_path"]).replace("\\", "/")
        )
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        if mask is None:
            raise FileNotFoundError(f"Could not read mask: {mask_path}")

        transformed = self.transform(image=image, mask=(mask > 0).astype(np.float32))
        image_tensor = transformed["image"]
        mask_tensor = transformed["mask"].float().unsqueeze(0)
        return image_tensor, mask_tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_training_arguments(
        parser,
        default_image_size=256,
        default_batch_size=8,
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--bilinear",
        action="store_true",
        help="Use bilinear upsampling instead of transposed convolutions.",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def build_transforms(image_size: int) -> tuple[A.Compose, A.Compose]:
    train_transform = A.Compose(
        [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.5),
            A.Normalize(mean=(0.5,), std=(0.5,)),
            ToTensorV2(),
        ]
    )
    validation_transform = A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=(0.5,), std=(0.5,)),
            ToTensorV2(),
        ]
    )
    return train_transform, validation_transform


def run_epoch(
    model: UNet,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
    optimizer: AdamW | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "dice": 0.0, "iou": 0.0}

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

    return {name: total / len(loader.dataset) for name, total in totals.items()}


def main() -> None:
    args = parse_args()

    seed_everything(args.seed)
    device = select_device()
    train_frame = pd.read_csv(args.split_root / "segmentation_train.csv")
    validation_frame = pd.read_csv(args.split_root / "segmentation_val.csv")

    validate_split_columns(
        train_frame,
        validation_frame,
        {"relative_path", "mask_relative_path"},
    )

    train_transform, validation_transform = build_transforms(args.image_size)

    train_dataset = SegmentationDataset(train_frame, args.data_root, train_transform)

    validation_dataset = SegmentationDataset(
        validation_frame,
        args.data_root,
        validation_transform,
    )

    train_loader, validation_loader = build_data_loaders(
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
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            args.threshold,
            optimizer=optimizer,
        )

        validation_metrics = run_epoch(
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
            f"Dice={train_metrics['dice']:.4f}; "
            f"val loss={validation_metrics['loss']:.4f}, "
            f"Dice={validation_metrics['dice']:.4f}, IoU={validation_metrics['iou']:.4f}"
        )

        if validation_metrics["dice"] > best_dice:
            best_dice = validation_metrics["dice"]
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_dice": best_dice,
                    "config": config,
                },
                run.checkpoint_path,
            )

    summary = {
        "run_id": run.run_id,
        "best_epoch": best_epoch,
        "best_val_dice": best_dice,
        "checkpoint": str(run.checkpoint_path),
        "config": config,
    }
    save_run_outputs(history, summary, run.output_dir)
    print(f"Best validation Dice={best_dice:.4f} at epoch {best_epoch}")
    print(f"Saved best checkpoint to {run.checkpoint_path}")


if __name__ == "__main__":
    main()
