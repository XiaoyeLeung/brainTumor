from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from data import (
    EXPERIMENT_SPLITS,
    ClassificationDataset,
    SegmentationDataset,
    build_classification_transforms,
    build_segmentation_transforms,
    load_classification_frame,
    load_segmentation_frame,
)
from models import UNet, build_resnet18_classifier
from runtime import save_run_outputs, select_device
from train_classification import collect_classification_predictions
from train_segmentation import collect_segmentation_predictions

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a saved classification or segmentation checkpoint."
    )
    parser.add_argument(
        "--task",
        choices=("classification", "segmentation"),
        required=True,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=EXPERIMENT_SPLITS[1:], required=True)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "brisc2025")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    config = checkpoint.get("config", {})

    if args.task == "classification":
        class_names = tuple(
            checkpoint.get(
                "class_names",
                ("glioma", "meningioma", "no_tumor", "pituitary"),
            )
        )
        image_size = int(config.get("image_size", 224))
        batch_size = args.batch_size or int(config.get("batch_size", 32))
        frame = load_classification_frame(args.split_root, args.split)
        _, transform = build_classification_transforms(image_size)
        dataset = ClassificationDataset(frame, args.data_root, transform)
        model = build_resnet18_classifier(len(class_names), pretrained=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        loader = _build_loader(dataset, batch_size, args.num_workers, device)
        predictions = collect_classification_predictions(
            model, loader, dataset.frame, class_names, device
        )
    else:
        image_size = int(config.get("image_size", 256))
        batch_size = args.batch_size or int(config.get("batch_size", 8))
        threshold = float(config.get("threshold", 0.5))
        frame = load_segmentation_frame(args.split_root, args.split)
        _, transform = build_segmentation_transforms(image_size)
        dataset = SegmentationDataset(frame, args.data_root, transform)
        model = UNet(
            n_channels=1,
            n_classes=1,
            bilinear=bool(config.get("bilinear", False)),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        loader = _build_loader(dataset, batch_size, args.num_workers, device)
        predictions = collect_segmentation_predictions(
            model, loader, dataset.frame, device, threshold
        )

    summary = {
        "task": args.task,
        "prediction_split": args.split,
        "prediction_count": len(predictions),
        "checkpoint": str(args.checkpoint),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_run_outputs([], summary, args.output_dir, predictions)
    print(
        f"Saved {len(predictions)} {args.split} predictions to "
        f"{args.output_dir / 'run.json'}"
    )


def _build_loader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )


if __name__ == "__main__":
    main()
