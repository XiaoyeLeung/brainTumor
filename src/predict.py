from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from data import (
    CLASS_NAMES,
    EXPERIMENT_SPLITS,
    ClassificationDataset,
    SegmentationDataset,
    build_classification_transforms,
    build_segmentation_transforms,
    load_classification_frame,
    load_segmentation_frame,
)
from decision_fusion import load_predictions
from models import UNet, build_resnet18_classifier
from runtime import save_run_outputs, select_device
from segmentation_postprocessing import (
    SegmentationPostprocessConfig,
    align_no_tumor_probabilities,
    load_postprocess_config,
)
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
        "--min-component-pixels",
        type=int,
        help="Remove predicted connected components smaller than this many pixels.",
    )
    parser.add_argument(
        "--classification-run",
        type=Path,
        help="Classification run.json used for no-tumor mask gating.",
    )
    parser.add_argument(
        "--no-tumor-gate-threshold",
        type=float,
        help="Clear a mask when classifier no-tumor probability reaches this value.",
    )
    parser.add_argument(
        "--postprocessing-config",
        type=Path,
        help="JSON file containing a segmentation_postprocessing configuration.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    args = parser.parse_args()
    if args.min_component_pixels is not None and args.min_component_pixels < 0:
        parser.error("--min-component-pixels must be non-negative")
    if args.no_tumor_gate_threshold is not None and not (
        0.0 <= args.no_tumor_gate_threshold <= 1.0
    ):
        parser.error("--no-tumor-gate-threshold must be in [0, 1]")
    return args


def resolve_postprocess_config(args: argparse.Namespace) -> SegmentationPostprocessConfig:
    stored = (
        load_postprocess_config(args.postprocessing_config)
        if args.postprocessing_config is not None
        else SegmentationPostprocessConfig()
    )
    return SegmentationPostprocessConfig(
        min_component_pixels=(
            args.min_component_pixels
            if args.min_component_pixels is not None
            else stored.min_component_pixels
        ),
        no_tumor_gate_threshold=(
            args.no_tumor_gate_threshold
            if args.no_tumor_gate_threshold is not None
            else stored.no_tumor_gate_threshold
        ),
    )


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    config = checkpoint.get("config", {})

    postprocess_config: SegmentationPostprocessConfig | None = None
    if args.task == "classification":
        if any(
            value is not None
            for value in (
                args.min_component_pixels,
                args.classification_run,
                args.no_tumor_gate_threshold,
                args.postprocessing_config,
            )
        ):
            raise ValueError(
                "Segmentation post-processing options cannot be used for classification"
            )
        class_names = tuple(
            checkpoint.get(
                "class_names",
                CLASS_NAMES,
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
        postprocess_config = resolve_postprocess_config(args)
        if (
            postprocess_config.no_tumor_gate_threshold is not None
            and args.classification_run is None
        ):
            raise ValueError(
                "--classification-run is required when classifier gating is enabled"
            )
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
        no_tumor_probabilities = None
        if postprocess_config.no_tumor_gate_threshold is not None:
            classification_predictions = load_predictions(
                args.classification_run,
                required_split=args.split,
            )
            no_tumor_probabilities = align_no_tumor_probabilities(
                dataset.frame,
                classification_predictions,
            )
        predictions = collect_segmentation_predictions(
            model,
            loader,
            dataset.frame,
            device,
            threshold,
            postprocess_config,
            no_tumor_probabilities,
        )

    summary = {
        "task": args.task,
        "prediction_split": args.split,
        "prediction_count": len(predictions),
        "checkpoint": str(args.checkpoint),
    }
    if postprocess_config is not None:
        summary["postprocessing"] = postprocess_config.to_dict()
        summary["classification_run"] = (
            str(args.classification_run) if args.classification_run else None
        )
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
