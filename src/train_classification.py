from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from data import (
    CLASS_NAMES,
    ClassificationDataset,
    build_train_validation_loaders,
    build_classification_transforms,
    load_classification_frames,
)
from models import (
    build_resnet18_classifier,
    set_resnet_backbone_trainable,
)
from runtime import (
    prepare_run,
    save_run_outputs,
    seed_everything,
    select_device,
    validate_run_name,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def classification_scores(
    targets: Sequence[int],
    predictions: Sequence[int],
) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(
            f1_score(targets, predictions, average="macro", zero_division=0)
        ),
        "macro_precision": float(
            precision_score(targets, predictions, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(targets, predictions, average="macro", zero_division=0)
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "brisc2025")
    parser.add_argument("--split-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=PROJECT_ROOT / "checkpoints"
    )
    parser.add_argument("--run-name", type=validate_run_name)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--head-epochs", type=int, default=5)
    parser.add_argument("--fine-tune-epochs", type=int, default=15)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--fine-tune-lr", type=float, default=1e-4)
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def run_classification_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: AdamW | None = None,
    head_only: bool = False,
) -> dict[str, float]:
    training = optimizer is not None
    if training and head_only:
        model.eval()
        model.fc.train()
    else:
        model.train(training)

    total_loss = 0.0
    targets: list[int] = []
    predictions: list[int] = []

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            targets.extend(labels.cpu().tolist())
            predictions.extend(logits.argmax(dim=1).cpu().tolist())

    scores = classification_scores(targets, predictions)
    scores["loss"] = total_loss / len(loader.dataset)
    return scores


def collect_classification_predictions(
    model: nn.Module,
    loader: DataLoader,
    frame: pd.DataFrame,
    class_names: tuple[str, ...],
    device: torch.device,
) -> list[dict[str, object]]:
    model.eval()
    predictions: list[dict[str, object]] = []
    offset = 0
    no_tumor_index = class_names.index("no_tumor")

    with torch.no_grad():
        for images, _ in loader:
            probabilities = torch.softmax(model(images.to(device)), dim=1).cpu()
            for index, values in enumerate(probabilities):
                row = frame.iloc[offset + index]
                prediction = {
                    "filename": row["filename"],
                    "relative_path": row["relative_path"],
                    "tumor_label": row["tumor_label"],
                    "predicted_class": class_names[int(values.argmax())],
                    "no_tumor_probability": float(values[no_tumor_index]),
                    **{
                        f"probability_{name}": float(values[class_index])
                        for class_index, name in enumerate(class_names)
                    },
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
    train_frame, validation_frame = load_classification_frames(args.split_root)

    train_transform, validation_transform = build_classification_transforms(
        args.image_size
    )
    train_dataset = ClassificationDataset(train_frame, args.data_root, train_transform)
    validation_dataset = ClassificationDataset(
        validation_frame, args.data_root, validation_transform
    )
    train_loader, validation_loader = build_train_validation_loaders(
        train_dataset,
        validation_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        seed=args.seed,
    )

    model = build_resnet18_classifier(
        num_classes=len(CLASS_NAMES),
        pretrained=not args.no_pretrained,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    history: list[dict[str, object]] = []
    best_f1 = -1.0
    best_epoch = 0
    global_epoch = 0
    initialization = "scratch" if args.no_pretrained else "pretrained"
    description = (
        f"resnet18-{initialization}_img{args.image_size}_bs{args.batch_size}_"
        f"headlr{args.head_lr:g}_ftlr{args.fine_tune_lr:g}_seed{args.seed}"
    )
    run, config = prepare_run(args, description)

    print(f"Run: {run.run_id}")
    print(
        f"Device: {device}; train={len(train_dataset)}, validation={len(validation_dataset)}"
    )
    stages = (
        ("head", args.head_epochs, args.head_lr, True),
        ("fine_tune", args.fine_tune_epochs, args.fine_tune_lr, False),
    )
    for stage, epochs, learning_rate, head_only in stages:
        if epochs == 0:
            continue
        set_resnet_backbone_trainable(model, trainable=not head_only)
        trainable_parameters = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        optimizer = AdamW(
            trainable_parameters,
            lr=learning_rate,
            weight_decay=args.weight_decay,
        )

        for stage_epoch in range(1, epochs + 1):
            global_epoch += 1
            train_metrics = run_classification_epoch(
                model,
                train_loader,
                criterion,
                device,
                optimizer=optimizer,
                head_only=head_only,
            )
            validation_metrics = run_classification_epoch(
                model,
                validation_loader,
                criterion,
                device,
            )
            row: dict[str, object] = {
                "epoch": global_epoch,
                "stage": stage,
                "stage_epoch": stage_epoch,
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"val_{key}": value for key, value in validation_metrics.items()},
            }
            history.append(row)
            print(
                f"Epoch {global_epoch:02d} [{stage}]: "
                f"train loss={train_metrics['loss']:.4f}, F1={train_metrics['macro_f1']:.4f}; "
                f"val loss={validation_metrics['loss']:.4f}, "
                f"F1={validation_metrics['macro_f1']:.4f}, "
                f"accuracy={validation_metrics['accuracy']:.4f}"
            )

            if validation_metrics["macro_f1"] > best_f1:
                best_f1 = validation_metrics["macro_f1"]
                best_epoch = global_epoch
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "class_names": CLASS_NAMES,
                        "epoch": global_epoch,
                        "val_macro_f1": best_f1,
                        "config": config,
                    },
                    run.checkpoint_path,
                )

    summary = {
        "run_id": run.run_id,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_f1,
        "checkpoint": str(run.checkpoint_path),
        "config": config,
    }
    checkpoint = torch.load(
        run.checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    predictions = collect_classification_predictions(
        model,
        validation_loader,
        validation_dataset.frame,
        CLASS_NAMES,
        device,
    )
    summary["prediction_split"] = "validation"
    summary["prediction_count"] = len(predictions)
    save_run_outputs(history, summary, run.output_dir, predictions)
    print(f"Best validation macro F1={best_f1:.4f} at epoch {best_epoch}")
    print(f"Saved best checkpoint to {run.checkpoint_path}")


if __name__ == "__main__":
    main()
