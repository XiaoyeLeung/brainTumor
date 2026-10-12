from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from metrics import classification_scores
from models import build_resnet18_classifier, set_resnet_backbone_trainable
from PIL import Image
from torch import nn
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
from torchvision import transforms

CLASS_NAMES = ("glioma", "meningioma", "no_tumor", "pituitary")


class ClassificationDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        data_root: Path,
        transform: transforms.Compose,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.data_root = data_root
        self.transform = transform
        self.label_to_index = {name: index for index, name in enumerate(CLASS_NAMES)}

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.frame.iloc[index]
        relative_path = Path(str(row["relative_path"]).replace("\\", "/"))
        with Image.open(self.data_root / relative_path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, self.label_to_index[row["tumor_label"]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_training_arguments(
        parser,
        default_image_size=224,
        default_batch_size=32,
    )
    parser.add_argument("--head-epochs", type=int, default=5)
    parser.add_argument("--fine-tune-epochs", type=int, default=15)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--fine-tune-lr", type=float, default=1e-4)
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def build_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            normalize,
        ]
    )
    validation_transform = transforms.Compose(
        [transforms.Resize((image_size, image_size)), transforms.ToTensor(), normalize]
    )
    return train_transform, validation_transform


def run_epoch(
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


def sample_frame(frame: pd.DataFrame, size: int, seed: int) -> pd.DataFrame:
    return frame.sample(n=min(size, len(frame)), random_state=seed).reset_index(
        drop=True
    )


def main() -> None:
    args = parse_args()

    seed_everything(args.seed)
    device = select_device()
    train_frame = pd.read_csv(args.split_root / "classification_train.csv")
    validation_frame = pd.read_csv(args.split_root / "classification_val.csv")
    validate_split_columns(
        train_frame,
        validation_frame,
        {"relative_path", "tumor_label"},
    )

    train_transform, validation_transform = build_transforms(args.image_size)
    train_dataset = ClassificationDataset(train_frame, args.data_root, train_transform)
    validation_dataset = ClassificationDataset(
        validation_frame, args.data_root, validation_transform
    )
    train_loader, validation_loader = build_data_loaders(
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
            train_metrics = run_epoch(
                model,
                train_loader,
                criterion,
                device,
                optimizer=optimizer,
                head_only=head_only,
            )
            validation_metrics = run_epoch(
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
    save_run_outputs(history, summary, run.output_dir)
    print(f"Best validation macro F1={best_f1:.4f} at epoch {best_epoch}")
    print(f"Saved best checkpoint to {run.checkpoint_path}")


if __name__ == "__main__":
    main()
