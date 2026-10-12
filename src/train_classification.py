from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from metrics import classification_scores
from models import build_resnet18_classifier, set_resnet_backbone_trainable
from PIL import Image
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
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
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "brisc2025")
    parser.add_argument("--split-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=PROJECT_ROOT / "checkpoints"
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--head-epochs", type=int, default=5)
    parser.add_argument("--fine-tune-epochs", type=int, default=15)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--fine-tune-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


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


def serializable_config(args: argparse.Namespace) -> dict[str, object]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def main() -> None:
    args = parse_args()

    seed_everything(args.seed)
    device = select_device()
    train_frame = pd.read_csv(args.split_root / "classification_train.csv")
    validation_frame = pd.read_csv(args.split_root / "classification_val.csv")
    required_columns = {"relative_path", "tumor_label"}
    if missing := required_columns - set(train_frame.columns):
        raise ValueError(f"Training CSV is missing columns: {sorted(missing)}")
    if missing := required_columns - set(validation_frame.columns):
        raise ValueError(f"Validation CSV is missing columns: {sorted(missing)}")

    train_transform, validation_transform = build_transforms(args.image_size)
    train_dataset = ClassificationDataset(train_frame, args.data_root, train_transform)
    validation_dataset = ClassificationDataset(
        validation_frame, args.data_root, validation_transform
    )
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    generator = torch.Generator().manual_seed(args.seed)
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

    model = build_resnet18_classifier(
        num_classes=len(CLASS_NAMES),
        pretrained=not args.no_pretrained,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    history: list[dict[str, object]] = []
    best_f1 = -1.0
    best_epoch = 0
    global_epoch = 0
    config = serializable_config(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.checkpoint_dir / "resnet18_best.ckpt"

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
                    checkpoint_path,
                )

    pd.DataFrame(history).to_csv(
        args.output_dir / "classification_history.csv",
        index=False,
    )
    summary = {
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_f1,
        "checkpoint": str(checkpoint_path),
        "config": config,
    }
    with (args.output_dir / "classification_summary.json").open("w") as file:
        json.dump(summary, file, indent=2)
    print(f"Best validation macro F1={best_f1:.4f} at epoch {best_epoch}")
    print(f"Saved best checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
