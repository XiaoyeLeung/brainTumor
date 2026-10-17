from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

CLASS_NAMES = ("glioma", "meningioma", "no_tumor", "pituitary")
EXPERIMENT_SPLITS = ("train", "validation", "test")


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


class SegmentationDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        data_root: Path,
        transform: Callable[..., dict[str, torch.Tensor]],
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.data_root = data_root
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        import cv2
        import numpy as np

        row = self.frame.iloc[index]
        image_path = self.data_root / Path(str(row["relative_path"]).replace("\\", "/"))
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        if pd.isna(row["mask_relative_path"]):
            if row["tumor_label"] != "no_tumor":
                raise ValueError(f"Missing mask for tumor-positive image: {image_path}")
            mask = np.zeros_like(image)
        else:
            mask_path = self.data_root / Path(
                str(row["mask_relative_path"]).replace("\\", "/")
            )
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(f"Could not read mask: {mask_path}")

        transformed = self.transform(image=image, mask=(mask > 0).astype(np.float32))
        return transformed["image"], transformed["mask"].float().unsqueeze(0)


class ReconstructionDataset(Dataset):
    """Unlabeled grayscale images used by the masked reconstruction task."""

    def __init__(
        self,
        frame: pd.DataFrame,
        data_root: Path,
        transform: transforms.Compose,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.data_root = data_root
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> torch.Tensor:
        row = self.frame.iloc[index]
        relative_path = Path(str(row["relative_path"]).replace("\\", "/"))
        with Image.open(self.data_root / relative_path) as image:
            return self.transform(image.convert("L"))


def build_classification_transforms(
    image_size: int,
) -> tuple[transforms.Compose, transforms.Compose]:
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


def build_reconstruction_transforms(
    image_size: int,
) -> tuple[transforms.Compose, transforms.Compose]:
    """Return mild train augmentation and deterministic reconstruction transforms."""
    normalize = transforms.Normalize(mean=(0.5,), std=(0.5,))
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(
                degrees=10,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05),
            ),
            transforms.ToTensor(),
            normalize,
        ]
    )
    validation_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_transform, validation_transform


def build_segmentation_transforms(image_size: int) -> tuple[object, object]:
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    import albumentations as A
    import cv2
    from albumentations.pytorch import ToTensorV2

    # train_transform = A.Compose(
    #     [
    #         A.Resize(image_size, image_size),
    #         A.HorizontalFlip(p=0.5),
    #         A.Rotate(limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.5),
    #         A.Normalize(mean=(0.5,), std=(0.5,)),
    #         ToTensorV2(),
    #     ]
    # )
    # validation_transform = A.Compose(
    #     [
    #         A.Resize(image_size, image_size),
    #         A.Normalize(mean=(0.5,), std=(0.5,)),
    #         ToTensorV2(),
    #     ]
    # )

    train_transform2 = A.Compose(
        [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.Affine(
                scale=(0.90, 1.10),
                translate_percent=(-0.05, 0.05),
                rotate=(-15, 15),
                shear=(-3, 3),
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                fill_mask=0,
                p=0.7,
            ),
            A.OneOf(
                [
                    A.RandomBrightnessContrast(
                        brightness_limit=0.15,
                        contrast_limit=0.15,
                        p=1.0,
                    ),
                    A.RandomGamma(gamma_limit=(85, 115), p=1.0),
                ],
                p=0.5,
            ),
            A.OneOf(
                [
                    A.GaussNoise(std_range=(0.01, 0.03), p=1.0),
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                ],
                p=0.15,
            ),
            A.Normalize(mean=(0.5,), std=(0.5,)),
            ToTensorV2(),
        ]
    )
    validation_transform2 = A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=(0.5,), std=(0.5,)),
            ToTensorV2(),
        ]
    )
    return train_transform2, validation_transform2


def load_classification_frame(split_root: Path, split: str) -> pd.DataFrame:
    """Load one classification split from an experiment manifest or legacy CSV."""
    if split not in EXPERIMENT_SPLITS:
        raise ValueError(f"split must be one of {EXPERIMENT_SPLITS}")

    manifest_path = split_root / "split_manifest.csv"
    if manifest_path.exists():
        frame = pd.read_csv(manifest_path)
        if "experiment_split" not in frame:
            raise ValueError(f"{manifest_path} is missing 'experiment_split'")
        result = frame.loc[frame["experiment_split"] == split].reset_index(drop=True)
    else:
        legacy_names = {
            "train": "classification_train.csv",
            "validation": "classification_val.csv",
        }
        if split not in legacy_names:
            raise ValueError("The test split requires a generated split_manifest.csv")
        result = pd.read_csv(split_root / legacy_names[split])
    _validate_frame_columns(
        result,
        {"relative_path", "filename", "tumor_label"},
        split,
    )
    return result


def load_classification_frames(
    split_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and validate classification train and validation rows."""
    return (
        load_classification_frame(split_root, "train"),
        load_classification_frame(split_root, "validation"),
    )


def load_segmentation_frame(split_root: Path, split: str) -> pd.DataFrame:
    """Load one aligned segmentation split."""
    classification = load_classification_frame(split_root, split)
    if "mask_relative_path" in classification.columns:
        _validate_frame_columns(classification, {"mask_relative_path"}, split)
        return classification

    legacy_names = {
        "train": "segmentation_train.csv",
        "validation": "segmentation_val.csv",
    }
    if split not in legacy_names:
        raise ValueError("The test split requires a generated split_manifest.csv")
    segmentation = pd.read_csv(split_root / legacy_names[split])
    _validate_frame_columns(
        segmentation,
        {"filename", "mask_relative_path"},
        split,
    )
    return build_aligned_segmentation_frame(classification, segmentation)


def load_segmentation_frames(
    split_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load aligned segmentation train and validation rows."""
    return (
        load_segmentation_frame(split_root, "train"),
        load_segmentation_frame(split_root, "validation"),
    )


def build_aligned_segmentation_frame(
    classification: pd.DataFrame,
    segmentation: pd.DataFrame,
) -> pd.DataFrame:
    mask_columns = ["filename", "mask_relative_path"]
    if "sha256" in segmentation.columns:
        mask_columns.append("sha256")
    masks = segmentation[mask_columns].rename(columns={"sha256": "mask_image_sha256"})
    if masks["filename"].duplicated().any():
        raise ValueError("Segmentation filenames must be unique")

    positive_filenames = set(
        classification.loc[classification["tumor_label"] != "no_tumor", "filename"]
    )
    mask_filenames = set(masks["filename"])
    if unexpected := mask_filenames - positive_filenames:
        raise ValueError(
            f"Segmentation rows do not belong to this split: {sorted(unexpected)[:5]}"
        )

    aligned = classification.merge(
        masks,
        on="filename",
        how="left",
        validate="one_to_one",
    )
    positive = aligned["tumor_label"] != "no_tumor"
    if aligned.loc[positive, "mask_relative_path"].isna().any():
        missing = aligned.loc[
            positive & aligned["mask_relative_path"].isna(), "filename"
        ].head(5)
        raise ValueError(f"Tumor-positive images are missing masks: {missing.tolist()}")
    if aligned.loc[~positive, "mask_relative_path"].notna().any():
        raise ValueError("No-tumor images must not have segmentation masks")
    if "mask_image_sha256" in aligned.columns and "sha256" in aligned.columns:
        mismatched = positive & (aligned["sha256"] != aligned["mask_image_sha256"])
        if mismatched.any():
            raise ValueError("Classification and segmentation image hashes differ")
        aligned = aligned.drop(columns="mask_image_sha256")
    return aligned


def _validate_split_columns(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    required_columns: set[str],
) -> None:
    if missing := required_columns - set(train_frame.columns):
        raise ValueError(f"Training CSV is missing columns: {sorted(missing)}")
    if missing := required_columns - set(validation_frame.columns):
        raise ValueError(f"Validation CSV is missing columns: {sorted(missing)}")


def _validate_frame_columns(
    frame: pd.DataFrame,
    required_columns: set[str],
    split: str,
) -> None:
    if missing := required_columns - set(frame.columns):
        raise ValueError(f"{split.title()} split is missing columns: {sorted(missing)}")


def build_train_validation_loaders(
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
