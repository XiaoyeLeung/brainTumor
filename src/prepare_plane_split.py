from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from data import build_aligned_segmentation_frame

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLANES = ("none", "axial", "coronal", "sagittal")
TARGET_PLANES = PLANES[1:]


def build_aligned_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    classification = manifest.loc[
        (manifest["task"] == "classification") & ~manifest["is_mask"]
    ].copy()
    segmentation_images = manifest.loc[
        (manifest["task"] == "segmentation") & ~manifest["is_mask"],
        ["filename", "sha256"],
    ]
    masks = manifest.loc[
        (manifest["task"] == "segmentation") & manifest["is_mask"],
        ["relative_path", "linked_image"],
    ].copy()
    masks["filename"] = masks["linked_image"].map(
        lambda value: Path(str(value).replace("\\", "/")).name
    )
    segmentation = segmentation_images.merge(
        masks[["filename", "relative_path"]].rename(
            columns={"relative_path": "mask_relative_path"}
        ),
        on="filename",
        validate="one_to_one",
    )
    return build_aligned_segmentation_frame(classification, segmentation).rename(
        columns={"split": "official_split"}
    )


def build_experiment_split(
    frame: pd.DataFrame,
    *,
    held_out_plane: str,
    seed: int,
    validation_fraction: float = 0.2,
) -> pd.DataFrame:
    if held_out_plane not in PLANES:
        raise ValueError(f"held_out_plane must be one of {PLANES}")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")

    official_train = frame.loc[frame["official_split"] == "train"]
    development = official_train
    test = frame.loc[frame["official_split"] == "test"]
    if held_out_plane != "none":
        development = official_train.loc[
            official_train["plane_label"] != held_out_plane
        ]
        test = official_train.loc[official_train["plane_label"] == held_out_plane]

    strata = development["tumor_label"] + "|" + development["plane_label"]
    train, validation = train_test_split(
        development,
        test_size=validation_fraction,
        random_state=seed,
        stratify=strata,
    )
    parts = []
    for name, part in (("train", train), ("validation", validation), ("test", test)):
        part = part.copy()
        part["experiment_split"] = name
        parts.append(part)
    result = pd.concat(parts, ignore_index=True)
    result["held_out_plane"] = held_out_plane
    result["split_seed"] = seed
    return result.sort_values(["experiment_split", "filename"]).reset_index(drop=True)


def build_matched_plane_splits(
    frame: pd.DataFrame,
    *,
    target_plane: str,
    seed: int,
    validation_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return equal-size LOPO and all-plane splits with a shared test set."""
    if target_plane not in TARGET_PLANES:
        raise ValueError(f"target_plane must be one of {TARGET_PLANES}")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")

    official_train = frame.loc[frame["official_split"] == "train"]
    test = frame.loc[
        (frame["official_split"] == "test") & (frame["plane_label"] == target_plane)
    ]
    lopo_development = official_train.loc[official_train["plane_label"] != target_plane]
    all_plane_development, _ = train_test_split(
        official_train,
        train_size=len(lopo_development),
        random_state=seed,
        stratify=official_train["tumor_label"] + "|" + official_train["plane_label"],
    )

    def make_split(development: pd.DataFrame, arm: str) -> pd.DataFrame:
        train, validation = train_test_split(
            development,
            test_size=validation_fraction,
            random_state=seed,
            stratify=development["tumor_label"] + "|" + development["plane_label"],
        )
        parts = []
        for name, part in (
            ("train", train),
            ("validation", validation),
            ("test", test),
        ):
            part = part.copy()
            part["experiment_split"] = name
            parts.append(part)
        result = pd.concat(parts, ignore_index=True)
        result["held_out_plane"] = target_plane if arm == "lopo" else "none"
        result["target_plane"] = target_plane
        result["comparison_arm"] = arm
        result["split_seed"] = seed
        return result.sort_values(["experiment_split", "filename"]).reset_index(
            drop=True
        )

    return make_split(lopo_development, "lopo"), make_split(
        all_plane_development, "all_planes"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="create one leakage-safe split manifest for a plane experiment."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "brisc2025" / "manifest.csv",
    )
    parser.add_argument("--held-out-plane", choices=PLANES, default="none")
    parser.add_argument(
        "--matched-pair",
        action="store_true",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "experiments",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    aligned = build_aligned_manifest(pd.read_csv(args.manifest))
    if args.matched_pair:
        if args.held_out_plane == "none":
            raise ValueError("--matched-pair requires a held-out plane")
        lopo, all_planes = build_matched_plane_splits(
            aligned,
            target_plane=args.held_out_plane,
            seed=args.seed,
            validation_fraction=args.validation_fraction,
        )
        splits = {"lopo": lopo, "all_planes": all_planes}
        root = args.output_root / f"matched_{args.held_out_plane}_seed{args.seed}"
        for arm, split in splits.items():
            output_path = root / arm / "split_manifest.csv"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            split.to_csv(output_path, index=False)
            counts = split["experiment_split"].value_counts()
            print(
                f"Saved {output_path}: train={counts.get('train', 0)}, "
                f"validation={counts.get('validation', 0)}, "
                f"test={counts.get('test', 0)}"
            )
        return

    split = build_experiment_split(
        aligned,
        held_out_plane=args.held_out_plane,
        seed=args.seed,
        validation_fraction=args.validation_fraction,
    )
    name = (
        "all_planes"
        if args.held_out_plane == "none"
        else f"leave_{args.held_out_plane}_out"
    )
    output_dir = args.output_root / f"{name}_seed{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "split_manifest.csv"
    split.to_csv(output_path, index=False)
    counts = split["experiment_split"].value_counts()
    print(
        f"Saved {output_path}: train={counts.get('train', 0)}, "
        f"validation={counts.get('validation', 0)}, test={counts.get('test', 0)}"
    )
    if args.held_out_plane != "none":
        print(
            "LOPO evaluation uses held-out official-training rows; "
            "official-test rows are excluded."
        )


if __name__ == "__main__":
    main()
