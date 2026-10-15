from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from data import build_aligned_segmentation_frame

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLANES = ("none", "axial", "coronal", "sagittal")


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create one leakage-safe split manifest for a plane experiment."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "brisc2025" / "manifest.csv",
    )
    parser.add_argument("--held-out-plane", choices=PLANES, default="none")
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
