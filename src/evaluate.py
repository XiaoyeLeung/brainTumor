from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from data import CLASS_NAMES
from decision_fusion import load_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute final classification, segmentation, and fusion metrics."
    )
    parser.add_argument("--classification-run", type=Path, required=True)
    parser.add_argument("--segmentation-run", type=Path, required=True)
    parser.add_argument("--fusion-report", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--id-column", default="filename")
    return parser.parse_args()


def classification_metrics(frame: pd.DataFrame) -> dict[str, float]:
    return {
        "accuracy": float(
            accuracy_score(frame["tumor_label"], frame["predicted_class"])
        ),
        "macro_f1": float(
            f1_score(
                frame["tumor_label"],
                frame["predicted_class"],
                labels=list(CLASS_NAMES),
                average="macro",
                zero_division=0,
            )
        ),
    }


def segmentation_metrics(frame: pd.DataFrame) -> dict[str, float | None]:
    positive = frame.loc[frame["tumor_label"] != "no_tumor"]
    negative = frame.loc[frame["tumor_label"] == "no_tumor"]
    if positive.empty:
        raise ValueError("Segmentation evaluation requires tumor-positive samples")
    return {
        "mean_dice_all": float(frame["mask_dice"].mean()),
        "mean_iou_all": float(frame["mask_iou"].mean()),
        "mean_dice_positive": float(positive["mask_dice"].mean()),
        "mean_iou_positive": float(positive["mask_iou"].mean()),
        "empty_mask_specificity": (
            float((negative["predicted_mask_area_fraction"] == 0.0).mean())
            if not negative.empty
            else None
        ),
        "mean_predicted_area_no_tumor": (
            float(negative["predicted_mask_area_fraction"].mean())
            if not negative.empty
            else None
        ),
    }


def evaluate(
    classification: pd.DataFrame,
    segmentation: pd.DataFrame,
    fusion_report: dict[str, object],
    *,
    id_column: str,
) -> dict[str, object]:
    if classification[id_column].duplicated().any():
        raise ValueError(f"Classification predictions contain duplicate {id_column}s")
    if segmentation[id_column].duplicated().any():
        raise ValueError(f"Segmentation predictions contain duplicate {id_column}s")
    if set(classification[id_column]) != set(segmentation[id_column]):
        raise ValueError("Classification and segmentation prediction IDs differ")

    fusion_samples = pd.DataFrame(fusion_report["samples"])
    if set(fusion_samples[id_column]) != set(classification[id_column]):
        raise ValueError("Fusion and prediction IDs differ")

    covered_ids = set(fusion_samples.loc[~fusion_samples["is_uncertain"], id_column])
    covered_classification = classification.loc[
        classification[id_column].isin(covered_ids)
    ]
    covered_segmentation = segmentation.loc[segmentation[id_column].isin(covered_ids)]
    if covered_classification.empty:
        raise ValueError("Fusion rejected every sample; covered metrics are undefined")

    summary = fusion_report["summary"]
    result: dict[str, object] = {
        "sample_count": len(classification),
        "classification": classification_metrics(classification),
        "segmentation": segmentation_metrics(segmentation),
        "fusion": {
            "coverage": float(len(covered_ids) / len(classification)),
            "agreement_rate": summary["agreement_rate"],
            "uncertain_rate": summary["uncertain_rate"],
            "covered_classification": classification_metrics(covered_classification),
            "covered_segmentation": segmentation_metrics(covered_segmentation),
        },
        "by_plane": {},
    }

    if "plane_label" in classification and "plane_label" in segmentation:
        planes = sorted(classification["plane_label"].unique())
        for plane in planes:
            result["by_plane"][plane] = {
                "classification": classification_metrics(
                    classification.loc[classification["plane_label"] == plane]
                ),
                "segmentation": segmentation_metrics(
                    segmentation.loc[segmentation["plane_label"] == plane]
                ),
            }
    return result


def main() -> None:
    args = parse_args()
    classification = load_predictions(args.classification_run, required_split="test")
    segmentation = load_predictions(args.segmentation_run, required_split="test")
    with args.fusion_report.open() as file:
        fusion_report = json.load(file)

    result = evaluate(
        classification,
        segmentation,
        fusion_report,
        id_column=args.id_column,
    )
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open("w") as file:
        json.dump(result, file, indent=2)
    print(json.dumps(result, indent=2))
    print(f"Saved final metrics to {args.output_file}")


if __name__ == "__main__":
    main()
