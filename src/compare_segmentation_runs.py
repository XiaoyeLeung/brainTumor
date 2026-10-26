from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from decision_fusion import load_predictions

PRIMARY_METRICS = (
    "mean_dice_all",
    "mean_iou_all",
    "mean_dice_positive",
    "mean_iou_positive",
    "empty_mask_specificity",
    "balanced_dice",
    "complete_miss_rate",
    "empty_false_positive_rate",
    "mean_predicted_area_no_tumor",
)


def segmentation_metrics(frame: pd.DataFrame) -> dict[str, float | int | None]:
    positive = frame.loc[frame["ground_truth_mask_area_fraction"] > 0.0]
    empty = frame.loc[frame["ground_truth_mask_area_fraction"] == 0.0]
    if frame.empty:
        raise ValueError("Segmentation comparison requires at least one sample")

    positive_dice = (
        float(positive["mask_dice"].mean()) if not positive.empty else None
    )
    empty_specificity = (
        float((empty["predicted_mask_area_fraction"] == 0.0).mean())
        if not empty.empty
        else None
    )
    complete_misses = int(
        (positive["predicted_mask_area_fraction"] == 0.0).sum()
    )
    empty_false_positives = int(
        (empty["predicted_mask_area_fraction"] > 0.0).sum()
    )
    return {
        "sample_count": int(len(frame)),
        "positive_count": int(len(positive)),
        "empty_count": int(len(empty)),
        "mean_dice_all": float(frame["mask_dice"].mean()),
        "mean_iou_all": float(frame["mask_iou"].mean()),
        "mean_dice_positive": positive_dice,
        "mean_iou_positive": (
            float(positive["mask_iou"].mean()) if not positive.empty else None
        ),
        "empty_mask_specificity": empty_specificity,
        "balanced_dice": (
            (positive_dice + empty_specificity) / 2.0
            if positive_dice is not None and empty_specificity is not None
            else None
        ),
        "complete_miss_count": complete_misses,
        "complete_miss_rate": (
            float(complete_misses / len(positive)) if not positive.empty else None
        ),
        "empty_false_positive_count": empty_false_positives,
        "empty_false_positive_rate": (
            float(empty_false_positives / len(empty)) if not empty.empty else None
        ),
        "mean_predicted_area_no_tumor": (
            float(empty["predicted_mask_area_fraction"].mean())
            if not empty.empty
            else None
        ),
    }


def grouped_metrics(
    frame: pd.DataFrame, column: str
) -> dict[str, dict[str, float | int | None]]:
    if column not in frame:
        return {}
    return {
        str(value): segmentation_metrics(group.reset_index(drop=True))
        for value, group in frame.groupby(column, sort=True)
    }


def compare_runs(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    id_column: str = "filename",
) -> dict[str, object]:
    for name, frame in (("baseline", baseline), ("candidate", candidate)):
        if frame[id_column].duplicated().any():
            raise ValueError(f"{name} predictions contain duplicate {id_column}s")
    if set(baseline[id_column]) != set(candidate[id_column]):
        raise ValueError("Baseline and candidate prediction IDs differ")

    baseline = baseline.sort_values(id_column).reset_index(drop=True)
    candidate = candidate.sort_values(id_column).reset_index(drop=True)
    for column in ("tumor_label", "plane_label", "ground_truth_mask_area_fraction"):
        if column in baseline and column in candidate:
            if not baseline[column].equals(candidate[column]):
                raise ValueError(f"Baseline and candidate {column} values differ")

    baseline_metrics = segmentation_metrics(baseline)
    candidate_metrics = segmentation_metrics(candidate)
    deltas = {
        metric: (
            float(candidate_metrics[metric] - baseline_metrics[metric])
            if candidate_metrics[metric] is not None
            and baseline_metrics[metric] is not None
            else None
        )
        for metric in PRIMARY_METRICS
    }
    deltas["complete_miss_count"] = int(
        candidate_metrics["complete_miss_count"]
        - baseline_metrics["complete_miss_count"]
    )
    deltas["empty_false_positive_count"] = int(
        candidate_metrics["empty_false_positive_count"]
        - baseline_metrics["empty_false_positive_count"]
    )
    return {
        "baseline": {
            "metrics": baseline_metrics,
            "by_plane": grouped_metrics(baseline, "plane_label"),
            "by_class": grouped_metrics(baseline, "tumor_label"),
        },
        "candidate": {
            "metrics": candidate_metrics,
            "by_plane": grouped_metrics(candidate, "plane_label"),
            "by_class": grouped_metrics(candidate, "tumor_label"),
        },
        "delta_candidate_minus_baseline": deltas,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare paired scratch and pretrained segmentation predictions."
    )
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--candidate-run", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--id-column", default="filename")
    parser.add_argument(
        "--split",
        choices=("validation", "test"),
        default="test",
        help="Prediction split required in both run files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline = load_predictions(args.baseline_run, required_split=args.split)
    candidate = load_predictions(args.candidate_run, required_split=args.split)
    report = compare_runs(baseline, candidate, id_column=args.id_column)
    report["summary"] = {
        "baseline_run": str(args.baseline_run),
        "candidate_run": str(args.candidate_run),
        "prediction_split": args.split,
    }
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open("w") as file:
        json.dump(report, file, indent=2)
    print(json.dumps(report["delta_candidate_minus_baseline"], indent=2))
    print(f"Saved comparison to {args.output_file}")


if __name__ == "__main__":
    main()
