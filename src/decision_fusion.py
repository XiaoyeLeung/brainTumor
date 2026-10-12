from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

TYPE_A = "type_a_classification_no_tumor_segmentation_tumor"
TYPE_B = "type_b_classification_tumor_segmentation_no_tumor"
NO_CONTRADICTION = "none"


def _require_columns(frame: pd.DataFrame, columns: set[str], name: str) -> None:
    if missing := columns - set(frame.columns):
        raise ValueError(f"{name} predictions are missing columns: {sorted(missing)}")


def _validate_unit_interval(values: pd.Series, name: str) -> None:
    if values.isna().any() or not values.between(0.0, 1.0).all():
        raise ValueError(f"{name} must contain non-missing values in [0, 1]")


def load_predictions(
    run_file: Path,
    required_split: str | None = None,
) -> pd.DataFrame:
    with run_file.open() as file:
        report = json.load(file)
    if "predictions" not in report:
        raise ValueError(f"{run_file} does not contain a predictions array")
    if required_split is not None:
        actual_split = report.get("summary", {}).get("prediction_split")
        if actual_split != required_split:
            raise ValueError(
                f"{run_file} contains {actual_split!r} predictions; "
                f"expected {required_split!r}"
            )
    return pd.DataFrame(report["predictions"])


def select_balanced_accuracy_threshold(
    targets: pd.Series,
    scores: pd.Series,
) -> tuple[float, float]:
    """Return the threshold and balanced accuracy, favoring sensitivity on ties."""
    _validate_unit_interval(scores, "scores")
    labels = targets.astype(bool).to_numpy()
    if len(labels) == 0 or len(np.unique(labels)) != 2:
        raise ValueError("Threshold selection requires both tumor and no-tumor rows")

    values = scores.to_numpy(dtype=float)
    candidates = np.unique(np.concatenate(([0.0], values, [1.0])))
    results = [
        (
            float(threshold),
            float(balanced_accuracy_score(labels, values >= threshold)),
        )
        for threshold in candidates
    ]
    return max(results, key=lambda result: (result[1], -result[0]))


def calibrate_thresholds(
    classification: pd.DataFrame,
    segmentation: pd.DataFrame,
    *,
    id_column: str = "filename",
) -> dict[str, object]:
    """Select tumor-presence thresholds from aligned validation predictions."""
    _require_columns(
        classification,
        {id_column, "tumor_label", "no_tumor_probability"},
        "Classification",
    )
    _require_columns(
        segmentation,
        {id_column, "tumor_label", "predicted_mask_area_fraction"},
        "Segmentation",
    )
    if (
        classification[id_column].duplicated().any()
        or segmentation[id_column].duplicated().any()
    ):
        raise ValueError(f"{id_column!r} values must be unique in both files")

    aligned = classification[
        [id_column, "tumor_label", "no_tumor_probability"]
    ].merge(
        segmentation[
            [id_column, "tumor_label", "predicted_mask_area_fraction"]
        ],
        on=id_column,
        how="outer",
        validate="one_to_one",
        indicator=True,
        suffixes=("_classification", "_segmentation"),
    )
    if not (aligned["_merge"] == "both").all():
        examples = aligned.loc[aligned["_merge"] != "both", id_column].head(5)
        raise ValueError(
            "Classification and segmentation predictions do not cover the same "
            f"validation samples. Examples: {examples.tolist()}"
        )
    if not (
        aligned["tumor_label_classification"]
        == aligned["tumor_label_segmentation"]
    ).all():
        raise ValueError("Classification and segmentation tumor labels differ")

    targets = aligned["tumor_label_classification"] != "no_tumor"
    classification_scores = 1.0 - aligned["no_tumor_probability"]
    area_scores = aligned["predicted_mask_area_fraction"]
    classification_threshold, classification_score = (
        select_balanced_accuracy_threshold(targets, classification_scores)
    )
    area_threshold, segmentation_score = select_balanced_accuracy_threshold(
        targets, area_scores
    )
    return {
        "classification_threshold": classification_threshold,
        "area_threshold": area_threshold,
        "classification_validation_balanced_accuracy": classification_score,
        "segmentation_validation_balanced_accuracy": segmentation_score,
        "validation_sample_count": len(aligned),
        "tie_break": "smallest_threshold",
    }


def fuse_predictions(
    classification: pd.DataFrame,
    segmentation: pd.DataFrame,
    *,
    classification_threshold: float,
    area_threshold: float,
    id_column: str = "filename",
) -> pd.DataFrame:
    for value, name in (
        (classification_threshold, "classification_threshold"),
        (area_threshold, "area_threshold"),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], received {value}")

    _require_columns(
        classification,
        {id_column, "predicted_class", "no_tumor_probability"},
        "Classification",
    )
    _require_columns(
        segmentation,
        {id_column, "predicted_mask_area_fraction"},
        "Segmentation",
    )
    if (
        classification[id_column].duplicated().any()
        or segmentation[id_column].duplicated().any()
    ):
        raise ValueError(f"{id_column!r} values must be unique in both files")

    result = classification.merge(
        segmentation[[id_column, "predicted_mask_area_fraction"]],
        on=id_column,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not (result["_merge"] == "both").all():
        examples = result.loc[result["_merge"] != "both", id_column].head(5).tolist()
        raise ValueError(
            "Classification and segmentation predictions do not cover the same "
            f"samples. Examples: {examples}"
        )
    result = result.drop(columns="_merge")

    _validate_unit_interval(result["no_tumor_probability"], "no_tumor_probability")
    _validate_unit_interval(
        result["predicted_mask_area_fraction"],
        "predicted_mask_area_fraction",
    )
    result["classification_tumor_probability"] = 1.0 - result[
        "no_tumor_probability"
    ]
    result["classification_says_tumor"] = (
        result["classification_tumor_probability"] >= classification_threshold
    )
    result["segmentation_says_tumor"] = (
        result["predicted_mask_area_fraction"] >= area_threshold
    )
    result["is_uncertain"] = (
        result["classification_says_tumor"] != result["segmentation_says_tumor"]
    )
    result["contradiction_type"] = NO_CONTRADICTION
    result.loc[
        ~result["classification_says_tumor"] & result["segmentation_says_tumor"],
        "contradiction_type",
    ] = TYPE_A
    result.loc[
        result["classification_says_tumor"] & ~result["segmentation_says_tumor"],
        "contradiction_type",
    ] = TYPE_B

    if "tumor_label" in result:
        result["ground_truth_tumor"] = result["tumor_label"] != "no_tumor"
        result["classification_class_correct"] = (
            result["predicted_class"] == result["tumor_label"]
        )
        for model in ("classification", "segmentation"):
            result[f"{model}_presence_correct"] = (
                result[f"{model}_says_tumor"] == result["ground_truth_tumor"]
            )
    return result


def summarize(
    frame: pd.DataFrame,
    *,
    classification_threshold: float,
    area_threshold: float,
) -> dict[str, object]:
    total = len(frame)
    uncertain = int(frame["is_uncertain"].sum())
    summary: dict[str, object] = {
        "classification_threshold": classification_threshold,
        "area_threshold": area_threshold,
        "sample_count": total,
        "consistent_count": total - uncertain,
        "inconsistent_count": uncertain,
        "agreement_rate": (total - uncertain) / total if total else None,
        "uncertain_rate": uncertain / total if total else None,
        "type_a_count": int((frame["contradiction_type"] == TYPE_A).sum()),
        "type_b_count": int((frame["contradiction_type"] == TYPE_B).sum()),
    }

    if "ground_truth_tumor" in frame:
        for name, subset in (
            ("consistent", frame.loc[~frame["is_uncertain"]]),
            ("inconsistent", frame.loc[frame["is_uncertain"]]),
        ):
            for column in (
                "classification_class_correct",
                "classification_presence_correct",
                "segmentation_presence_correct",
            ):
                summary[f"{column.removesuffix('_correct')}_error_rate_{name}"] = (
                    float((~subset[column]).mean()) if not subset.empty else None
                )
    return summary


def save_report(
    frame: pd.DataFrame,
    summary: dict[str, object],
    output_file: Path,
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    records = frame.astype(object).where(frame.notna(), None).to_dict(orient="records")
    with output_file.open("w") as file:
        json.dump({"summary": summary, "samples": records}, file, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fuse two models' decisions into one JSON audit report."
    )
    parser.add_argument("--classification-run", type=Path, required=True)
    parser.add_argument("--segmentation-run", type=Path, required=True)
    parser.add_argument("--classification-threshold", type=float, required=True)
    parser.add_argument("--area-threshold", type=float, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--id-column", default="filename")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    decisions = fuse_predictions(
        load_predictions(args.classification_run),
        load_predictions(args.segmentation_run),
        classification_threshold=args.classification_threshold,
        area_threshold=args.area_threshold,
        id_column=args.id_column,
    )
    summary = summarize(
        decisions,
        classification_threshold=args.classification_threshold,
        area_threshold=args.area_threshold,
    )
    save_report(decisions, summary, args.output_file)
    print(
        f"Saved {summary['sample_count']} decisions and "
        f"{summary['inconsistent_count']} inconsistencies to {args.output_file}"
    )


if __name__ == "__main__":
    main()
