from __future__ import annotations

import argparse
import json
from pathlib import Path

from decision_fusion import calibrate_thresholds, load_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select fusion thresholds from validation predictions."
    )
    parser.add_argument("--classification-run", type=Path, required=True)
    parser.add_argument("--segmentation-run", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--id-column", default="filename")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = calibrate_thresholds(
        load_predictions(args.classification_run, required_split="validation"),
        load_predictions(args.segmentation_run, required_split="validation"),
        id_column=args.id_column,
    )
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open("w") as file:
        json.dump(result, file, indent=2)
    print(
        f"Selected classification threshold "
        f"{result['classification_threshold']:.6g} and area threshold "
        f"{result['area_threshold']:.6g}; saved {args.output_file}"
    )


if __name__ == "__main__":
    main()
