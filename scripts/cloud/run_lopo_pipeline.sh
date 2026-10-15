#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ -f .venv-cloud/bin/activate ]]; then
    source .venv-cloud/bin/activate
fi

PYTHON_BIN="${BRAINTUMOR_PYTHON:-python}"
SEED="${BRAINTUMOR_SEED:-42}"
NUM_WORKERS="${BRAINTUMOR_NUM_WORKERS:-4}"
read -r -a PLANES <<< "${BRAINTUMOR_LOPO_PLANES:-axial coronal sagittal}"
export PYTHONUNBUFFERED=1

for HELD_OUT_PLANE in "${PLANES[@]}"; do
    case "${HELD_OUT_PLANE}" in
        axial|coronal|sagittal) ;;
        *)
            echo "Invalid held-out plane: ${HELD_OUT_PLANE}" >&2
            exit 2
            ;;
    esac

    EXPERIMENT_NAME="leave_${HELD_OUT_PLANE}_out_seed${SEED}"
    SPLIT_ROOT="${PROJECT_ROOT}/data/experiments/${EXPERIMENT_NAME}"
    RESULT_ROOT="${PROJECT_ROOT}/results/${EXPERIMENT_NAME}"
    CHECKPOINT_ROOT="${PROJECT_ROOT}/checkpoints/${EXPERIMENT_NAME}"
    PREDICTION_ROOT="${RESULT_ROOT}/predictions"
    REPORT_ROOT="${RESULT_ROOT}/reports"

    echo "LOPO fold started: hold out ${HELD_OUT_PLANE}"
    BRAINTUMOR_HELD_OUT_PLANE="${HELD_OUT_PLANE}" \
        BRAINTUMOR_REUSE_CLASSIFICATION_CHECKPOINT=0 \
        bash scripts/cloud/run_development_pipeline.sh

    "${PYTHON_BIN}" src/predict.py \
        --task classification \
        --checkpoint "${CHECKPOINT_ROOT}/classification/best.pt" \
        --split test \
        --split-root "${SPLIT_ROOT}" \
        --output-dir "${PREDICTION_ROOT}/classification_heldout" \
        --num-workers "${NUM_WORKERS}" \
        --device cuda

    "${PYTHON_BIN}" src/predict.py \
        --task segmentation \
        --checkpoint "${CHECKPOINT_ROOT}/segmentation/best.pt" \
        --split test \
        --split-root "${SPLIT_ROOT}" \
        --output-dir "${PREDICTION_ROOT}/segmentation_heldout" \
        --num-workers "${NUM_WORKERS}" \
        --device cuda

    CLASSIFICATION_THRESHOLD="$(
        "${PYTHON_BIN}" -c \
            'import json,sys; print(json.load(open(sys.argv[1]))["classification_threshold"])' \
            "${REPORT_ROOT}/thresholds.json"
    )"
    AREA_THRESHOLD="$(
        "${PYTHON_BIN}" -c \
            'import json,sys; print(json.load(open(sys.argv[1]))["area_threshold"])' \
            "${REPORT_ROOT}/thresholds.json"
    )"

    "${PYTHON_BIN}" src/decision_fusion.py \
        --classification-run "${PREDICTION_ROOT}/classification_heldout/run.json" \
        --segmentation-run "${PREDICTION_ROOT}/segmentation_heldout/run.json" \
        --classification-threshold "${CLASSIFICATION_THRESHOLD}" \
        --area-threshold "${AREA_THRESHOLD}" \
        --output-file "${REPORT_ROOT}/heldout_fusion.json"

    "${PYTHON_BIN}" src/evaluate.py \
        --classification-run "${PREDICTION_ROOT}/classification_heldout/run.json" \
        --segmentation-run "${PREDICTION_ROOT}/segmentation_heldout/run.json" \
        --fusion-report "${REPORT_ROOT}/heldout_fusion.json" \
        --output-file "${REPORT_ROOT}/heldout_metrics.json"

    echo "LOPO fold finished: ${HELD_OUT_PLANE}"
    echo "Metrics: ${REPORT_ROOT}/heldout_metrics.json"
done

echo "All requested LOPO folds finished. Official-test inference was not run."
