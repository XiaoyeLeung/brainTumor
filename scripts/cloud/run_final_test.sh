#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ -f .venv-cloud/bin/activate ]]; then
    source .venv-cloud/bin/activate
fi

PYTHON_BIN="${BRAINTUMOR_PYTHON:-python}"
SEED="${BRAINTUMOR_SEED:-42}"
HELD_OUT_PLANE="${BRAINTUMOR_HELD_OUT_PLANE:-none}"
NUM_WORKERS="${BRAINTUMOR_NUM_WORKERS:-4}"

if [[ "${HELD_OUT_PLANE}" != "none" ]]; then
    echo "run_final_test.sh is only for the all-plane official-test run." >&2
    echo "Use run_lopo_pipeline.sh for held-out-plane evaluation." >&2
    exit 2
fi

if [[ "${HELD_OUT_PLANE}" == "none" ]]; then
    EXPERIMENT_NAME="all_planes_seed${SEED}"
else
    EXPERIMENT_NAME="leave_${HELD_OUT_PLANE}_out_seed${SEED}"
fi

SPLIT_ROOT="${PROJECT_ROOT}/data/experiments/${EXPERIMENT_NAME}"
RESULT_ROOT="${PROJECT_ROOT}/results/${EXPERIMENT_NAME}"
CHECKPOINT_ROOT="${PROJECT_ROOT}/checkpoints/${EXPERIMENT_NAME}"
PREDICTION_ROOT="${RESULT_ROOT}/predictions"
REPORT_ROOT="${RESULT_ROOT}/reports"
LOG_ROOT="${RESULT_ROOT}/logs"
THRESHOLD_FILE="${REPORT_ROOT}/thresholds.json"

for required_file in \
    "${SPLIT_ROOT}/split_manifest.csv" \
    "${CHECKPOINT_ROOT}/classification/best.pt" \
    "${CHECKPOINT_ROOT}/segmentation/best.pt" \
    "${THRESHOLD_FILE}"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "Missing development artifact: ${required_file}" >&2
        exit 2
    fi
done

mkdir -p "${PREDICTION_ROOT}" "${REPORT_ROOT}" "${LOG_ROOT}"
exec > >(tee -a "${LOG_ROOT}/final_test.log") 2>&1

echo "FINAL TEST: ${EXPERIMENT_NAME}"
echo "Started: $(date --iso-8601=seconds)"

"${PYTHON_BIN}" src/predict.py \
    --task classification \
    --checkpoint "${CHECKPOINT_ROOT}/classification/best.pt" \
    --split test \
    --split-root "${SPLIT_ROOT}" \
    --output-dir "${PREDICTION_ROOT}/classification_test" \
    --num-workers "${NUM_WORKERS}" \
    --device cuda

"${PYTHON_BIN}" src/predict.py \
    --task segmentation \
    --checkpoint "${CHECKPOINT_ROOT}/segmentation/best.pt" \
    --split test \
    --split-root "${SPLIT_ROOT}" \
    --output-dir "${PREDICTION_ROOT}/segmentation_test" \
    --classification-run "${PREDICTION_ROOT}/classification_test/run.json" \
    --postprocessing-config "${THRESHOLD_FILE}" \
    --num-workers "${NUM_WORKERS}" \
    --device cuda

CLASSIFICATION_THRESHOLD="$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["classification_threshold"])' "${THRESHOLD_FILE}")"
AREA_THRESHOLD="$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["area_threshold"])' "${THRESHOLD_FILE}")"
FUSION_FILE="${REPORT_ROOT}/test_fusion.json"

"${PYTHON_BIN}" src/decision_fusion.py \
    --classification-run "${PREDICTION_ROOT}/classification_test/run.json" \
    --segmentation-run "${PREDICTION_ROOT}/segmentation_test/run.json" \
    --classification-threshold "${CLASSIFICATION_THRESHOLD}" \
    --area-threshold "${AREA_THRESHOLD}" \
    --output-file "${FUSION_FILE}"

"${PYTHON_BIN}" src/evaluate.py \
    --classification-run "${PREDICTION_ROOT}/classification_test/run.json" \
    --segmentation-run "${PREDICTION_ROOT}/segmentation_test/run.json" \
    --fusion-report "${FUSION_FILE}" \
    --output-file "${REPORT_ROOT}/test_metrics.json"

echo "Final test finished: $(date --iso-8601=seconds)"
echo "Do not tune the model after viewing ${REPORT_ROOT}/test_metrics.json"
