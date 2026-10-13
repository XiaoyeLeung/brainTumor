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
PATIENCE="${BRAINTUMOR_EARLY_STOPPING_PATIENCE:-5}"
HEAD_EPOCHS="${BRAINTUMOR_CLASSIFICATION_HEAD_EPOCHS:-2}"
FINE_TUNE_EPOCHS="${BRAINTUMOR_CLASSIFICATION_FINE_TUNE_EPOCHS:-30}"
SEGMENTATION_EPOCHS="${BRAINTUMOR_SEGMENTATION_EPOCHS:-30}"
NUM_WORKERS="${BRAINTUMOR_NUM_WORKERS:-4}"

if [[ "${HELD_OUT_PLANE}" == "none" ]]; then
    EXPERIMENT_NAME="all_planes_seed${SEED}"
else
    EXPERIMENT_NAME="leave_${HELD_OUT_PLANE}_out_seed${SEED}"
fi

SPLIT_ROOT="${PROJECT_ROOT}/data/experiments/${EXPERIMENT_NAME}"
RESULT_ROOT="${PROJECT_ROOT}/results/${EXPERIMENT_NAME}"
CHECKPOINT_ROOT="${PROJECT_ROOT}/checkpoints/${EXPERIMENT_NAME}"
TRAIN_ROOT="${RESULT_ROOT}/training"
PREDICTION_ROOT="${RESULT_ROOT}/predictions"
REPORT_ROOT="${RESULT_ROOT}/reports"
LOG_ROOT="${RESULT_ROOT}/logs"
mkdir -p "${TRAIN_ROOT}" "${PREDICTION_ROOT}" "${REPORT_ROOT}" "${LOG_ROOT}"

exec > >(tee -a "${LOG_ROOT}/pipeline.log") 2>&1

echo "Experiment: ${EXPERIMENT_NAME}"
echo "Started: $(date --iso-8601=seconds)"
nvidia-smi
"${PYTHON_BIN}" -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print(torch.cuda.get_device_name(0))'
"${PYTHON_BIN}" src/verify_dataset.py

SPLIT_ARGUMENTS=(
    --held-out-plane "${HELD_OUT_PLANE}"
    --seed "${SEED}"
    --output-root "${PROJECT_ROOT}/data/experiments"
)
"${PYTHON_BIN}" src/prepare_plane_split.py "${SPLIT_ARGUMENTS[@]}"
cp "${SPLIT_ROOT}/split_manifest.csv" "${REPORT_ROOT}/split_manifest.csv"

"${PYTHON_BIN}" src/train_classification.py \
    --split-root "${SPLIT_ROOT}" \
    --output-dir "${TRAIN_ROOT}" \
    --checkpoint-dir "${CHECKPOINT_ROOT}" \
    --run-name classification \
    --image-size 256 \
    --batch-size 32 \
    --head-epochs "${HEAD_EPOCHS}" \
    --fine-tune-epochs "${FINE_TUNE_EPOCHS}" \
    --earlyStopping "${PATIENCE}" \
    --num-workers "${NUM_WORKERS}" \
    --device cuda \
    --seed "${SEED}"

"${PYTHON_BIN}" src/train_segmentation.py \
    --split-root "${SPLIT_ROOT}" \
    --output-dir "${TRAIN_ROOT}" \
    --checkpoint-dir "${CHECKPOINT_ROOT}" \
    --run-name segmentation \
    --image-size 256 \
    --batch-size 8 \
    --epochs "${SEGMENTATION_EPOCHS}" \
    --earlyStopping "${PATIENCE}" \
    --num-workers "${NUM_WORKERS}" \
    --device cuda \
    --seed "${SEED}"

for split in validation test; do
    "${PYTHON_BIN}" src/predict.py \
        --task classification \
        --checkpoint "${CHECKPOINT_ROOT}/classification/best.pt" \
        --split "${split}" \
        --split-root "${SPLIT_ROOT}" \
        --output-dir "${PREDICTION_ROOT}/classification_${split}" \
        --num-workers "${NUM_WORKERS}" \
        --device cuda

    "${PYTHON_BIN}" src/predict.py \
        --task segmentation \
        --checkpoint "${CHECKPOINT_ROOT}/segmentation/best.pt" \
        --split "${split}" \
        --split-root "${SPLIT_ROOT}" \
        --output-dir "${PREDICTION_ROOT}/segmentation_${split}" \
        --num-workers "${NUM_WORKERS}" \
        --device cuda
done

THRESHOLD_FILE="${REPORT_ROOT}/thresholds.json"
"${PYTHON_BIN}" src/calibrate_thresholds.py \
    --classification-run "${PREDICTION_ROOT}/classification_validation/run.json" \
    --segmentation-run "${PREDICTION_ROOT}/segmentation_validation/run.json" \
    --output-file "${THRESHOLD_FILE}"

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

"${PYTHON_BIN}" -m pip freeze > "${REPORT_ROOT}/environment.txt"
nvidia-smi > "${REPORT_ROOT}/nvidia-smi.txt"
git rev-parse HEAD > "${REPORT_ROOT}/git-commit.txt"
git status --porcelain > "${REPORT_ROOT}/git-status.txt"

echo "Finished: $(date --iso-8601=seconds)"
echo "Results: ${RESULT_ROOT}"
