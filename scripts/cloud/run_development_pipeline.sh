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
DEFAULT_PATIENCE="${BRAINTUMOR_EARLY_STOPPING_PATIENCE:-5}"
CLASSIFICATION_PATIENCE="${BRAINTUMOR_CLASSIFICATION_EARLY_STOPPING_PATIENCE:-${DEFAULT_PATIENCE}}"
PRETRAINING_PATIENCE="${BRAINTUMOR_PRETRAINING_EARLY_STOPPING_PATIENCE:-${DEFAULT_PATIENCE}}"
SEGMENTATION_PATIENCE="${BRAINTUMOR_SEGMENTATION_EARLY_STOPPING_PATIENCE:-${DEFAULT_PATIENCE}}"
HEAD_EPOCHS="${BRAINTUMOR_CLASSIFICATION_HEAD_EPOCHS:-2}"
FINE_TUNE_EPOCHS="${BRAINTUMOR_CLASSIFICATION_FINE_TUNE_EPOCHS:-30}"
SEGMENTATION_EPOCHS="${BRAINTUMOR_SEGMENTATION_EPOCHS:-30}"
MASKED_PRETRAINING="${BRAINTUMOR_MASKED_PRETRAINING:-0}"
PRETRAINING_EPOCHS="${BRAINTUMOR_PRETRAINING_EPOCHS:-30}"
PRETRAINING_BATCH_SIZE="${BRAINTUMOR_PRETRAINING_BATCH_SIZE:-8}"
PRETRAINING_PATCH_SIZE="${BRAINTUMOR_PRETRAINING_PATCH_SIZE:-16}"
PRETRAINING_MASK_RATIO="${BRAINTUMOR_PRETRAINING_MASK_RATIO:-0.35}"
PRETRAINING_NOISE_STD="${BRAINTUMOR_PRETRAINING_NOISE_STD:-0.05}"
PRETRAINING_LR="${BRAINTUMOR_PRETRAINING_LR:-0.001}"
PRETRAINING_TRANSFER_SCOPE="${BRAINTUMOR_PRETRAINING_TRANSFER_SCOPE:-encoder_decoder}"
SEGMENTATION_FP_WEIGHT="${BRAINTUMOR_SEGMENTATION_FP_WEIGHT:-0.7}"
EMPTY_MASK_WEIGHT="${BRAINTUMOR_EMPTY_MASK_WEIGHT:-2.0}"
SEGMENTATION_SELECTION_METRIC="${BRAINTUMOR_SEGMENTATION_SELECTION_METRIC:-balanced_dice}"
SEGMENTATION_POSTPROCESSING="${BRAINTUMOR_SEGMENTATION_POSTPROCESSING:-1}"
MIN_COMPONENT_PIXELS="${BRAINTUMOR_MIN_COMPONENT_PIXELS:-64}"
NO_TUMOR_GATE_THRESHOLD="${BRAINTUMOR_NO_TUMOR_GATE_THRESHOLD:-0.9}"
NUM_WORKERS="${BRAINTUMOR_NUM_WORKERS:-4}"
EXPERIMENT_TAG="${BRAINTUMOR_EXPERIMENT_TAG:-}"
export PYTHONUNBUFFERED=1

case "${PRETRAINING_TRANSFER_SCOPE}" in
    encoder | encoder_decoder) ;;
    *)
        echo "BRAINTUMOR_PRETRAINING_TRANSFER_SCOPE must be encoder or encoder_decoder" >&2
        exit 2
        ;;
esac

if [[ "${HELD_OUT_PLANE}" == "none" ]]; then
    SPLIT_NAME="all_planes_seed${SEED}"
else
    SPLIT_NAME="leave_${HELD_OUT_PLANE}_out_seed${SEED}"
fi

EXPERIMENT_NAME="${SPLIT_NAME}${EXPERIMENT_TAG:+_${EXPERIMENT_TAG}}"
SPLIT_ROOT="${PROJECT_ROOT}/data/experiments/${SPLIT_NAME}"
RESULT_ROOT="${PROJECT_ROOT}/results/${EXPERIMENT_NAME}"
CHECKPOINT_ROOT="${PROJECT_ROOT}/checkpoints/${EXPERIMENT_NAME}"
TRAIN_ROOT="${RESULT_ROOT}/training"
PREDICTION_ROOT="${RESULT_ROOT}/predictions"
REPORT_ROOT="${RESULT_ROOT}/reports"
LOG_ROOT="${RESULT_ROOT}/logs"
mkdir -p "${TRAIN_ROOT}" "${PREDICTION_ROOT}" "${REPORT_ROOT}" "${LOG_ROOT}"

exec > >(tee -a "${LOG_ROOT}/development_pipeline.log") 2>&1

echo "Experiment: ${EXPERIMENT_NAME}"
echo "Masked U-Net pretraining: ${MASKED_PRETRAINING}"
echo "Pretraining epochs: ${PRETRAINING_EPOCHS}"
echo "Pretraining transfer scope: ${PRETRAINING_TRANSFER_SCOPE}"
echo "Segmentation selection metric: ${SEGMENTATION_SELECTION_METRIC}"
echo "Segmentation epochs: ${SEGMENTATION_EPOCHS}"
echo "Segmentation postprocessing: ${SEGMENTATION_POSTPROCESSING}"
echo "Development pipeline started: $(date --iso-8601=seconds)"
nvidia-smi
"${PYTHON_BIN}" -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print(torch.cuda.get_device_name(0))'
"${PYTHON_BIN}" src/verify_dataset.py

"${PYTHON_BIN}" src/prepare_plane_split.py \
    --held-out-plane "${HELD_OUT_PLANE}" \
    --seed "${SEED}" \
    --output-root "${PROJECT_ROOT}/data/experiments"
cp "${SPLIT_ROOT}/split_manifest.csv" "${REPORT_ROOT}/split_manifest.csv"

CLASSIFICATION_CHECKPOINT="${BRAINTUMOR_CLASSIFICATION_CHECKPOINT:-${CHECKPOINT_ROOT}/classification/best.pt}"
if [[ "${BRAINTUMOR_REUSE_CLASSIFICATION_CHECKPOINT:-0}" == "1" ]]; then
    if [[ ! -f "${CLASSIFICATION_CHECKPOINT}" ]]; then
        echo "Requested classification checkpoint reuse, but it is missing: ${CLASSIFICATION_CHECKPOINT}" >&2
        exit 2
    fi
    echo "Reusing classification checkpoint: ${CLASSIFICATION_CHECKPOINT}"
else
    "${PYTHON_BIN}" src/train_classification.py \
        --split-root "${SPLIT_ROOT}" \
        --output-dir "${TRAIN_ROOT}" \
        --checkpoint-dir "${CHECKPOINT_ROOT}" \
        --run-name classification \
        --image-size 256 \
        --batch-size 32 \
        --head-epochs "${HEAD_EPOCHS}" \
        --fine-tune-epochs "${FINE_TUNE_EPOCHS}" \
        --earlyStopping "${CLASSIFICATION_PATIENCE}" \
        --num-workers "${NUM_WORKERS}" \
        --device cuda \
        --seed "${SEED}"
fi

SEGMENTATION_INITIALIZATION_ARGS=()
case "${MASKED_PRETRAINING}" in
    0) ;;
    1)
        "${PYTHON_BIN}" src/pretrain_segmentation.py \
            --split-root "${SPLIT_ROOT}" \
            --output-dir "${TRAIN_ROOT}" \
            --checkpoint-dir "${CHECKPOINT_ROOT}" \
            --run-name segmentation_pretraining \
            --image-size 256 \
            --batch-size "${PRETRAINING_BATCH_SIZE}" \
            --epochs "${PRETRAINING_EPOCHS}" \
            --lr "${PRETRAINING_LR}" \
            --patch-size "${PRETRAINING_PATCH_SIZE}" \
            --mask-ratio "${PRETRAINING_MASK_RATIO}" \
            --noise-std "${PRETRAINING_NOISE_STD}" \
            --earlyStopping "${PRETRAINING_PATIENCE}" \
            --num-workers "${NUM_WORKERS}" \
            --device cuda \
            --seed "${SEED}"
        SEGMENTATION_INITIALIZATION_ARGS=(
            --initial-checkpoint "${CHECKPOINT_ROOT}/segmentation_pretraining/best.pt"
            --initialization-scope "${PRETRAINING_TRANSFER_SCOPE}"
        )
        ;;
    *)
        echo "BRAINTUMOR_MASKED_PRETRAINING must be 0 or 1" >&2
        exit 2
        ;;
esac

"${PYTHON_BIN}" src/train_segmentation.py \
    --split-root "${SPLIT_ROOT}" \
    --output-dir "${TRAIN_ROOT}" \
    --checkpoint-dir "${CHECKPOINT_ROOT}" \
    --run-name segmentation \
    --image-size 256 \
    --batch-size 8 \
    --epochs "${SEGMENTATION_EPOCHS}" \
    --false-positive-weight "${SEGMENTATION_FP_WEIGHT}" \
    --empty-mask-weight "${EMPTY_MASK_WEIGHT}" \
    --selection-metric "${SEGMENTATION_SELECTION_METRIC}" \
    --earlyStopping "${SEGMENTATION_PATIENCE}" \
    --num-workers "${NUM_WORKERS}" \
    --device cuda \
    --seed "${SEED}" \
    "${SEGMENTATION_INITIALIZATION_ARGS[@]}"

"${PYTHON_BIN}" src/predict.py \
    --task classification \
    --checkpoint "${CLASSIFICATION_CHECKPOINT}" \
    --split validation \
    --split-root "${SPLIT_ROOT}" \
    --output-dir "${PREDICTION_ROOT}/classification_validation" \
    --num-workers "${NUM_WORKERS}" \
    --device cuda

SEGMENTATION_PREDICTION_ARGS=()
case "${SEGMENTATION_POSTPROCESSING}" in
    0) ;;
    1)
        SEGMENTATION_PREDICTION_ARGS=(
            --classification-run "${PREDICTION_ROOT}/classification_validation/run.json"
            --min-component-pixels "${MIN_COMPONENT_PIXELS}"
            --no-tumor-gate-threshold "${NO_TUMOR_GATE_THRESHOLD}"
        )
        ;;
    *)
        echo "BRAINTUMOR_SEGMENTATION_POSTPROCESSING must be 0 or 1" >&2
        exit 2
        ;;
esac

"${PYTHON_BIN}" src/predict.py \
    --task segmentation \
    --checkpoint "${CHECKPOINT_ROOT}/segmentation/best.pt" \
    --split validation \
    --split-root "${SPLIT_ROOT}" \
    --output-dir "${PREDICTION_ROOT}/segmentation_validation" \
    --num-workers "${NUM_WORKERS}" \
    --device cuda \
    "${SEGMENTATION_PREDICTION_ARGS[@]}"

"${PYTHON_BIN}" src/calibrate_thresholds.py \
    --classification-run "${PREDICTION_ROOT}/classification_validation/run.json" \
    --segmentation-run "${PREDICTION_ROOT}/segmentation_validation/run.json" \
    --output-file "${REPORT_ROOT}/thresholds.json"

"${PYTHON_BIN}" -m pip freeze > "${REPORT_ROOT}/environment.txt"
nvidia-smi > "${REPORT_ROOT}/nvidia-smi.txt"
git rev-parse HEAD > "${REPORT_ROOT}/git-commit.txt"
git status --porcelain > "${REPORT_ROOT}/git-status.txt"

echo "Development pipeline finished: $(date --iso-8601=seconds)"
echo "Official-test inference has not been run."
echo "Results: ${RESULT_ROOT}"
