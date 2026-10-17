#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Controlled experiment: longer masked reconstruction followed by full U-Net
# transfer. Keep training loss and validation prediction behavior neutral so
# balanced-Dice checkpoint selection is the only false-positive intervention.
export BRAINTUMOR_EXPERIMENT_TAG="${BRAINTUMOR_EXPERIMENT_TAG:-pretrain60_t2_balanced}"
export BRAINTUMOR_MASKED_PRETRAINING=1
export BRAINTUMOR_PRETRAINING_EPOCHS="${BRAINTUMOR_PRETRAINING_EPOCHS:-60}"
export BRAINTUMOR_PRETRAINING_EARLY_STOPPING_PATIENCE="${BRAINTUMOR_PRETRAINING_EARLY_STOPPING_PATIENCE:-10}"
export BRAINTUMOR_PRETRAINING_TRANSFER_SCOPE=encoder_decoder

export BRAINTUMOR_SEGMENTATION_EPOCHS="${BRAINTUMOR_SEGMENTATION_EPOCHS:-60}"
export BRAINTUMOR_SEGMENTATION_EARLY_STOPPING_PATIENCE="${BRAINTUMOR_SEGMENTATION_EARLY_STOPPING_PATIENCE:-10}"
export BRAINTUMOR_SEGMENTATION_SELECTION_METRIC=balanced_dice
export BRAINTUMOR_SEGMENTATION_FP_WEIGHT=0.5
export BRAINTUMOR_EMPTY_MASK_WEIGHT=1.0
export BRAINTUMOR_SEGMENTATION_POSTPROCESSING=0

exec "${SCRIPT_DIR}/run_development_pipeline.sh"
