# Experiments

## all-plane baseline

Train and validate on images from axial, coronal, and sagittal planes. Evaluate on the official test set after development choices are frozen.

## Leave-one-plane-out experiment

The LOPO pipeline uses only official-training rows. For each fold, one plane is
held out for evaluation while the other two planes provide training and
validation data. Early stopping and threshold calibration never see the held-out
plane, and official-test rows are excluded.

## Size-matched all-plane control

Create a LOPO split and an equal-size all-plane split with the same official-test
target-plane images.

# Code

## Source code

- [calibrate_thresholds.py](src/calibrate_thresholds.py) Selects fusion thresholds from validation predictions.
- [compare_segmentation_runs.py](src/compare_segmentation_runs.py) Compares segmentation results from two runs.
- [data.py](src/data.py) Loads datasets, aligns masks, and defines image transforms.
- [decision_fusion.py](src/decision_fusion.py) Combines classifier and segmenter predictions.
- [evaluate.py](src/evaluate.py) Calculates final evaluation metrics.
- [models.py](src/models.py) Defines the neural network models
- [predict.py](src/predict.py) Runs saved models and exports predictions.
- [prepare_plane_split.py](src/prepare_plane_split.py) Creates plane-based experiment splits.
- [pretrain_segmentation.py](src/pretrain_segmentation.py) Pretrains U-Net through masked image reconstruction.
- [runtime.py](src/runtime.py) Handles seeds, devices, run paths, and reports.
- [segmentation_postprocessing.py](src/segmentation_postprocessing.py) Filters predicted masks and applies classifier gating.
- [train_classification.py](src/train_classification.py) — Trains the classifier
- [train_segmentation.py](src/train_segmentation.py) Trains segmenter
- [verify_dataset.py](src/verify_dataset.py) — Checks dataset files against the manifest

## Google Cloud

- `scripts/cloud/setup_vm.sh` set up environment on gcloud
- `scripts/cloud/run_development_pipeline.sh` runs training, validation without test set
- `scripts/cloud/run_final_test.sh` run the model on test set after the development pipeline is frozen.
- `scripts/cloud/run_lopo_pipeline.sh` runs three LOPO experiments
