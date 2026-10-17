Dear my friend Chrysi ;)
Thanks !!!!
To use github: 
```
git clone https://github.com/XiaoyeLeung/brainTumor.git
cd brainTumor

conda create -n brainTumor python=3.12
conda activate brainTumor
pip install -r requirements.txt
```

The directory should have the structure:
```
brainTumor/
├── brisc2025/
├── data/
├── notebooks/
├── src/
└── requirements.txt
```


Official test data must remain untouched.

HAVE A NICE DAY!!!

## Google Cloud full-dataset run

The repository keeps four cloud scripts:

- `scripts/cloud/setup_vm.sh` creates the Python environment and verifies CUDA.
- `scripts/cloud/run_development_pipeline.sh` runs training, validation inference,
  and threshold calibration without official-test inference.
- `scripts/cloud/run_final_test.sh` performs the one-time official-test inference,
  fusion, and evaluation after the development pipeline is frozen.
- `scripts/cloud/run_lopo_pipeline.sh` runs three held-out-plane experiments using
  official-training data only.

After the GPU VM starts, connect to it and prepare the repository:

```bash
git clone https://github.com/XiaoyeLeung/brainTumor.git
cd brainTumor
bash scripts/cloud/setup_vm.sh
```

Copy the local `brisc2025/` directory into `~/brainTumor/` on the VM, then
verify it:

```bash
source .venv-cloud/bin/activate
python src/verify_dataset.py
```

Start development training in a persistent background process:

```bash
nohup bash scripts/cloud/run_development_pipeline.sh > cloud-run.log 2>&1 < /dev/null &
tail -f cloud-run.log
```

The development run performs:

```text
full split -> classification training -> segmentation training
-> validation inference -> validation threshold calibration
```

Segmentation training uses a false-positive-aware Tversky loss, gives extra BCE
weight to no-tumor images, and selects checkpoints using the mean of tumor Dice
and empty-mask specificity. Inference removes connected components smaller than
64 pixels and clears masks when the classifier assigns at least 0.90 probability
to `no_tumor`. The validation run records these post-processing settings in
`thresholds.json`; final-test and LOPO inference load that configuration rather
than duplicating it.

The defaults can be overridden before a development run:

```bash
BRAINTUMOR_SEGMENTATION_FP_WEIGHT=0.7 \
BRAINTUMOR_EMPTY_MASK_WEIGHT=2.0 \
BRAINTUMOR_MIN_COMPONENT_PIXELS=64 \
BRAINTUMOR_NO_TUMOR_GATE_THRESHOLD=0.9 \
bash scripts/cloud/run_development_pipeline.sh
```

It records the Python environment, GPU information, Git commit, checkpoints,
validation predictions, thresholds, and a combined pipeline log. It does not run
inference on official-test images.

After all development decisions are frozen, run the official test exactly once:

```bash
bash scripts/cloud/run_final_test.sh
```

The final evaluation uses all 1,000 official-test images. Do not tune the model
after viewing the result.

After training, download `results/` and `checkpoints/` before stopping the VM.
Stopping the VM stops GPU and CPU charges, but its persistent disk continues to
incur storage charges.

## Leave-one-plane-out experiment

The LOPO pipeline uses only official-training rows. For each fold, one plane is
held out for evaluation while the other two planes provide training and
validation data. Early stopping and threshold calibration never see the held-out
plane, and official-test rows are excluded.

Run all three folds sequentially on the VM:

```bash
nohup bash scripts/cloud/run_lopo_pipeline.sh > lopo-run.log 2>&1 < /dev/null &
tail -f lopo-run.log
```

To run only one fold, set `BRAINTUMOR_LOPO_PLANES`, for example:

```bash
BRAINTUMOR_LOPO_PLANES=axial bash scripts/cloud/run_lopo_pipeline.sh
```

Each fold writes `heldout_metrics.json` and `heldout_fusion.json` under
`results/leave_<plane>_out_seed42/reports/`. The official-test pipeline remains
a separate, explicit final step.
