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
