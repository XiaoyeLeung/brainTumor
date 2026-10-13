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

The repository keeps only two cloud scripts:

- `scripts/cloud/setup_vm.sh` creates the Python environment and verifies CUDA.
- `scripts/cloud/run_full_pipeline.sh` runs training, inference, calibration,
  fusion, and final evaluation.

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

Start the full all-plane baseline in a persistent background process:

```bash
nohup bash scripts/cloud/run_full_pipeline.sh > cloud-run.log 2>&1 < /dev/null &
tail -f cloud-run.log
```

The default run performs the full sequence:

```text
full split -> classification training -> segmentation training
-> validation/test inference -> validation threshold calibration
-> test fusion -> final metrics
```

It records the Python environment, GPU information, Git commit, checkpoints,
predictions, thresholds, fusion report, metrics, and a combined pipeline log.
The final evaluation uses all 1,000 official-test images. Freeze the pipeline
before that evaluation and do not tune it after viewing the result.

After training, download `results/` and `checkpoints/` before stopping the VM.
Stopping the VM stops GPU and CPU charges, but its persistent disk continues to
incur storage charges.
