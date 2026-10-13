#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

python3 -m venv --system-site-packages .venv-cloud
source .venv-cloud/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-cloud.txt

python -c 'import torch, torchvision; assert torch.cuda.is_available(), "CUDA is unavailable"; print(f"torch={torch.__version__}, torchvision={torchvision.__version__}, GPU={torch.cuda.get_device_name(0)}")'
python -m compileall -q src
if [[ -d tests ]]; then
    python -m pytest -q
else
    echo "No tracked tests directory in this clone; source compilation passed."
fi

echo "Cloud environment is ready at ${PROJECT_ROOT}/.venv-cloud"
