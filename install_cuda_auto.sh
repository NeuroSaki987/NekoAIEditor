#!/usr/bin/env bash
set -euo pipefail
python -m pip install --upgrade pip

# Prefer the current PyTorch CUDA wheel indexes. Override with:
#   NEKOAI_PYTORCH_CUDA_INDEX=https://download.pytorch.org/whl/cu132 bash install_cuda_auto.sh
indices=()
if [[ -n "${NEKOAI_PYTORCH_CUDA_INDEX:-}" ]]; then
  indices+=("${NEKOAI_PYTORCH_CUDA_INDEX}")
fi
indices+=(
  "https://download.pytorch.org/whl/cu132"
  "https://download.pytorch.org/whl/cu130"
  "https://download.pytorch.org/whl/cu126"
  "https://download.pytorch.org/whl/cu128"
)

ok=0
for idx in "${indices[@]}"; do
  echo "Trying PyTorch CUDA index: ${idx}"
  if python -m pip install torch torchvision torchaudio --index-url "${idx}"; then
    ok=1
    break
  fi
  echo "Failed index ${idx}; trying next..."
done

if [[ "${ok}" != "1" ]]; then
  echo "All CUDA wheel attempts failed. Install from https://pytorch.org/get-started/locally/ and rerun: python -m pip install -r requirements.txt" >&2
  exit 1
fi

python -m pip install -r requirements.txt
python - <<'PY'
import torch
print('Torch:', torch.__version__)
print('Torch CUDA runtime:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(i), 'CC', torch.cuda.get_device_capability(i))
PY
