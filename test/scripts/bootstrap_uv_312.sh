#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/bootstrap_env_common.sh"

cd "$PROJECT_ROOT"

KERNEL_DISPLAY_PREFIX="Python 3.12 (Quant Stat)"
TORCH_VERSION="2.10.0"
TORCHVISION_VERSION="0.25.0"
TORCHAUDIO_VERSION="2.10.0"

ensure_gpu_server
detect_torch_index_url
print_existing_envs
prepare_env_identity "quant_uv" "py312"
mkdir -p .uv-cache
export UV_CACHE_DIR="$PROJECT_ROOT/.uv-cache"

echo "[bootstrap-uv-312] TORCH_INDEX_URL=$TORCH_INDEX_URL"

UV_NO_CONFIG=1 uv venv --python 3.12 "$VENV_PATH" --clear
source "$VENV_PATH/bin/activate"

# Install project dependencies first. If pyproject.toml/uv.lock installs a default torch build,
# the next step forcibly replaces it with the CUDA build matching this server.
UV_NO_CONFIG=1 uv sync --no-cache
uv pip install --reinstall ipykernel

# Force PyTorch to match the detected CUDA runtime, e.g. cu126 on RTX 4090 server.
uv pip install --reinstall --index-url "$TORCH_INDEX_URL" \
  "torch==$TORCH_VERSION" \
  "torchvision==$TORCHVISION_VERSION" \
  "torchaudio==$TORCHAUDIO_VERSION"

python - <<'PY'
import sys
import torch
print("[torch-check] python:", sys.executable)
print("[torch-check] torch:", torch.__version__)
print("[torch-check] torch cuda:", torch.version.cuda)
print("[torch-check] cuda available:", torch.cuda.is_available())
print("[torch-check] device count:", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit("[torch-check] CUDA is not available. Check NVIDIA driver and TORCH_INDEX_URL.")
print("[torch-check] device name:", torch.cuda.get_device_name(0))
PY

install_kernel "$KERNEL_DISPLAY_PREFIX [$ENV_NAME]"
run_smoke_test
print_runtime_summary
