#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/bootstrap_env_common.sh"

cd "$PROJECT_ROOT"

KERNEL_DISPLAY_PREFIX="Python 3.13 (Quant Stat)"
TORCH_VERSION="2.10.0"
TORCHVISION_VERSION="0.25.0"
TORCHAUDIO_VERSION="2.10.0"

if ! command -v python3.13 >/dev/null 2>&1; then
  echo "[bootstrap-venv-313] python3.13 not found on this server."
  exit 1
fi

ensure_gpu_server
detect_torch_index_url
print_existing_envs
prepare_env_identity "quant_venv" "py313"
mkdir -p .uv-cache
export UV_CACHE_DIR="$PROJECT_ROOT/.uv-cache"

echo "[bootstrap-venv-313] TORCH_INDEX_URL=$TORCH_INDEX_URL"

python3.13 -m venv --clear "$VENV_PATH"
source "$VENV_PATH/bin/activate"

python -m pip install --upgrade pip
UV_CACHE_DIR="$PROJECT_ROOT/.uv-cache" UV_NO_CONFIG=1 uv export --format requirements-txt --output requirements.txt
python -m pip install -r requirements.txt
python -m pip install --force-reinstall ipykernel

# Force PyTorch to match the detected CUDA runtime, e.g. cu126 on RTX 4090 server.
python -m pip install --force-reinstall --index-url "$TORCH_INDEX_URL" \
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
print_vscode_jupyter_connection_hint