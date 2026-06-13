#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/bootstrap_env_common.sh"

cd "$PROJECT_ROOT"

KERNEL_DISPLAY_PREFIX="Python 3.12 (Quant Stat)"
TORCH_VERSION="2.10.0"
TORCHVISION_VERSION="0.25.0"
TORCHAUDIO_VERSION="2.10.0"

if ! command -v python3.12 >/dev/null 2>&1; then
  echo "[bootstrap-venv-312] python3.12 not found on this server."
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[bootstrap-venv-312] nvidia-smi not found. GPU server check failed."
  exit 1
fi

detect_torch_index_url
print_existing_envs
prepare_env_identity "quant_venv" "py312"
mkdir -p .uv-cache
export UV_CACHE_DIR="$PROJECT_ROOT/.uv-cache"

python3.12 -m venv --clear "$VENV_PATH"
source "$VENV_PATH/bin/activate"

python -m pip install --upgrade pip
UV_CACHE_DIR="$PROJECT_ROOT/.uv-cache" uv export --format requirements-txt --output requirements.txt
python -m pip install -r requirements.txt
python -m pip install ipykernel
python -m pip install --index-url "$TORCH_INDEX_URL" \
  "torch==$TORCH_VERSION" \
  "torchvision==$TORCHVISION_VERSION" \
  "torchaudio==$TORCHAUDIO_VERSION"

install_kernel "$KERNEL_DISPLAY_PREFIX [$ENV_NAME]"
run_smoke_test
print_runtime_summary
