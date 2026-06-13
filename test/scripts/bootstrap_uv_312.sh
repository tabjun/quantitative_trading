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

uv venv --python 3.12 "$VENV_PATH" --clear
source "$VENV_PATH/bin/activate"

uv sync
uv pip install ipykernel
uv pip install --index-url "$TORCH_INDEX_URL" \
  "torch==$TORCH_VERSION" \
  "torchvision==$TORCHVISION_VERSION" \
  "torchaudio==$TORCHAUDIO_VERSION"

install_kernel "$KERNEL_DISPLAY_PREFIX [$ENV_NAME]"
print_runtime_summary
