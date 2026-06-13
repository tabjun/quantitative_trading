#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "$ROOT_DIR/.." && pwd)"
ENV_ROOT="$PROJECT_ROOT/.venvs"
KERNEL_ROOT="$HOME/.local/share/jupyter/kernels"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

ensure_gpu_server() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[bootstrap] nvidia-smi not found. GPU server check failed."
    exit 1
  fi
}

detect_torch_index_url() {
  local cuda_version
  cuda_version="$(nvidia-smi | awk -F': ' '/CUDA Version/ {print $2; exit}' | awk '{print $1}')"
  case "$cuda_version" in
    12.6*|12.7*|12.8*|13.*)
      TORCH_INDEX_URL="https://download.pytorch.org/whl/cu126"
      ;;
    11.8*|11.7*|11.6*)
      TORCH_INDEX_URL="https://download.pytorch.org/whl/cu118"
      ;;
    *)
      echo "[bootstrap] Unsupported CUDA version: ${cuda_version:-unknown}"
      exit 1
      ;;
  esac
  CUDA_VERSION="$cuda_version"
}

print_existing_envs() {
  mkdir -p "$ENV_ROOT"
  mkdir -p "$KERNEL_ROOT"
  echo "[bootstrap] Existing env directories under $ENV_ROOT"
  find "$ENV_ROOT" -mindepth 1 -maxdepth 1 -type d -printf "  - %f -> %p\n" 2>/dev/null || true
  echo "[bootstrap] Existing Jupyter kernels under $KERNEL_ROOT"
  find "$KERNEL_ROOT" -mindepth 1 -maxdepth 1 -type d -printf "  - %f -> %p\n" 2>/dev/null || true
}

prepare_env_identity() {
  local prefix="$1"
  local py_tag="$2"
  local default_name="${prefix}_${py_tag}_${TIMESTAMP}"

  ENV_NAME="${ENV_NAME:-$default_name}"
  KERNEL_NAME="${KERNEL_NAME:-$ENV_NAME}"
  VENV_PATH="$ENV_ROOT/$ENV_NAME"
}

print_cleanup_help() {
  echo "[bootstrap] To remove this env later:"
  echo "  rm -rf \"$VENV_PATH\""
  echo "[bootstrap] To remove this kernel later:"
  echo "  jupyter kernelspec uninstall \"$KERNEL_NAME\""
  echo "[bootstrap] Activation command:"
  echo "  source \"$VENV_PATH/bin/activate\""
}

install_kernel() {
  local display_name="$1"
  local kernel_dir="$KERNEL_ROOT/$KERNEL_NAME"
  if [ -d "$kernel_dir" ]; then
    rm -rf "$kernel_dir"
  fi
  python -m ipykernel install --user --name "$KERNEL_NAME" --display-name "$display_name"
}

print_runtime_summary() {
  echo "[bootstrap] CUDA version: $CUDA_VERSION"
  echo "[bootstrap] Torch index: $TORCH_INDEX_URL"
  echo "[bootstrap] Env name: $ENV_NAME"
  echo "[bootstrap] Env path: $VENV_PATH"
  echo "[bootstrap] Kernel name: $KERNEL_NAME"
  python --version
  python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
  print_cleanup_help
}
