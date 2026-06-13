#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "$ROOT_DIR/.." && pwd)"
ENV_ROOT="$PROJECT_ROOT/.venvs"
KERNEL_ROOT="$HOME/.local/share/jupyter/kernels"
TIMESTAMP="$(TZ=Asia/Seoul date +%Y%m%d_%H%M%S)"

ensure_gpu_server() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[bootstrap] nvidia-smi not found. GPU server check failed."
    exit 1
  fi
}

detect_torch_index_url() {
  local cuda_version

  cuda_version="$(nvidia-smi | grep -oP 'CUDA Version:\s*\K[0-9]+\.[0-9]+' | head -n 1)"

  if [[ -z "$cuda_version" ]]; then
    echo "[bootstrap] Failed to detect CUDA version from nvidia-smi."
    nvidia-smi
    exit 1
  fi

  case "$cuda_version" in
    12.6)
      TORCH_INDEX_URL="https://download.pytorch.org/whl/cu126"
      ;;
    12.4)
      TORCH_INDEX_URL="https://download.pytorch.org/whl/cu124"
      ;;
    12.1)
      TORCH_INDEX_URL="https://download.pytorch.org/whl/cu121"
      ;;
    *)
      echo "[bootstrap] Unsupported CUDA version: $cuda_version"
      echo "[bootstrap] nvidia-smi output:"
      nvidia-smi
      exit 1
      ;;
  esac

  export TORCH_INDEX_URL
  CUDA_VERSION="$cuda_version"
  export CUDA_VERSION

  echo "[bootstrap] CUDA_VERSION=$cuda_version"
  echo "[bootstrap] TORCH_INDEX_URL=$TORCH_INDEX_URL"
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

run_smoke_test() {
  echo "[bootstrap] Running smoke test..."
  python - <<'PY'
import importlib
import json
import sys
from pathlib import Path

packages = [
    "dotenv",
    "numpy",
    "pandas",
    "matplotlib",
    "scipy",
    "duckdb",
    "statsmodels",
    "optuna",
    "torch",
    "ipykernel",
    "openai",
    "google.generativeai",
    "fastdtw",
]

failed = []
for name in packages:
    try:
        importlib.import_module(name)
        print(f"[smoke] import ok: {name}")
    except Exception as exc:
        failed.append((name, repr(exc)))
        print(f"[smoke] import failed: {name} -> {exc}")

try:
    import torch
    print(f"[smoke] torch version: {torch.__version__}")
    print(f"[smoke] torch cuda version: {torch.version.cuda}")
    print(f"[smoke] torch cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[smoke] torch device count: {torch.cuda.device_count()}")
        print(f"[smoke] torch current device: {torch.cuda.current_device()}")
        print(f"[smoke] torch device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
except Exception as exc:
    failed.append(("torch_cuda_check", repr(exc)))
    print(f"[smoke] torch cuda check failed: {exc}")

kernel_name = Path.home() / ".local" / "share" / "jupyter" / "kernels"
if not kernel_name.exists():
    failed.append(("jupyter_kernel_dir", "kernel directory missing"))
    print("[smoke] kernel directory missing")
else:
    print(f"[smoke] kernel directory ok: {kernel_name}")

if failed:
    print("[smoke] bootstrap smoke test failed")
    for name, detail in failed:
        print(f"  - {name}: {detail}")
    sys.exit(1)

print("[smoke] bootstrap smoke test passed")
PY
}

print_vscode_jupyter_connection_hint() {
  local raw_url
  local path_query
  local external_url

  raw_url="$(jupyter server list 2>/dev/null | grep -Eo 'https?://[^ ]+\?token=[^ ]+' | head -n 1 || true)"

  if [[ -z "$raw_url" ]]; then
    echo "[vscode] No running Jupyter server URL found."
    echo "[vscode] Run: jupyter server list"
    return 0
  fi

  path_query="$(echo "$raw_url" | sed -E 's#^https?://[^/]+##')"
  external_url="https://stat5.kmu.ac.kr:9500${path_query}"

  echo
  echo "[vscode] Copy this Jupyter server URL:"
  echo "$external_url"
  echo
  echo "[vscode] VSCode connection order:"
  echo "  Select Kernel"
  echo "  -> Existing Jupyter Server"
  echo "  -> paste the URL above"
  echo "  -> select the registered kernel"
  echo
}