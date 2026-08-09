"""서버 자원 프로필, 환경 로깅, 공유 FEATURE_SETS, 예측·평가.

8번에서 옮겼다. `ResourceProfile` dataclass와 자원 프로필 helper, `FEATURE_SETS`
(register_feature_groups가 mutate하는 단일 공유 dict), `predict`/`evaluate_predictions`를
담는다. `predict`는 models.make_loader를 사용한다.
"""

from __future__ import annotations

import json
import os
import platform
import sys
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from . import models


@dataclass(frozen=True)
class ResourceProfile:
    name: str
    device: str
    cpu_logical: int
    cpu_physical: int
    memory_gb: float
    available_memory_gb: float | None
    gpu_name: str | None
    gpu_memory_gb: float | None
    n_jobs: int
    max_workers: int
    num_workers: int
    torch_num_threads: int
    torch_interop_threads: int
    optuna_n_jobs: int
    batch_size: int | None
    pin_memory: bool


def detect_memory_gb() -> tuple[float, float | None]:
    try:
        import psutil

        vm = psutil.virtual_memory()
        return vm.total / 1024**3, vm.available / 1024**3
    except Exception:
        return float("nan"), None


def detect_gpu() -> tuple[str | None, float | None]:
    if not torch.cuda.is_available():
        return None, None
    index = torch.cuda.current_device()
    prop = torch.cuda.get_device_properties(index)
    return prop.name, prop.total_memory / 1024**3


def build_resource_profile(name: str, device_override: str | None = None) -> ResourceProfile:
    cpu_logical = os.cpu_count() or 1
    cpu_physical = min(16, cpu_logical)
    memory_gb, available_memory_gb = detect_memory_gb()
    gpu_name, gpu_memory_gb = detect_gpu()
    device = device_override or ("cuda" if torch.cuda.is_available() else "cpu")

    if name == "school_4090_15gb":
        return ResourceProfile(
            name=name,
            device=device,
            cpu_logical=cpu_logical,
            cpu_physical=cpu_physical,
            memory_gb=memory_gb,
            available_memory_gb=available_memory_gb,
            gpu_name=gpu_name,
            gpu_memory_gb=gpu_memory_gb,
            n_jobs=min(16, cpu_logical),
            max_workers=min(16, cpu_logical),
            num_workers=4,
            torch_num_threads=min(16, cpu_logical),
            torch_interop_threads=4,
            optuna_n_jobs=1 if device == "cuda" else min(4, cpu_logical),
            batch_size=48,
            pin_memory=device == "cuda",
        )

    return ResourceProfile(
        name=name,
        device=device,
        cpu_logical=cpu_logical,
        cpu_physical=cpu_physical,
        memory_gb=memory_gb,
        available_memory_gb=available_memory_gb,
        gpu_name=gpu_name,
        gpu_memory_gb=gpu_memory_gb,
        n_jobs=min(8, cpu_logical),
        max_workers=min(8, cpu_logical),
        num_workers=min(2, cpu_logical),
        torch_num_threads=min(8, cpu_logical),
        torch_interop_threads=2,
        optuna_n_jobs=1,
        batch_size=32,
        pin_memory=device == "cuda",
    )


def apply_resource_profile(profile: ResourceProfile) -> None:
    for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        os.environ[key] = str(profile.torch_num_threads)
    torch.set_num_threads(profile.torch_num_threads)
    try:
        torch.set_num_interop_threads(profile.torch_interop_threads)
    except RuntimeError as exc:
        print(f"[resource] torch interop threads already set: {exc}")


def log_environment(profile: ResourceProfile) -> dict[str, object]:
    env = {
        "python_executable": sys.executable,
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "cpu_logical": profile.cpu_logical,
        "cpu_physical": profile.cpu_physical,
        "memory_total_gb": profile.memory_gb,
        "memory_available_gb": profile.available_memory_gb,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": profile.gpu_name,
        "gpu_memory_gb": profile.gpu_memory_gb,
        "selected_resource_profile": profile.name,
        "applied_n_jobs": profile.n_jobs,
        "applied_max_workers": profile.max_workers,
        "applied_num_workers": profile.num_workers,
        "applied_torch_threads": profile.torch_num_threads,
        "applied_torch_interop_threads": profile.torch_interop_threads,
        "applied_optuna_n_jobs": profile.optuna_n_jobs,
        "applied_batch_size": profile.batch_size,
    }
    print(json.dumps(env, ensure_ascii=False, indent=2))
    return env


FEATURE_SETS: dict[str, list[str]] = {
    "returns_core": [
        "log_return_1",
        "return_4",
        "return_16",
        "realized_vol_16",
        "hl_range_pct",
        "volume_z_96",
        "value_z_96",
    ],
    "technical_liquidity": [
        "log_return_1",
        "return_4",
        "return_16",
        "realized_vol_16",
        "ema_gap_16",
        "ema_gap_64",
        "volume_z_96",
        "value_z_96",
        "turnover_proxy",
    ],
    "wide_stationary": [
        "log_return_1",
        "return_4",
        "return_16",
        "return_64",
        "realized_vol_16",
        "realized_vol_64",
        "hl_range_pct",
        "ema_gap_16",
        "ema_gap_64",
        "volume_z_96",
        "value_z_96",
        "turnover_proxy",
    ],
}


def predict(model: nn.Module, split: dict[str, np.ndarray], profile: ResourceProfile, batch_size: int) -> np.ndarray:
    device = torch.device(profile.device)
    loader = models.make_loader(split, batch_size, False, profile)
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in loader:
            pred = model(xb.to(device, non_blocking=True)).detach().cpu().numpy()
            preds.append(pred)
    return np.concatenate(preds)


def evaluate_predictions(split: dict[str, np.ndarray], pred_return: np.ndarray) -> dict[str, float]:
    actual_return = split["y"]
    prev_close = split["prev_close"]
    target_close = split["target_close"]
    pred_close = prev_close * np.exp(pred_return)
    persistence_close = prev_close
    price_error = pred_close - target_close
    persistence_error = persistence_close - target_close
    mae_krw = float(np.mean(np.abs(price_error)))
    rmse_krw = float(np.sqrt(np.mean(price_error**2)))
    persistence_mae_krw = float(np.mean(np.abs(persistence_error)))
    return_mae = float(np.mean(np.abs(pred_return - actual_return)))
    return_rmse = float(np.sqrt(np.mean((pred_return - actual_return) ** 2)))
    sign_agreement = float(np.mean(np.sign(pred_return) == np.sign(actual_return)))
    direction_accuracy = float(np.mean((pred_return > 0) == (actual_return > 0)))
    pred_var = float(np.var(pred_return))
    actual_var = float(np.var(actual_return))
    variance_ratio = pred_var / (actual_var + 1e-12)
    near_zero_share = float(np.mean(np.abs(pred_return) < 1e-4))
    persistence_gap = mae_krw - persistence_mae_krw
    copy_risk_ratio = mae_krw / (persistence_mae_krw + 1e-9)
    collapse_score = float((near_zero_share > 0.70) + (variance_ratio < 0.10) + (copy_risk_ratio > 0.95))
    return {
        "mae_krw": mae_krw,
        "rmse_krw": rmse_krw,
        "persistence_mae_krw": persistence_mae_krw,
        "persistence_gap_krw": persistence_gap,
        "copy_risk_ratio": float(copy_risk_ratio),
        "return_mae": return_mae,
        "return_rmse": return_rmse,
        "direction_accuracy": direction_accuracy,
        "sign_agreement": sign_agreement,
        "pred_return_std": float(np.std(pred_return)),
        "actual_return_std": float(np.std(actual_return)),
        "variance_ratio": float(variance_ratio),
        "near_zero_return_share": near_zero_share,
        "collapse_score": collapse_score,
    }
