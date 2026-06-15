# [FOR COMMIT TRACKING ONLY - DO NOT EXECUTE]
# This file is automatically mirrored from the corresponding .ipynb for git diff purposes.
# Actual research execution should be performed in the Jupyter Notebook (.ipynb)
# or in an approved remote/server environment.

# %% [markdown]
# # 8번 최적화 breadth training 실험
#
# 6번 안정화 기준과 7번 stage plan을 실제 GPU 학습 backend로 연결하는 노트북입니다.
# 이번 버전은 알고리즘뿐 아니라 preprocessing, normalization, loss, optimizer/scheduler, gradient policy, ensemble 축을 함께 비교합니다.
# 실제 실행은 학교 서버 CUDA 커널에서 수행하고, 결과 CSV/PNG/Markdown 보고서를 `test/results`와 `test/images`에 저장합니다.

# %%
# [FOR COMMIT TRACKING ONLY - DO NOT EXECUTE]
# This file is automatically mirrored from the corresponding .ipynb for git diff purposes.
# Actual research execution should be performed in the Jupyter Notebook (.ipynb)
# or in an approved remote/server environment.

# %% [markdown]
# # 8번 최적화 breadth training 실험
#
# 6번 안정화 기준과 7번 stage plan을 실제 GPU 학습 backend로 연결하는 노트북입니다.
# 이번 버전은 알고리즘뿐 아니라 preprocessing, normalization, loss, optimizer/scheduler, gradient policy, ensemble 축을 함께 비교합니다.
# 실제 실행은 학교 서버 CUDA 커널에서 수행하고, 결과 CSV/PNG/Markdown 보고서를 `test/results`와 `test/images`에 저장합니다.

# %%
#!/usr/bin/env python
# coding: utf-8

"""8번: 6번 안정화 기준을 반영한 실제 breadth training 실험.

이 파일은 notebook source와 동명 .py mirror를 함께 유지하기 위한 서버 실행용 코드다.
Codex 로컬 세션에서는 heavy training을 실행하지 않고, 학교 서버 CUDA 커널에서 실행한다.

핵심 목적:
- 6번의 target/normalization/loss/model-selection 기준을 실제 학습에 반영한다.
- 7번의 stage plan에서 빠졌던 확장 모델군과 앙상블 학습 backend를 구현한다.
- 학습 곡선, 예측 그래프, collapse 진단 그래프, leaderboard를 파일로 저장한다.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# 0. 경로와 재현성
# ---------------------------------------------------------------------------


def find_repo_root(start: Path) -> Path:
    def is_repo_root(path: Path) -> bool:
        return (path / "pyproject.toml").exists() and (path / "test").exists()

    env_root = os.environ.get("QUANTITATIVE_TRADING_ROOT")
    candidates = []
    if env_root:
        candidates.append(Path(env_root))
    candidates.extend([start, *start.parents])
    candidates.extend(
        [
            Path.home() / "personal_ai_project" / "quantitative_trading",
            Path.home() / "OneDrive" / "바탕 화면" / "Analysis" / "toy_agent_project" / "quantitative_trading",
        ]
    )
    for path in candidates:
        if is_repo_root(path):
            return path
    return start


REPO_ROOT = find_repo_root(Path.cwd())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RESULTS_DIR = REPO_ROOT / "test" / "results"
IMAGES_DIR = REPO_ROOT / "test" / "images"
ARTIFACT_PREFIX = "8_optimization_breadth_training"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# 1. 서버 자원 프로필
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# 2. 데이터 로딩과 기초 통계
# ---------------------------------------------------------------------------


def resolve_db_path(db_arg: str | None) -> Path:
    if db_arg:
        candidate = Path(db_arg)
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"지정한 DuckDB 파일이 없다: {candidate}")

    candidate = REPO_ROOT / "data" / "upbit_data.db"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        "DuckDB 파일을 찾지 못했다. 기본 경로는 repo root/data/upbit_data.db 이며, 필요하면 --db로만 오버라이드한다."
    )


def load_price_data(db_path: Path, table: str, ticker: str | None, max_rows: int | None) -> pd.DataFrame:
    import duckdb

    where = ""
    params: list[object] = []
    with duckdb.connect(str(db_path), read_only=True) as con:
        columns = [row[1] for row in con.execute(f"PRAGMA table_info('{table}')").fetchall()]
        if ticker and "ticker" in columns:
            where = "WHERE ticker = ?"
            params.append(ticker)
        query = f"""
        SELECT *
        FROM {table}
        {where}
        ORDER BY timestamp
        """
        if max_rows:
            query += f" LIMIT {int(max_rows)}"
        df = con.execute(query, params).fetchdf()
    if df.empty:
        raise ValueError(f"{table}에서 데이터를 읽지 못했다. ticker={ticker!r}")
    return df


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    aliases = {
        "candle_date_time_kst": "timestamp",
        "trade_price": "close",
        "opening_price": "open",
        "high_price": "high",
        "low_price": "low",
        "candle_acc_trade_volume": "volume",
        "candle_acc_trade_price": "value",
    }
    for old, new in aliases.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]
    required = ["timestamp", "open", "high", "low", "close", "volume"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"필수 가격 컬럼 누락: {missing}")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


def rolling_z(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=max(8, window // 4)).mean()
    std = series.rolling(window, min_periods=max(8, window // 4)).std()
    return (series - mean) / std.replace(0, np.nan)


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

PREPROCESSING_MODES = [
    "none",
    "winsorize",
    "asinh",
    "diff",
    "ema_residual",
    "frequency_highpass",
]

NORMALIZATION_MODES = [
    "none",
    "global_standard",
    "window_standard",
    "window_robust",
    "window_minmax",
    "revin",
    "robust_revin",
    "asinh_revin",
]

LOSS_MODES = [
    "return_mse",
    "return_mae",
    "return_huber",
    "return_huber_directional",
    "vol_weighted_huber",
    "anti_collapse_huber",
    "quantile_pinball",
]

OPTIMIZER_MODES = ["adamw", "adam", "rmsprop", "sgd_momentum"]
SCHEDULER_MODES = ["none", "cosine", "onecycle", "plateau"]
GRADIENT_POLICIES = ["none", "clip0.5", "clip1", "clip5", "adaptive"]


def make_features(raw: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(raw)
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)
    value = df["value"].astype(float) if "value" in df.columns else close * volume

    df["log_close"] = np.log(close.clip(lower=1e-9))
    df["log_return_1"] = df["log_close"].diff()
    df["return_4"] = df["log_close"].diff(4)
    df["return_16"] = df["log_close"].diff(16)
    df["return_64"] = df["log_close"].diff(64)
    df["realized_vol_16"] = df["log_return_1"].rolling(16, min_periods=8).std()
    df["realized_vol_64"] = df["log_return_1"].rolling(64, min_periods=16).std()
    df["hl_range_pct"] = (high - low) / close.replace(0, np.nan)
    df["ema_gap_16"] = close / close.ewm(span=16, adjust=False).mean() - 1.0
    df["ema_gap_64"] = close / close.ewm(span=64, adjust=False).mean() - 1.0
    df["volume_z_96"] = rolling_z(np.log1p(volume), 96)
    df["value_z_96"] = rolling_z(np.log1p(value), 96)
    df["turnover_proxy"] = np.log1p(value).diff()
    df["target_return"] = df["log_return_1"].shift(-1)
    df["prev_close"] = close
    df["target_close"] = close.shift(-1)
    return df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)


def basic_statistics(df: pd.DataFrame) -> dict[str, object]:
    ret = df["log_return_1"]
    stats = {
        "rows": int(len(df)),
        "start": str(df["timestamp"].min()),
        "end": str(df["timestamp"].max()),
        "close_mean": float(df["close"].mean()),
        "close_std": float(df["close"].std()),
        "close_min": float(df["close"].min()),
        "close_max": float(df["close"].max()),
        "return_mean": float(ret.mean()),
        "return_std": float(ret.std()),
        "return_skew": float(ret.skew()),
        "return_kurtosis": float(ret.kurtosis()),
        "missing_cells": int(df.isna().sum().sum()),
    }
    return stats


def moving_average_np(window: np.ndarray, kernel: int) -> np.ndarray:
    if kernel <= 1:
        return window.copy()
    pad = kernel // 2
    padded = np.pad(window, ((pad, pad), (0, 0)), mode="edge")
    out = np.empty_like(window)
    weights = np.ones(kernel, dtype=np.float32) / kernel
    for col in range(window.shape[1]):
        out[:, col] = np.convolve(padded[:, col], weights, mode="valid")[: window.shape[0]]
    return out


def apply_window_preprocessing(window: np.ndarray, mode: str) -> np.ndarray:
    """논문/사례 기반 preprocessing 후보를 window 단위로 적용한다."""

    if mode == "none":
        return window
    if mode == "winsorize":
        lo = np.nanquantile(window, 0.01, axis=0, keepdims=True)
        hi = np.nanquantile(window, 0.99, axis=0, keepdims=True)
        return np.clip(window, lo, hi)
    if mode == "asinh":
        median = np.nanmedian(window, axis=0, keepdims=True)
        iqr = np.nanquantile(window, 0.75, axis=0, keepdims=True) - np.nanquantile(window, 0.25, axis=0, keepdims=True)
        return np.arcsinh((window - median) / np.where(iqr < 1e-6, 1.0, iqr))
    if mode == "diff":
        diff = np.diff(window, axis=0, prepend=window[:1])
        return diff
    if mode == "ema_residual":
        trend = moving_average_np(window, kernel=min(9, max(3, window.shape[0] // 8 * 2 + 1)))
        return window - trend
    if mode == "frequency_highpass":
        fft = np.fft.rfft(window, axis=0)
        cutoff = min(3, fft.shape[0])
        fft[:cutoff] = 0
        return np.fft.irfft(fft, n=window.shape[0], axis=0).astype(np.float32)
    raise ValueError(f"지원하지 않는 preprocessing: {mode}")


def apply_window_normalization(window: np.ndarray, mode: str) -> np.ndarray:
    """정규화는 단순 standard뿐 아니라 robust/reversible/frequency 계열도 비교한다."""

    if mode == "none" or mode == "global_standard":
        return window
    if mode in {"window_standard", "revin"}:
        mean = window.mean(axis=0, keepdims=True)
        std = window.std(axis=0, keepdims=True)
        return (window - mean) / np.where(std < 1e-6, 1.0, std)
    if mode in {"window_robust", "robust_revin"}:
        median = np.nanmedian(window, axis=0, keepdims=True)
        iqr = np.nanquantile(window, 0.75, axis=0, keepdims=True) - np.nanquantile(window, 0.25, axis=0, keepdims=True)
        return (window - median) / np.where(iqr < 1e-6, 1.0, iqr)
    if mode == "window_minmax":
        lo = np.nanmin(window, axis=0, keepdims=True)
        hi = np.nanmax(window, axis=0, keepdims=True)
        return 2.0 * (window - lo) / np.where((hi - lo) < 1e-6, 1.0, hi - lo) - 1.0
    if mode == "asinh_revin":
        median = np.nanmedian(window, axis=0, keepdims=True)
        iqr = np.nanquantile(window, 0.75, axis=0, keepdims=True) - np.nanquantile(window, 0.25, axis=0, keepdims=True)
        return np.arcsinh((window - median) / np.where(iqr < 1e-6, 1.0, iqr))
    raise ValueError(f"지원하지 않는 normalization: {mode}")


def build_windows(
    df: pd.DataFrame,
    feature_columns: list[str],
    seq_len: int,
    preprocessing: str,
    normalization: str,
    max_windows: int | None,
    stride: int,
) -> dict[str, np.ndarray]:
    values = df[feature_columns].to_numpy(np.float32)
    targets = df["target_return"].to_numpy(np.float32)
    prev_close = df["prev_close"].to_numpy(np.float32)
    target_close = df["target_close"].to_numpy(np.float32)
    timestamps = df["timestamp"].astype(str).to_numpy()

    end_indices = list(range(seq_len, len(df), max(1, stride)))
    if max_windows and len(end_indices) > max_windows:
        end_indices = end_indices[-max_windows:]

    xs, ys, prevs, closes, times = [], [], [], [], []
    for end in end_indices:
        window = values[end - seq_len : end].copy()
        window = apply_window_preprocessing(window, preprocessing)
        window = apply_window_normalization(window, normalization)
        xs.append(window)
        ys.append(targets[end])
        prevs.append(prev_close[end])
        closes.append(target_close[end])
        times.append(timestamps[end])

    x = np.stack(xs).astype(np.float32)
    y = np.asarray(ys, dtype=np.float32)
    prev = np.asarray(prevs, dtype=np.float32)
    close = np.asarray(closes, dtype=np.float32)
    if normalization == "global_standard":
        flat = x.reshape(-1, x.shape[-1])
        mean = flat.mean(axis=0, keepdims=True)
        std = flat.std(axis=0, keepdims=True)
        x = (x - mean.reshape(1, 1, -1)) / np.where(std.reshape(1, 1, -1) < 1e-6, 1.0, std.reshape(1, 1, -1))
    return {"x": x, "y": y, "prev_close": prev, "target_close": close, "timestamp": np.asarray(times)}


def time_split(data: dict[str, np.ndarray], train_ratio: float, val_ratio: float) -> dict[str, dict[str, np.ndarray]]:
    n = len(data["y"])
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    splits = {
        "train": slice(0, train_end),
        "val": slice(train_end, val_end),
        "test": slice(val_end, n),
    }
    return {name: {key: value[idx] for key, value in data.items()} for name, idx in splits.items()}


# ---------------------------------------------------------------------------
# 3. 모델군
# ---------------------------------------------------------------------------


class LinearForecaster(nn.Module):
    def __init__(self, seq_len: int, n_features: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(seq_len * n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class RecurrentForecaster(nn.Module):
    def __init__(self, n_features: int, hidden: int, kind: str = "lstm"):
        super().__init__()
        rnn_cls = nn.LSTM if kind == "lstm" else nn.GRU
        self.rnn = rnn_cls(n_features, hidden, batch_first=True, num_layers=1)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return self.head(out[:, -1]).squeeze(-1)


class TCNForecaster(nn.Module):
    def __init__(self, n_features: int, hidden: int = 96, levels: int = 3):
        super().__init__()
        layers: list[nn.Module] = []
        in_ch = n_features
        for level in range(levels):
            dilation = 2**level
            layers.extend(
                [
                    nn.Conv1d(in_ch, hidden, kernel_size=3, padding=dilation, dilation=dilation),
                    nn.ReLU(),
                    nn.Conv1d(hidden, hidden, kernel_size=1),
                    nn.ReLU(),
                ]
            )
            in_ch = hidden
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x.transpose(1, 2))[..., : x.shape[1]]
        return self.head(z[:, :, -1]).squeeze(-1)


class TransformerForecaster(nn.Module):
    def __init__(self, seq_len: int, n_features: int, hidden: int = 96, heads: int = 4):
        super().__init__()
        self.proj = nn.Linear(n_features, hidden)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, hidden))
        layer = nn.TransformerEncoderLayer(hidden, heads, hidden * 2, batch_first=True, dropout=0.1)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.proj(x) + self.pos[:, : x.shape[1]]
        z = self.encoder(z)
        return self.head(z[:, -1]).squeeze(-1)


class DLinearLike(nn.Module):
    def __init__(self, seq_len: int, n_features: int):
        super().__init__()
        self.seasonal = nn.Linear(seq_len, 1)
        self.trend = nn.Linear(seq_len, 1)
        self.feature_head = nn.Linear(n_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        trend = F.avg_pool1d(x.transpose(1, 2), kernel_size=5, stride=1, padding=2).transpose(1, 2)
        seasonal = x - trend
        y = self.seasonal(seasonal.transpose(1, 2)).squeeze(-1)
        t = self.trend(trend.transpose(1, 2)).squeeze(-1)
        return self.feature_head(y + t).squeeze(-1)


class NLinearLike(nn.Module):
    def __init__(self, seq_len: int, n_features: int):
        super().__init__()
        self.linear = nn.Linear(seq_len, 1)
        self.feature_head = nn.Linear(n_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = x[:, -1:, :]
        z = x - base
        y = self.linear(z.transpose(1, 2)).squeeze(-1)
        return self.feature_head(y).squeeze(-1)


class PatchTSTLike(nn.Module):
    def __init__(self, seq_len: int, n_features: int, hidden: int = 96, patch_len: int = 8, heads: int = 4):
        super().__init__()
        self.patch_len = patch_len
        self.n_patches = max(1, seq_len // patch_len)
        self.proj = nn.Linear(n_features * patch_len, hidden)
        layer = nn.TransformerEncoderLayer(hidden, heads, hidden * 2, batch_first=True, dropout=0.1)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, f = x.shape
        usable = self.n_patches * self.patch_len
        z = x[:, -usable:].reshape(b, self.n_patches, f * self.patch_len)
        z = self.encoder(self.proj(z))
        return self.head(z.mean(dim=1)).squeeze(-1)


class AutoformerLike(nn.Module):
    def __init__(self, seq_len: int, n_features: int, hidden: int = 96, heads: int = 4):
        super().__init__()
        self.season_proj = nn.Linear(n_features, hidden)
        self.trend_head = nn.Linear(n_features, 1)
        layer = nn.TransformerEncoderLayer(hidden, heads, hidden * 2, batch_first=True, dropout=0.1)
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        trend = F.avg_pool1d(x.transpose(1, 2), kernel_size=9, stride=1, padding=4).transpose(1, 2)
        seasonal = x - trend
        z = self.encoder(self.season_proj(seasonal))
        return self.head(z.mean(dim=1)).squeeze(-1) + self.trend_head(trend[:, -1]).squeeze(-1)


class ITransformerLike(nn.Module):
    def __init__(self, seq_len: int, n_features: int, hidden: int = 96, heads: int = 4):
        super().__init__()
        self.var_proj = nn.Linear(seq_len, hidden)
        layer = nn.TransformerEncoderLayer(hidden, heads, hidden * 2, batch_first=True, dropout=0.1)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x.transpose(1, 2)
        z = self.encoder(self.var_proj(z))
        return self.head(z.mean(dim=1)).squeeze(-1)


class ModernTCNLike(nn.Module):
    def __init__(self, n_features: int, hidden: int = 96):
        super().__init__()
        self.in_proj = nn.Conv1d(n_features, hidden, kernel_size=1)
        self.depthwise = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=7, padding=3, groups=hidden),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=7, padding=3, groups=hidden),
            nn.GELU(),
        )
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.in_proj(x.transpose(1, 2))
        z = self.depthwise(z)
        return self.head(z[:, :, -1]).squeeze(-1)


class MambaLike(nn.Module):
    def __init__(self, n_features: int, hidden: int = 96):
        super().__init__()
        self.proj = nn.Linear(n_features, hidden * 2)
        self.conv = nn.Conv1d(hidden, hidden, kernel_size=5, padding=4, groups=hidden)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.proj(x).chunk(2, dim=-1)
        z = value * torch.sigmoid(gate)
        z = self.conv(z.transpose(1, 2))[..., : x.shape[1]]
        return self.head(z[:, :, -1]).squeeze(-1)


class TimesNetLike(nn.Module):
    def __init__(self, n_features: int, hidden: int = 96):
        super().__init__()
        self.conv3 = nn.Conv1d(n_features, hidden, kernel_size=3, padding=1)
        self.conv7 = nn.Conv1d(n_features, hidden, kernel_size=7, padding=3)
        self.head = nn.Sequential(nn.GELU(), nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x.transpose(1, 2)
        a = self.conv3(z).mean(dim=-1)
        b = self.conv7(z).mean(dim=-1)
        return self.head(torch.cat([a, b], dim=-1)).squeeze(-1)


class TimeXerLike(nn.Module):
    def __init__(self, seq_len: int, n_features: int, hidden: int = 96, heads: int = 4):
        super().__init__()
        self.temporal = TransformerForecaster(seq_len, n_features, hidden, heads)
        self.exog_gate = nn.Sequential(nn.Linear(n_features, hidden), nn.GELU(), nn.Linear(hidden, 1), nn.Tanh())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.temporal(x)
        gate = self.exog_gate(x[:, -1]).squeeze(-1)
        return base * (1.0 + 0.1 * gate)


def make_model(name: str, seq_len: int, n_features: int, hidden: int) -> nn.Module:
    factories: dict[str, Callable[[], nn.Module]] = {
        "Linear": lambda: LinearForecaster(seq_len, n_features, hidden),
        "LSTM": lambda: RecurrentForecaster(n_features, hidden, "lstm"),
        "GRU": lambda: RecurrentForecaster(n_features, hidden, "gru"),
        "TCN": lambda: TCNForecaster(n_features, hidden),
        "Transformer": lambda: TransformerForecaster(seq_len, n_features, hidden),
        "DLinearLike": lambda: DLinearLike(seq_len, n_features),
        "NLinearLike": lambda: NLinearLike(seq_len, n_features),
        "PatchTSTLike": lambda: PatchTSTLike(seq_len, n_features, hidden),
        "AutoformerLike": lambda: AutoformerLike(seq_len, n_features, hidden),
        "ITransformerLike": lambda: ITransformerLike(seq_len, n_features, hidden),
        "ModernTCNLike": lambda: ModernTCNLike(n_features, hidden),
        "MambaLike": lambda: MambaLike(n_features, hidden),
        "TimesNetLike": lambda: TimesNetLike(n_features, hidden),
        "TimeXerLike": lambda: TimeXerLike(seq_len, n_features, hidden),
    }
    if name not in factories:
        raise KeyError(f"알 수 없는 모델명: {name}")
    return factories[name]()


BASE_MODELS = ["Linear", "LSTM", "GRU", "TCN", "Transformer"]
EXPANDED_MODELS = [
    "AutoformerLike",
    "PatchTSTLike",
    "DLinearLike",
    "NLinearLike",
    "TimesNetLike",
    "TimeXerLike",
    "ITransformerLike",
    "ModernTCNLike",
    "MambaLike",
]
ALL_SINGLE_MODELS = BASE_MODELS + EXPANDED_MODELS
ENSEMBLES = {
    "LSTM_Autoformer_mean": ["LSTM", "AutoformerLike"],
    "TCN_Transformer_mean": ["TCN", "Transformer"],
    "Linear_sequence_residual": ["Linear", "LSTM", "TCN"],
    "validation_weighted_topk": ALL_SINGLE_MODELS,
}


# ---------------------------------------------------------------------------
# 4. 학습, 평가, collapse 진단
# ---------------------------------------------------------------------------


def make_loader(split: dict[str, np.ndarray], batch_size: int, shuffle: bool, profile: ResourceProfile) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(split["x"]), torch.from_numpy(split["y"]))
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=profile.num_workers,
        pin_memory=profile.pin_memory,
        drop_last=False,
    )


def make_optimizer(model: nn.Module, optimizer_name: str, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    if optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if optimizer_name == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    if optimizer_name == "sgd_momentum":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=weight_decay)
    raise ValueError(f"지원하지 않는 optimizer: {optimizer_name}")


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str,
    epochs: int,
    steps_per_epoch: int,
    lr: float,
) -> torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau | None:
    if scheduler_name == "none":
        return None
    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    if scheduler_name == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            epochs=max(1, epochs),
            steps_per_epoch=max(1, steps_per_epoch),
        )
    if scheduler_name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    raise ValueError(f"지원하지 않는 scheduler: {scheduler_name}")


def grad_norm(model: nn.Module) -> float:
    total = 0.0
    for param in model.parameters():
        if param.grad is None:
            continue
        value = param.grad.detach().data.norm(2).item()
        total += value * value
    return float(math.sqrt(total))


def apply_gradient_policy(model: nn.Module, policy: str, epoch: int) -> float:
    norm_before = grad_norm(model)
    if policy == "none":
        return norm_before
    if policy == "clip0.5":
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    elif policy == "clip1":
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    elif policy == "clip5":
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    elif policy == "adaptive":
        threshold = min(5.0, max(0.5, 0.5 + 0.1 * epoch))
        nn.utils.clip_grad_norm_(model.parameters(), threshold)
    else:
        raise ValueError(f"지원하지 않는 gradient policy: {policy}")
    return norm_before


def objective_loss(pred: torch.Tensor, target: torch.Tensor, loss_name: str) -> torch.Tensor:
    if loss_name == "return_mse":
        return F.mse_loss(pred, target)
    if loss_name == "return_mae":
        return F.l1_loss(pred, target)
    if loss_name == "return_huber":
        return F.smooth_l1_loss(pred, target, beta=0.001)
    if loss_name == "return_huber_directional":
        huber = F.smooth_l1_loss(pred, target, beta=0.001)
        direction = torch.sign(target).clamp(min=-1, max=1)
        directional = F.softplus(-pred * direction * 100.0).mean()
        return huber + 0.05 * directional
    if loss_name == "vol_weighted_huber":
        weight = 1.0 + torch.clamp(torch.abs(target) / (target.detach().std() + 1e-6), 0, 5)
        return (weight * F.smooth_l1_loss(pred, target, beta=0.001, reduction="none")).mean()
    if loss_name == "anti_collapse_huber":
        huber = F.smooth_l1_loss(pred, target, beta=0.001)
        target_std = target.detach().std() + 1e-6
        pred_std = pred.std() + 1e-6
        variance_floor = F.relu(0.25 * target_std - pred_std) / target_std
        zero_pull = torch.exp(-torch.abs(pred) / 0.0005).mean()
        direction = torch.sign(target).clamp(min=-1, max=1)
        directional = F.softplus(-pred * direction * 100.0).mean()
        return huber + 0.05 * directional + 0.02 * variance_floor + 0.01 * zero_pull
    if loss_name == "quantile_pinball":
        tau = 0.5
        error = target - pred
        return torch.maximum(tau * error, (tau - 1.0) * error).mean()
    raise ValueError(f"지원하지 않는 loss: {loss_name}")


def train_one_model(
    model: nn.Module,
    splits: dict[str, dict[str, np.ndarray]],
    profile: ResourceProfile,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    loss_name: str,
    optimizer_name: str,
    scheduler_name: str,
    gradient_policy: str,
) -> tuple[nn.Module, pd.DataFrame]:
    device = torch.device(profile.device)
    model = model.to(device)
    train_loader = make_loader(splits["train"], batch_size, True, profile)
    val_loader = make_loader(splits["val"], batch_size, False, profile)
    optimizer = make_optimizer(model, optimizer_name, lr, weight_decay)
    scheduler = make_scheduler(optimizer, scheduler_name, epochs, len(train_loader), lr)
    rows = []
    best_state = None
    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        grad_norms = []
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = objective_loss(pred, yb, loss_name)
            loss.backward()
            grad_norms.append(apply_gradient_policy(model, gradient_policy, epoch))
            optimizer.step()
            if scheduler_name == "onecycle" and scheduler is not None:
                scheduler.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                pred = model(xb)
                loss = objective_loss(pred, yb, loss_name)
                val_losses.append(float(loss.detach().cpu()))
        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        if scheduler is not None and scheduler_name == "plateau":
            scheduler.step(val_loss)
        elif scheduler is not None and scheduler_name not in {"onecycle", "plateau"}:
            scheduler.step()
        lr_now = float(optimizer.param_groups[0]["lr"])
        rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "grad_norm_mean": float(np.mean(grad_norms)) if grad_norms else float("nan"),
                "grad_norm_max": float(np.max(grad_norms)) if grad_norms else float("nan"),
                "lr": lr_now,
            }
        )
        if val_loss < best_val:
            best_val = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(
            f"epoch={epoch:03d} train={train_loss:.6f} val={val_loss:.6f} "
            f"grad={rows[-1]['grad_norm_mean']:.4f} lr={lr_now:.2e}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, pd.DataFrame(rows)


def predict(model: nn.Module, split: dict[str, np.ndarray], profile: ResourceProfile, batch_size: int) -> np.ndarray:
    device = torch.device(profile.device)
    loader = make_loader(split, batch_size, False, profile)
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


# ---------------------------------------------------------------------------
# 5. 시각화와 보고서
# ---------------------------------------------------------------------------


def save_learning_curve(curves: pd.DataFrame, out_path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=160)
    ax.plot(curves["epoch"], curves["train_loss"], label="train loss", linewidth=2)
    ax.plot(curves["epoch"], curves["val_loss"], label="validation loss", linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.grid(alpha=0.25)
    if "grad_norm_mean" in curves.columns:
        ax2 = ax.twinx()
        ax2.plot(curves["epoch"], curves["grad_norm_mean"], label="grad norm mean", color="tab:green", alpha=0.65)
        ax2.set_ylabel("gradient norm")
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="best")
    else:
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_prediction_plot(split: dict[str, np.ndarray], pred_return: np.ndarray, out_path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = min(300, len(pred_return))
    idx = np.arange(n)
    actual_return = split["y"][-n:]
    pred_return = pred_return[-n:]
    prev_close = split["prev_close"][-n:]
    target_close = split["target_close"][-n:]
    pred_close = prev_close * np.exp(pred_return)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), dpi=160, sharex=True)
    axes[0].plot(idx, actual_return, label="actual next return", linewidth=1.5)
    axes[0].plot(idx, pred_return, label="predicted next return", linewidth=1.5)
    axes[0].axhline(0, color="black", linewidth=0.8, alpha=0.5)
    axes[0].set_ylabel("log return")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(idx, target_close, label="actual next close", linewidth=1.5)
    axes[1].plot(idx, pred_close, label="predicted next close", linewidth=1.5)
    axes[1].plot(idx, prev_close, label="persistence baseline", linewidth=1.0, alpha=0.7)
    axes[1].set_ylabel("KRW")
    axes[1].set_xlabel("latest test samples")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_collapse_bar(summary: pd.DataFrame, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if summary.empty:
        return
    cols = ["variance_ratio", "near_zero_return_share", "copy_risk_ratio"]
    data = summary.set_index("case_id")[cols].tail(25)
    fig, ax = plt.subplots(figsize=(max(10, len(data) * 0.45), 5), dpi=160)
    data.plot(kind="bar", ax=ax)
    ax.axhline(1.0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_title("Collapse diagnostics by case")
    ax.set_ylabel("diagnostic value")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def write_report(
    report_path: Path,
    env: dict[str, object],
    stats: dict[str, object],
    args: argparse.Namespace,
    summary: pd.DataFrame,
    image_paths: list[Path],
) -> None:
    lines = [
        "# 8번 최적화 breadth training 결과 보고서",
        "",
        "## 의도",
        "",
        "8번은 7번의 계획 출력에서 빠졌던 실제 학습 backend를 구현한 실험이다. 6번에서 안정화한 target, normalization, loss, model selection 기준을 유지하면서 모델군과 테스트 케이스를 넓혀, shortcut collapse와 직전가 복사 문제가 다시 발생하는지 확인한다.",
        "",
        "## 방법론 설명",
        "",
        "- 예측 목표는 다음 시점의 가격 자체가 아니라 다음 로그수익률이다. 가격 레벨을 직접 맞추면 직전 가격을 복사해도 loss가 작아지는 쉬운 해가 생길 수 있기 때문이다.",
        "- 평가는 예측 로그수익률을 `prev_close * exp(pred_return)`로 KRW 가격에 복원한 뒤 MAE/RMSE를 계산한다.",
        "- 전처리는 `winsorize`, `asinh`, `diff`, `ema_residual`, `frequency_highpass`를 포함한다. 이는 극단값, heavy-tail, 추세, 저주파 성분이 학습을 쉬운 해로 끌고 가는지 확인하기 위한 축이다.",
        "- 정규화는 `window_standard`, `window_robust`, `window_minmax`, `revin`, `robust_revin`, `asinh_revin`을 포함한다. RevIN 계열 아이디어처럼 window 또는 instance 단위 통계로 비정상 분포 이동을 완화하는지 확인한다.",
        "- 손실함수는 MSE/MAE/Huber뿐 아니라 방향성 벌점, 변동성 가중 Huber, anti-collapse Huber, pinball loss를 비교한다.",
        "- optimizer/scheduler는 AdamW, Adam, RMSProp, SGD+Nesterov와 cosine, onecycle, plateau를 비교한다. 이는 같은 모델이라도 경사 하강 궤적이 달라지면 쉬운 해로 가는 정도가 바뀔 수 있기 때문이다.",
        "- gradient policy는 no clipping, 고정 clipping, adaptive clipping을 비교한다. 학습 곡선에는 loss와 gradient norm을 함께 저장한다.",
        "- collapse 진단은 예측 수익률 분산이 실제보다 지나치게 작은지, 예측이 0 근처에 몰리는지, 단순 persistence baseline과 거의 같은지 함께 본다.",
        "",
        "## 실행 환경",
        "",
        "```json",
        json.dumps(env, ensure_ascii=False, indent=2),
        "```",
        "",
        "## 기초 통계량",
        "",
        "```json",
        json.dumps(stats, ensure_ascii=False, indent=2),
        "```",
        "",
        "## 실행 설정",
        "",
        "```json",
        json.dumps(vars(args), ensure_ascii=False, indent=2, default=str),
        "```",
        "",
        "## 결과 요약",
        "",
    ]
    if summary.empty:
        lines.append("아직 완료된 case가 없다.")
    else:
        top = summary.sort_values(["collapse_score", "mae_krw"]).head(20)
        lines.extend(["```text", top.to_string(index=False), "```"])
    lines.extend(["", "## 그림 해석 기준", ""])
    lines.extend(
        [
            "- 좋은 학습 곡선은 train loss와 validation loss가 함께 완만히 내려가고, validation loss가 초반부터 평평하게 멈추지 않아야 한다.",
            "- 좋은 예측 그림은 예측 수익률이 0에만 붙어 있지 않고, 실제 수익률의 방향 변화 일부를 따라가야 한다.",
            "- 좋은 KRW 복원 그림은 단순 persistence 선과 완전히 겹치지 않으면서도 실제 가격을 무리하게 과대 진동시키지 않아야 한다.",
            "- 나쁜 그림은 loss만 줄지만 예측 수익률 분산이 거의 0이거나, KRW 예측이 직전가 baseline과 사실상 겹치는 경우다.",
        ]
    )
    lines.extend(["", "## 저장된 시각화", ""])
    for path in image_paths:
        rel = path.relative_to(REPO_ROOT).as_posix()
        lines.append(f"- `{rel}`")
    lines.extend(
        [
            "",
            "## 다음 판단",
            "",
            "이 보고서는 단일 수치 1등 모델을 고르는 것이 아니라, 어떤 모델군/정규화/loss 조합이 쉬운 해로 붕괴하는지 먼저 거르는 데 사용한다. collapse 진단을 통과한 조합만 후속 독립변수, 데이터마트, 텍스트 컨텍스트 결합 연구로 넘긴다.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# 6. 실험 실행
# ---------------------------------------------------------------------------


def model_list_for_suite(suite: str, model_arg: str) -> list[str]:
    if model_arg and model_arg != "auto":
        if model_arg == "all":
            return ALL_SINGLE_MODELS
        return [item.strip() for item in model_arg.split(",") if item.strip()]
    if suite == "breadth_probe":
        return ALL_SINGLE_MODELS
    if suite == "core_tuning_probe":
        return BASE_MODELS
    if suite == "scale_confirmation":
        return BASE_MODELS + ["AutoformerLike", "PatchTSTLike", "DLinearLike"]
    if suite in {"preprocessing_cross_check", "normalization_cross_check", "loss_cross_check", "optimization_cross_check", "gradient_cross_check"}:
        return ["Linear", "LSTM", "TCN", "Transformer", "AutoformerLike", "PatchTSTLike"]
    if suite == "ensemble_probe":
        return sorted(set(sum(ENSEMBLES.values(), [])))
    if suite == "full":
        return ALL_SINGLE_MODELS
    raise ValueError(f"알 수 없는 suite: {suite}")


def case_grid(args: argparse.Namespace) -> list[dict[str, str]]:
    feature_sets = args.feature_sets.split(",")
    preprocessings = args.preprocessings.split(",")
    normalizations = args.normalizations.split(",")
    losses = args.losses.split(",")
    optimizers = args.optimizers.split(",")
    schedulers = args.schedulers.split(",")
    gradient_policies = args.gradient_policies.split(",")
    models = model_list_for_suite(args.suite, args.models)

    if args.suite == "preprocessing_cross_check":
        feature_sets = [feature_sets[0]]
        normalizations = [normalizations[0]]
        losses = [losses[0]]
        optimizers = [optimizers[0]]
        schedulers = [schedulers[0]]
        gradient_policies = [gradient_policies[0]]
    elif args.suite == "normalization_cross_check":
        feature_sets = [feature_sets[0]]
        preprocessings = [preprocessings[0]]
        losses = [losses[0]]
        optimizers = [optimizers[0]]
        schedulers = [schedulers[0]]
        gradient_policies = [gradient_policies[0]]
    elif args.suite == "loss_cross_check":
        feature_sets = [feature_sets[0]]
        preprocessings = [preprocessings[0]]
        normalizations = [normalizations[0]]
        optimizers = [optimizers[0]]
        schedulers = [schedulers[0]]
        gradient_policies = [gradient_policies[0]]
    elif args.suite == "optimization_cross_check":
        feature_sets = [feature_sets[0]]
        preprocessings = [preprocessings[0]]
        normalizations = [normalizations[0]]
        losses = [losses[0]]
        gradient_policies = [gradient_policies[0]]
    elif args.suite == "gradient_cross_check":
        feature_sets = [feature_sets[0]]
        preprocessings = [preprocessings[0]]
        normalizations = [normalizations[0]]
        losses = [losses[0]]
        optimizers = [optimizers[0]]
        schedulers = [schedulers[0]]
    elif args.suite in {"breadth_probe", "scale_confirmation", "ensemble_probe"}:
        feature_sets = [feature_sets[0]]
        preprocessings = [preprocessings[0]]
        normalizations = [normalizations[0]]
        losses = [losses[0]]
        optimizers = [optimizers[0]]
        schedulers = [schedulers[0]]
        gradient_policies = [gradient_policies[0]]
    elif args.suite == "core_tuning_probe":
        feature_sets = [feature_sets[0]]
        # 5개 기본 모델을 같은 조건에서 넓게 흔들되, 폭발 방지를 위해 축별 후보 수를 제한한다.
        preprocessings = preprocessings[:2]
        normalizations = normalizations[:2]
        losses = losses[:2]
        optimizers = optimizers[:1]
        schedulers = schedulers[:2]
        gradient_policies = gradient_policies[:2]

    cases = []
    for feature_set in feature_sets:
        for preprocessing in preprocessings:
            for normalization in normalizations:
                for loss in losses:
                    for optimizer in optimizers:
                        for scheduler in schedulers:
                            for gradient_policy in gradient_policies:
                                for model in models:
                                    cases.append(
                                        {
                                            "feature_set": feature_set.strip(),
                                            "preprocessing": preprocessing.strip(),
                                            "normalization": normalization.strip(),
                                            "loss": loss.strip(),
                                            "optimizer": optimizer.strip(),
                                            "scheduler": scheduler.strip(),
                                            "gradient_policy": gradient_policy.strip(),
                                            "model": model.strip(),
                                        }
                                    )
    return cases


def limit_cases(cases: list[dict[str, str]], max_cases: int) -> list[dict[str, str]]:
    if max_cases <= 0 or len(cases) <= max_cases:
        return cases
    print(f"[case-limit] {len(cases)}개 case 중 앞 {max_cases}개만 실행한다. 전체 실행은 --max-cases 0 사용.")
    return cases[:max_cases]


def run_case(
    case: dict[str, str],
    base_df: pd.DataFrame,
    args: argparse.Namespace,
    profile: ResourceProfile,
    timestamp: str,
) -> tuple[dict[str, object], pd.DataFrame, np.ndarray]:
    feature_columns = FEATURE_SETS[case["feature_set"]]
    data = build_windows(
        base_df,
        feature_columns,
        seq_len=args.seq_len,
        preprocessing=case["preprocessing"],
        normalization=case["normalization"],
        max_windows=args.max_windows,
        stride=args.stride,
    )
    splits = time_split(data, args.train_ratio, args.val_ratio)
    batch_size = args.batch_size or profile.batch_size or 32
    while True:
        try:
            model = make_model(case["model"], args.seq_len, len(feature_columns), args.hidden)
            model, curves = train_one_model(
                model,
                splits,
                profile,
                epochs=args.epochs,
                batch_size=batch_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
                loss_name=case["loss"],
                optimizer_name=case["optimizer"],
                scheduler_name=case["scheduler"],
                gradient_policy=case["gradient_policy"],
            )
            pred = predict(model, splits["test"], profile, batch_size)
            break
        except RuntimeError as exc:
            is_oom = "out of memory" in str(exc).lower()
            if not is_oom or batch_size <= 4:
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            next_batch = max(4, batch_size // 2)
            print(f"[oom-retry] batch_size {batch_size} -> {next_batch}")
            batch_size = next_batch
            gc.collect()
    metrics = evaluate_predictions(splits["test"], pred)
    case_id = (
        f"{case['model']}__{case['feature_set']}__{case['preprocessing']}__{case['normalization']}__"
        f"{case['loss']}__{case['optimizer']}__{case['scheduler']}__{case['gradient_policy']}"
    )
    result = {
        "case_id": case_id,
        "suite": args.suite,
        "model": case["model"],
        "feature_set": case["feature_set"],
        "preprocessing": case["preprocessing"],
        "normalization": case["normalization"],
        "loss": case["loss"],
        "optimizer": case["optimizer"],
        "scheduler": case["scheduler"],
        "gradient_policy": case["gradient_policy"],
        "seq_len": args.seq_len,
        "batch_size": batch_size,
        "train_windows": len(splits["train"]["y"]),
        "val_windows": len(splits["val"]["y"]),
        "test_windows": len(splits["test"]["y"]),
        **metrics,
    }

    digest = hashlib.sha1(case_id.encode("utf-8")).hexdigest()[:10]
    safe_id = f"{case['model']}__{digest}"
    curve_path = RESULTS_DIR / f"{ARTIFACT_PREFIX}_{timestamp}_{safe_id}_curves.csv"
    curves.assign(case_id=case_id).to_csv(curve_path, index=False, encoding="utf-8-sig")
    save_learning_curve(
        curves,
        IMAGES_DIR / f"{ARTIFACT_PREFIX}_{timestamp}_{safe_id}_learning_curve.png",
        f"{case_id} learning curve",
    )
    save_prediction_plot(
        splits["test"],
        pred,
        IMAGES_DIR / f"{ARTIFACT_PREFIX}_{timestamp}_{safe_id}_prediction.png",
        f"{case_id} prediction diagnostics",
    )
    return result, curves.assign(case_id=case_id), pred


def run_ensembles(
    single_summary: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    test_split: dict[str, np.ndarray],
) -> list[dict[str, object]]:
    rows = []
    if single_summary.empty:
        return rows
    for name, members in ENSEMBLES.items():
        available = [member for member in members if member in predictions]
        if len(available) < 2:
            continue
        if name == "validation_weighted_topk":
            ranked = single_summary[single_summary["model"].isin(available)].sort_values("mae_krw").head(5)
            available = ranked["model"].tolist()
            weights = 1.0 / (ranked["mae_krw"].to_numpy() + 1e-9)
            weights = weights / weights.sum()
        else:
            weights = np.ones(len(available)) / len(available)
        stacked = np.stack([predictions[member] for member in available])
        pred = np.average(stacked, axis=0, weights=weights)
        metrics = evaluate_predictions(test_split, pred)
        rows.append(
            {
                "case_id": name,
                "suite": "ensemble_probe",
                "model": name,
                "feature_set": "ensemble",
                "normalization": "member_defined",
                "loss": "member_defined",
                "seq_len": np.nan,
                "train_windows": np.nan,
                "val_windows": np.nan,
                "test_windows": len(test_split["y"]),
                **metrics,
            }
        )
    return rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="8번 최적화 breadth training 실험")
    parser.add_argument("--db", default=None)
    parser.add_argument("--table", default="btc_15m_advance")
    parser.add_argument("--ticker", default=None)
    parser.add_argument("--profile", default="school_4090_15gb")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"])
    parser.add_argument(
        "--suite",
        default="breadth_probe",
        choices=[
            "breadth_probe",
            "core_tuning_probe",
            "preprocessing_cross_check",
            "normalization_cross_check",
            "loss_cross_check",
            "optimization_cross_check",
            "gradient_cross_check",
            "ensemble_probe",
            "scale_confirmation",
            "full",
        ],
    )
    parser.add_argument("--models", default="auto")
    parser.add_argument("--feature-sets", default="wide_stationary,returns_core,technical_liquidity")
    parser.add_argument("--preprocessings", default="none,winsorize,asinh,diff,ema_residual,frequency_highpass")
    parser.add_argument("--normalizations", default="window_standard,window_robust,asinh_revin,revin,global_standard,window_minmax")
    parser.add_argument(
        "--losses",
        default="return_huber,return_huber_directional,anti_collapse_huber,vol_weighted_huber,quantile_pinball,return_mse",
    )
    parser.add_argument("--optimizers", default="adamw,adam,rmsprop,sgd_momentum")
    parser.add_argument("--schedulers", default="cosine,none,onecycle,plateau")
    parser.add_argument("--gradient-policies", default="clip1,adaptive,clip0.5,clip5,none")
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--max-rows", type=int, default=40000)
    parser.add_argument("--max-windows", type=int, default=4096)
    parser.add_argument("--max-cases", type=int, default=120)
    parser.add_argument("--print-cases", type=int, default=30)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-failure", action="store_true")
    return parser.parse_known_args(argv)[0]


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    profile = build_resource_profile(args.profile, args.device)
    apply_resource_profile(profile)
    env = log_environment(profile)

    cases = limit_cases(case_grid(args), args.max_cases)
    print(f"[plan] suite={args.suite} cases={len(cases)}")
    for case in cases[: max(0, args.print_cases)]:
        print("[plan]", case)
    if len(cases) > args.print_cases:
        print(f"[plan] ... {len(cases) - args.print_cases}개 case 출력 생략")

    if args.dry_run:
        print("[dry-run] 데이터 로드와 학습을 수행하지 않고 종료한다.")
        return

    db_path = resolve_db_path(args.db)
    raw = load_price_data(db_path, args.table, args.ticker, args.max_rows)
    features = make_features(raw)
    stats = basic_statistics(features)
    print("[data]", json.dumps(stats, ensure_ascii=False, indent=2))

    summary_rows: list[dict[str, object]] = []
    curve_frames: list[pd.DataFrame] = []
    predictions_by_model: dict[str, np.ndarray] = {}
    latest_test_split: dict[str, np.ndarray] | None = None
    image_paths: list[Path] = []

    for index, case in enumerate(cases, start=1):
        print(f"\n[case {index}/{len(cases)}] {case}")
        try:
            result, curves, pred = run_case(case, features, args, profile, timestamp)
            summary_rows.append(result)
            curve_frames.append(curves)
            predictions_by_model[case["model"]] = pred

            feature_columns = FEATURE_SETS[case["feature_set"]]
            data = build_windows(
                features,
                feature_columns,
                args.seq_len,
                case["preprocessing"],
                case["normalization"],
                args.max_windows,
                args.stride,
            )
            latest_test_split = time_split(data, args.train_ratio, args.val_ratio)["test"]

            digest = hashlib.sha1(str(result["case_id"]).encode("utf-8")).hexdigest()[:10]
            safe_id = f"{result['model']}__{digest}"
            image_paths.extend(
                [
                    IMAGES_DIR / f"{ARTIFACT_PREFIX}_{timestamp}_{safe_id}_learning_curve.png",
                    IMAGES_DIR / f"{ARTIFACT_PREFIX}_{timestamp}_{safe_id}_prediction.png",
                ]
            )
            pd.DataFrame(summary_rows).to_csv(
                RESULTS_DIR / f"{ARTIFACT_PREFIX}_{timestamp}_summary.csv",
                index=False,
                encoding="utf-8-sig",
            )
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and torch.cuda.is_available():
                torch.cuda.empty_cache()
            if not args.continue_on_failure:
                raise
            print(f"[case failed] {case}: {exc}")
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    summary = pd.DataFrame(summary_rows)
    if args.suite in {"ensemble_probe", "full"} and latest_test_split is not None:
        ensemble_rows = run_ensembles(summary, predictions_by_model, latest_test_split)
        if ensemble_rows:
            summary = pd.concat([summary, pd.DataFrame(ensemble_rows)], ignore_index=True)

    summary_path = RESULTS_DIR / f"{ARTIFACT_PREFIX}_{timestamp}_summary.csv"
    curves_path = RESULTS_DIR / f"{ARTIFACT_PREFIX}_{timestamp}_curves.csv"
    collapse_path = IMAGES_DIR / f"{ARTIFACT_PREFIX}_{timestamp}_collapse_diagnostics.png"
    report_path = RESULTS_DIR / f"{ARTIFACT_PREFIX}_{timestamp}_report.md"

    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    if curve_frames:
        pd.concat(curve_frames, ignore_index=True).to_csv(curves_path, index=False, encoding="utf-8-sig")
    save_collapse_bar(summary, collapse_path)
    image_paths.append(collapse_path)
    write_report(report_path, env, stats, args, summary, image_paths)

    print(f"[done] summary: {summary_path}")
    print(f"[done] curves: {curves_path}")
    print(f"[done] report: {report_path}")


if __name__ == "__main__":
    main()
