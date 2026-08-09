# [FOR COMMIT TRACKING ONLY - DO NOT EXECUTE]
# This file is automatically mirrored from the corresponding .ipynb for git diff purposes.
# Actual research execution should be performed in the Jupyter Notebook (.ipynb)
# or in an approved remote/server environment.

# %% [markdown]
# # 10번 objective·ensemble 본실험
#
# 9번에서 선별한 전처리 후보를 고정한 뒤, 손실함수 구성 요소와 seed/validation ensemble이 persistence 미달, 0수익률 평탄화, 출력 분산 폭주를 줄이는지 확인합니다.
#
# 모든 표와 그림은 notebook inline output으로만 표시하며 서버에 PNG/CSV/Markdown 파일을 저장하지 않습니다.

# %%
"""10번: objective 구성과 validation-only ensemble을 확인하는 본실험."""

from __future__ import annotations

import argparse
import copy
import gc
import importlib.util
import json
import math
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


def find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents, Path.home() / "personal_ai_project" / "quantitative_trading"]:
        if (candidate / "pyproject.toml").exists() and (candidate / "test").exists():
            return candidate
    return start


REPO_ROOT = find_repo_root(Path.cwd())
DIAGNOSTIC_PATH = REPO_ROOT / "test" / "models" / "9_preprocessing_uncertainty_diagnostics_test.py"


def load_diagnostic_backend():
    spec = importlib.util.spec_from_file_location("preprocessing_uncertainty_backend", DIAGNOSTIC_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"9번 backend를 불러오지 못했다: {DIAGNOSTIC_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


diag = load_diagnostic_backend()
base = diag.base

DEFAULT_MODELS = ["Linear", "PatchTSTLike"]
FAILURE_CONTROL_MODELS = ["TimesNetLike", "AutoformerLike"]
DEFAULT_PREPROCESSINGS = [
    "seasonal_diff16",
    "frequency_bandpass",
    "median_residual_5",
    "linear_detrend+asinh_robust",
    "winsor_025",
    "none",
]
FAILURE_CONTROL_PREPROCESSINGS = ["none", "seasonal_diff16", "winsor_025"]
DEFAULT_OBJECTIVES = [
    "huber",
    "directional_huber",
    "variance_huber",
    "correlation_huber",
    "tail_huber",
    "regime_huber",
    "anti_collapse_v2",
    "balanced_composite",
]
DEFAULT_SEEDS = [42, 137, 2026]

PARALLEL_RESOURCE_SLOTS = {
    "exclusive": {
        "device": None,
        "gpu_memory_fraction": None,
        "n_jobs": 16,
        "max_workers": 16,
        "num_workers": 4,
        "torch_num_threads": 16,
        "torch_interop_threads": 4,
        "batch_size": 48,
    },
    "point_primary": {
        "device": None,
        "gpu_memory_fraction": 0.52,
        "n_jobs": 8,
        "max_workers": 8,
        "num_workers": 2,
        "torch_num_threads": 8,
        "torch_interop_threads": 2,
        "batch_size": 32,
    },
    "risk_secondary": {
        "device": None,
        "gpu_memory_fraction": 0.36,
        "n_jobs": 6,
        "max_workers": 6,
        "num_workers": 2,
        "torch_num_threads": 6,
        "torch_interop_threads": 2,
        "batch_size": 24,
    },
    "cpu_companion": {
        "device": "cpu",
        "gpu_memory_fraction": None,
        "n_jobs": 8,
        "max_workers": 8,
        "num_workers": 2,
        "torch_num_threads": 8,
        "torch_interop_threads": 2,
        "batch_size": 64,
    },
}


def apply_parallel_resource_slot(profile, slot_name: str):
    if slot_name not in PARALLEL_RESOURCE_SLOTS:
        raise ValueError(f"지원하지 않는 parallel slot: {slot_name}")
    settings = PARALLEL_RESOURCE_SLOTS[slot_name]
    device = settings["device"] or profile.device
    adjusted = replace(
        profile,
        name=f"{profile.name}:{slot_name}",
        device=device,
        n_jobs=min(int(settings["n_jobs"]), profile.cpu_logical),
        max_workers=min(int(settings["max_workers"]), profile.cpu_logical),
        num_workers=min(int(settings["num_workers"]), profile.cpu_logical),
        torch_num_threads=min(int(settings["torch_num_threads"]), profile.cpu_logical),
        torch_interop_threads=min(int(settings["torch_interop_threads"]), profile.cpu_logical),
        optuna_n_jobs=1 if device == "cuda" else min(4, profile.cpu_logical),
        batch_size=int(settings["batch_size"]),
        pin_memory=device == "cuda",
    )
    fraction = settings["gpu_memory_fraction"]
    if device == "cuda" and torch.cuda.is_available() and fraction is not None:
        try:
            torch.cuda.set_per_process_memory_fraction(float(fraction), torch.cuda.current_device())
        except RuntimeError as exc:
            print(f"[parallel-resource] GPU memory fraction 적용을 건너뜀: {exc}")
    details = {
        "parallel_slot": slot_name,
        "device": device,
        "gpu_memory_fraction": fraction if device == "cuda" else None,
        "batch_size": adjusted.batch_size,
        "num_workers": adjusted.num_workers,
        "torch_num_threads": adjusted.torch_num_threads,
        "torch_interop_threads": adjusted.torch_interop_threads,
    }
    print("[parallel-resource]", json.dumps(details, ensure_ascii=False, indent=2))
    return adjusted, details


def configure_inline_matplotlib() -> None:
    diag.configure_inline_matplotlib()


def show_figure(fig) -> None:
    diag.show_figure(fig)


def display_table(title: str, frame: pd.DataFrame) -> None:
    print(f"\n[{title}]")
    try:
        from IPython.display import display

        display(frame)
    except Exception:
        print(frame.to_string(index=False))


def estimate_timestamp_gap_count(raw: pd.DataFrame) -> int:
    if "timestamp" not in raw.columns or len(raw) < 3:
        return 0
    ts = pd.to_datetime(raw["timestamp"], errors="coerce").dropna().sort_values()
    if len(ts) < 3:
        return 0
    delta = ts.diff().dropna()
    if delta.empty:
        return 0
    expected = delta.mode().iloc[0]
    return int((delta > expected).sum())


def build_missingness_audit(raw: pd.DataFrame, features: pd.DataFrame, horizon_label: str) -> pd.DataFrame:
    normalized_raw = base.normalize_columns(raw.copy())
    raw_ohlc_cols = [col for col in ["open", "high", "low", "close", "volume"] if col in normalized_raw.columns]
    raw_ohlc_missing = int(normalized_raw[raw_ohlc_cols].isna().sum().sum()) if raw_ohlc_cols else 0
    rows_dropped = int(len(raw) - len(features))
    timestamp_gap_count = estimate_timestamp_gap_count(normalized_raw)
    return pd.DataFrame(
        [
            {
                "case": "원본 OHLC/거래량 결측",
                "where": "거래소 원천 데이터",
                "why": "수집 누락 또는 timestamp gap",
                "interpolate": "조건부",
                "current_policy": (
                    f"현재 raw OHLC missing={raw_ohlc_missing}, inferred gaps={timestamp_gap_count}. "
                    "실제 원천 gap가 있으면 먼저 재수집/재샘플링을 검토하고, 라벨 구간 보간은 기본 금지."
                ),
            },
            {
                "case": "rolling warm-up",
                "where": "log_return/std/EMA 파생변수 시작 구간",
                "why": "lookback 길이가 아직 안 찼기 때문",
                "interpolate": "아니오",
                "current_policy": f"`make_features()` 단계에서 dropna로 제거. 현재 제거 행 수={rows_dropped}.",
            },
            {
                "case": "one-step target tail",
                "where": "next-return / next-close 라벨",
                "why": "마지막 행에는 다음 봉이 없음",
                "interpolate": "아니오",
                "current_policy": f"{horizon_label} 라벨 생성 꼬리 구간은 학습 샘플에서 제외.",
            },
            {
                "case": "window preprocessing 수치 NaN/inf",
                "where": "diff / robust scale / frequency 변환",
                "why": "0에 가까운 scale 또는 수치 폭주",
                "interpolate": "아니오",
                "current_policy": "시간 보간 대신 clip, robust scale, nan_to_num 같은 안정화 처리 사용.",
            },
        ]
    )


def draw_candles(
    ax,
    x: np.ndarray,
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    width: float,
    up_color: str,
    down_color: str,
    alpha: float,
    linewidth: float = 0.9,
    fill: bool = True,
) -> None:
    from matplotlib.patches import Rectangle

    for xi, o, h, l, c in zip(x, open_, high, low, close):
        color = up_color if c >= o else down_color
        ax.vlines(xi, l, h, color=color, linewidth=linewidth, alpha=alpha)
        lower = min(o, c)
        height = max(abs(c - o), 1e-9)
        rect = Rectangle(
            (xi - width / 2.0, lower),
            width,
            height,
            facecolor=color if fill else "none",
            edgecolor=color,
            linewidth=linewidth,
            alpha=alpha,
        )
        ax.add_patch(rect)


def candle_proxy_arrays(split, pred_return: np.ndarray, n: int) -> dict[str, np.ndarray]:
    prev_close = np.asarray(split["prev_close"][-n:], dtype=np.float32)
    target_close = np.asarray(split["target_close"][-n:], dtype=np.float32)
    target_open = np.asarray(split.get("target_open", prev_close)[-n:], dtype=np.float32)
    target_high_source = np.asarray(split.get("target_high", target_close)[-n:], dtype=np.float32)
    target_low_source = np.asarray(split.get("target_low", target_close)[-n:], dtype=np.float32)
    target_high = np.maximum(target_high_source, np.maximum(target_open, target_close))
    target_low = np.minimum(target_low_source, np.minimum(target_open, target_close))
    predicted_close = prev_close * np.exp(np.asarray(pred_return[-n:], dtype=np.float32))
    predicted_open = prev_close
    predicted_high = np.maximum(predicted_open, predicted_close)
    predicted_low = np.minimum(predicted_open, predicted_close)
    return {
        "timestamp": np.asarray(split["timestamp"][-n:]),
        "actual_open": target_open,
        "actual_high": target_high,
        "actual_low": target_low,
        "actual_close": target_close,
        "pred_open": predicted_open,
        "pred_high": predicted_high,
        "pred_low": predicted_low,
        "pred_close": predicted_close,
        "persistence_close": prev_close,
    }


def show_candlestick_comparison(ax, split, pred_return: np.ndarray, title: str, n: int = 60) -> None:
    from matplotlib.lines import Line2D

    n = min(n, len(pred_return))
    frame = candle_proxy_arrays(split, pred_return, n)
    x = np.arange(n, dtype=float)
    draw_candles(
        ax,
        x - 0.18,
        frame["actual_open"],
        frame["actual_high"],
        frame["actual_low"],
        frame["actual_close"],
        0.32,
        up_color="tab:red",
        down_color="tab:blue",
        alpha=0.65,
    )
    draw_candles(
        ax,
        x + 0.18,
        frame["pred_open"],
        frame["pred_high"],
        frame["pred_low"],
        frame["pred_close"],
        0.28,
        up_color="tab:orange",
        down_color="tab:green",
        alpha=0.85,
        fill=False,
    )
    ax.plot(x, frame["persistence_close"], color="black", linewidth=0.9, alpha=0.55)
    ax.set_title(title)
    ax.set_ylabel("KRW")
    ax.legend(
        handles=[
            Line2D([0], [0], color="tab:red", linewidth=6, alpha=0.65, label="actual candle"),
            Line2D([0], [0], color="tab:orange", linewidth=2, label="predicted proxy candle"),
            Line2D([0], [0], color="black", linewidth=1, alpha=0.55, label="persistence close"),
        ],
        fontsize=8,
        loc="best",
    )


def prediction_preview_table(
    split,
    pred_return: np.ndarray,
    n_rows: int = 12,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
) -> pd.DataFrame:
    n_rows = min(n_rows, len(pred_return))
    frame = candle_proxy_arrays(split, pred_return, n_rows)
    actual_return = split["y"][-n_rows:]
    pred_slice = pred_return[-n_rows:]
    table = pd.DataFrame(
        {
            "timestamp": frame["timestamp"],
            "actual_open": np.round(frame["actual_open"], 0),
            "actual_high": np.round(frame["actual_high"], 0),
            "actual_low": np.round(frame["actual_low"], 0),
            "actual_close": np.round(frame["actual_close"], 0),
            "pred_open_proxy": np.round(frame["pred_open"], 0),
            "pred_close_proxy": np.round(frame["pred_close"], 0),
            "persistence_close": np.round(frame["persistence_close"], 0),
            "actual_return": np.round(actual_return, 6),
            "predicted_return": np.round(pred_slice, 6),
        }
    )
    if lower is not None and upper is not None:
        table["return_lower"] = np.round(lower[-n_rows:], 6)
        table["return_upper"] = np.round(upper[-n_rows:], 6)
    return table


def safe_std(values: torch.Tensor) -> torch.Tensor:
    return values.std(unbiased=False).clamp_min(1e-6)


def loss_components(pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    huber_each = F.smooth_l1_loss(pred, target, beta=0.001, reduction="none")
    huber = huber_each.mean()

    target_sign = torch.sign(target)
    directional = F.softplus(-pred * target_sign * 100.0).mean()

    target_std = safe_std(target.detach())
    pred_std = safe_std(pred)
    variance_ratio = pred_std / target_std
    variance = torch.abs(torch.log(variance_ratio.clamp(0.05, 20.0)))

    pred_centered = pred - pred.mean()
    target_centered = target - target.mean()
    correlation = (pred_centered * target_centered).mean() / (
        safe_std(pred_centered) * safe_std(target_centered)
    )
    correlation_penalty = 1.0 - correlation.clamp(-1.0, 1.0)

    scale = target.detach().abs().median().clamp_min(1e-5)
    tail_weight = 1.0 + 2.0 * torch.clamp(target.detach().abs() / (3.0 * scale), 0.0, 3.0)
    tail = (tail_weight * huber_each).mean()

    volatility = target.detach().abs()
    threshold = torch.quantile(volatility, 0.75)
    high_mask = volatility >= threshold
    low_mask = ~high_mask
    high_loss = huber_each[high_mask].mean() if high_mask.any() else huber
    low_loss = huber_each[low_mask].mean() if low_mask.any() else huber
    regime = 0.5 * (high_loss + low_loss)

    near_zero = torch.exp(-pred.abs() / 0.00025).mean()
    mean_bias = torch.abs(pred.mean() - target.mean()) / target_std
    return {
        "huber": huber,
        "direction": directional,
        "variance": variance,
        "correlation": correlation_penalty,
        "tail": tail,
        "regime": regime,
        "near_zero": near_zero,
        "mean_bias": mean_bias,
    }


OBJECTIVE_WEIGHTS = {
    "huber": {},
    "directional_huber": {"direction": 0.20},
    "variance_huber": {"variance": 0.12},
    "correlation_huber": {"correlation": 0.10},
    "tail_huber": {"tail": 0.30},
    "regime_huber": {"regime": 0.30},
    "anti_collapse_v2": {
        "direction": 0.10,
        "variance": 0.10,
        "near_zero": 0.05,
    },
    "balanced_composite": {
        "direction": 0.08,
        "variance": 0.08,
        "correlation": 0.06,
        "mean_bias": 0.03,
    },
}


def objective_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    objective_name: str,
    auxiliary_scale: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if objective_name not in OBJECTIVE_WEIGHTS:
        raise ValueError(f"지원하지 않는 objective: {objective_name}")
    components = loss_components(pred, target)
    huber = components["huber"]
    total = huber
    for name, weight in OBJECTIVE_WEIGHTS[objective_name].items():
        component = components[name]
        # Auxiliary terms are dimensionless or differently scaled. Match their
        # detached magnitude to Huber so no single term dominates by units alone.
        magnitude_match = huber.detach() / component.detach().abs().clamp_min(1e-6)
        total = total + auxiliary_scale * weight * magnitude_match * component
    return total, components


def make_loader(split, batch_size: int, shuffle: bool, profile):
    return base.make_loader(split, batch_size, shuffle, profile)


def train_model(
    model,
    splits,
    profile,
    args: argparse.Namespace,
    objective_name: str,
    batch_size: int,
) -> tuple[torch.nn.Module, pd.DataFrame]:
    device = torch.device(profile.device)
    model = model.to(device)
    train_loader = make_loader(splits["train"], batch_size, True, profile)
    val_loader = make_loader(splits["val"], batch_size, False, profile)
    optimizer = base.make_optimizer(model, args.optimizer, args.lr, args.weight_decay)
    scheduler = base.make_scheduler(optimizer, args.scheduler, args.epochs, len(train_loader), args.lr)
    best_state = None
    best_val = float("inf")
    patience_left = args.patience
    rows: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        auxiliary_scale = min(1.0, max(0.0, (epoch - 1) / max(1, args.objective_warmup_epochs)))
        model.train()
        train_totals: list[float] = []
        train_gradients: list[float] = []
        component_sums: defaultdict[str, list[float]] = defaultdict(list)
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss, components = objective_loss(pred, yb, objective_name, auxiliary_scale)
            loss.backward()
            train_gradients.append(base.apply_gradient_policy(model, args.gradient_policy, epoch))
            optimizer.step()
            if args.scheduler == "onecycle" and scheduler is not None:
                scheduler.step()
            train_totals.append(float(loss.detach().cpu()))
            for name, value in components.items():
                component_sums[name].append(float(value.detach().cpu()))

        model.eval()
        val_totals: list[float] = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                loss, _ = objective_loss(model(xb), yb, objective_name, auxiliary_scale)
                val_totals.append(float(loss.detach().cpu()))

        train_loss = float(np.mean(train_totals))
        val_loss = float(np.mean(val_totals))
        if scheduler is not None and args.scheduler == "plateau":
            scheduler.step(val_loss)
        elif scheduler is not None and args.scheduler not in {"onecycle", "plateau"}:
            scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "grad_norm_mean": float(np.mean(train_gradients)),
            "grad_norm_max": float(np.max(train_gradients)),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "auxiliary_scale": auxiliary_scale,
        }
        for name, values in component_sums.items():
            row[f"component_{name}"] = float(np.mean(values))
        rows.append(row)
        print(
            f"epoch={epoch:03d} train={train_loss:.6f} val={val_loss:.6f} "
            f"grad={row['grad_norm_mean']:.4f} lr={row['lr']:.2e}"
        )

        if val_loss < best_val - args.min_delta:
            best_val = val_loss
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"[early-stop] epoch={epoch} best_val={best_val:.6f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, pd.DataFrame(rows)


def validation_selection_score(metrics: dict[str, float]) -> float:
    copy_term = math.log1p(max(0.0, metrics["copy_risk_ratio"]))
    variance_term = abs(math.log(max(1e-4, min(1e4, metrics["variance_ratio"]))))
    direction_term = max(0.0, 0.50 - metrics["direction_accuracy"]) * 4.0
    zero_term = metrics["near_zero_return_share"]
    return float(copy_term + 0.35 * variance_term + direction_term + zero_term)


def regime_metrics(split, prediction: np.ndarray) -> dict[str, float]:
    actual = split["y"]
    absolute_return = np.abs(actual)
    threshold = float(np.quantile(absolute_return, 0.75))
    high_mask = absolute_return >= threshold
    low_mask = ~high_mask
    error = np.abs(prediction - actual)
    return {
        "high_vol_return_mae": float(np.mean(error[high_mask])),
        "low_vol_return_mae": float(np.mean(error[low_mask])),
        "high_vol_direction_accuracy": float(
            np.mean((prediction[high_mask] > 0) == (actual[high_mask] > 0))
        ),
        "low_vol_direction_accuracy": float(
            np.mean((prediction[low_mask] > 0) == (actual[low_mask] > 0))
        ),
    }


def build_cases(args: argparse.Namespace) -> list[dict[str, object]]:
    models = [item.strip() for item in args.models.split(",") if item.strip()]
    preprocessings = [item.strip() for item in args.preprocessings.split(",") if item.strip()]
    objectives = [item.strip() for item in args.objectives.split(",") if item.strip()]
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    cases: list[dict[str, object]] = []

    if args.suite == "parallel_point_probe":
        point_preprocessings = [
            name for name in ["seasonal_diff16", "winsor_025"] if name in preprocessings
        ]
        point_objectives = [
            name for name in ["huber", "balanced_composite"] if name in objectives
        ]
        for preprocessing in point_preprocessings:
            for model in models:
                for objective in point_objectives:
                    for seed in seeds:
                        cases.append(
                            {
                                "model": model,
                                "preprocessing": preprocessing,
                                "objective": objective,
                                "seed": seed,
                            }
                        )
    elif args.suite == "objective_screen":
        for preprocessing in preprocessings:
            for model in models:
                for objective in objectives:
                    cases.append(
                        {
                            "model": model,
                            "preprocessing": preprocessing,
                            "objective": objective,
                            "seed": seeds[0],
                        }
                    )
    elif args.suite == "failure_mode_control":
        control_objectives = [
            name
            for name in ["huber", "variance_huber", "anti_collapse_v2", "balanced_composite"]
            if name in objectives
        ]
        for preprocessing in FAILURE_CONTROL_PREPROCESSINGS:
            for model in FAILURE_CONTROL_MODELS:
                for objective in control_objectives:
                    cases.append(
                        {
                            "model": model,
                            "preprocessing": preprocessing,
                            "objective": objective,
                            "seed": seeds[0],
                        }
                    )
    elif args.suite == "seed_confirmation":
        for preprocessing in preprocessings:
            for model in models:
                for objective in objectives[: min(4, len(objectives))]:
                    for seed in seeds:
                        cases.append(
                            {
                                "model": model,
                                "preprocessing": preprocessing,
                                "objective": objective,
                                "seed": seed,
                            }
                        )
    elif args.suite in {"ensemble_confirmation", "full"}:
        for preprocessing in preprocessings:
            for model in models:
                for objective in objectives:
                    for seed in seeds:
                        cases.append(
                            {
                                "model": model,
                                "preprocessing": preprocessing,
                                "objective": objective,
                                "seed": seed,
                            }
                        )
    else:
        raise ValueError(f"알 수 없는 suite: {args.suite}")
    return cases[: args.max_cases] if args.max_cases > 0 else cases


def run_case(case, features: pd.DataFrame, profile, args: argparse.Namespace):
    base.set_seed(int(case["seed"]))
    feature_columns = base.FEATURE_SETS[args.feature_set]
    base.apply_window_preprocessing = diag.apply_preprocessing_pipeline
    data = base.build_windows(
        features,
        feature_columns,
        args.seq_len,
        str(case["preprocessing"]),
        args.normalization,
        args.max_windows,
        args.stride,
    )
    splits = base.time_split(data, args.train_ratio, args.val_ratio)
    batch_size = args.batch_size or profile.batch_size or 32

    while True:
        try:
            model = base.make_model(str(case["model"]), args.seq_len, len(feature_columns), args.hidden)
            param_count = sum(parameter.numel() for parameter in model.parameters())
            model, curves = train_model(model, splits, profile, args, str(case["objective"]), batch_size)
            val_pred = base.predict(model, splits["val"], profile, batch_size)
            test_pred = base.predict(model, splits["test"], profile, batch_size)
            break
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower() or batch_size <= 4:
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            next_batch = max(4, batch_size // 2)
            print(f"[oom-retry] batch_size {batch_size} -> {next_batch}")
            batch_size = next_batch

    val_metrics = base.evaluate_predictions(splits["val"], val_pred)
    test_metrics = base.evaluate_predictions(splits["test"], test_pred)
    result = {
        "case_id": (
            f"{case['model']}__{case['preprocessing']}__{case['objective']}__seed{case['seed']}"
        ),
        "model": case["model"],
        "preprocessing": case["preprocessing"],
        "objective": case["objective"],
        "seed": int(case["seed"]),
        "param_count": int(param_count),
        "batch_size": int(batch_size),
        "val_selection_score": validation_selection_score(val_metrics),
        "val_copy_risk_ratio": val_metrics["copy_risk_ratio"],
        "val_variance_ratio": val_metrics["variance_ratio"],
        "val_direction_accuracy": val_metrics["direction_accuracy"],
        **test_metrics,
        **regime_metrics(splits["test"], test_pred),
    }
    show_case_diagnostics(curves, splits["test"], test_pred, result)
    return result, val_pred, test_pred, splits


def show_case_diagnostics(curves: pd.DataFrame, test_split, pred: np.ndarray, result: dict[str, object]) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(19, 8), dpi=140)
    axes[0, 0].plot(curves["epoch"], curves["train_loss"], label="train")
    axes[0, 0].plot(curves["epoch"], curves["val_loss"], label="validation")
    axes[0, 0].set_title("Total objective")
    axes[0, 0].legend()

    component_columns = [column for column in curves if column.startswith("component_")]
    for column in component_columns:
        axes[0, 1].plot(curves["epoch"], curves[column], label=column.replace("component_", ""))
    axes[0, 1].set_title("Objective components")
    axes[0, 1].legend(fontsize=8, ncol=2)

    axes[0, 2].plot(curves["epoch"], curves["grad_norm_mean"], label="mean")
    axes[0, 2].plot(curves["epoch"], curves["grad_norm_max"], label="max")
    axes[0, 2].set_title("Gradient norm")
    axes[0, 2].legend()

    n = min(200, len(pred))
    actual_return = test_split["y"][-n:]
    predicted_return = pred[-n:]
    axes[1, 0].plot(actual_return, label="actual return")
    axes[1, 0].plot(predicted_return, label="predicted return")
    axes[1, 0].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 0].set_title("Return prediction")
    axes[1, 0].legend()

    show_candlestick_comparison(
        axes[1, 1],
        test_split,
        pred,
        "Next-candle comparison",
        n=min(60, len(pred)),
    )

    limit = max(float(np.max(np.abs(actual_return))), float(np.max(np.abs(predicted_return))), 1e-4)
    axes[1, 2].scatter(actual_return, predicted_return, s=13, alpha=0.5)
    axes[1, 2].plot([-limit, limit], [-limit, limit], color="black", linewidth=0.8)
    pearson = float(np.corrcoef(actual_return, predicted_return)[0, 1])
    axes[1, 2].set_xlim(-limit, limit)
    axes[1, 2].set_ylim(-limit, limit)
    axes[1, 2].set_xlabel("actual return")
    axes[1, 2].set_ylabel("predicted return")
    axes[1, 2].set_title(f"Calibration scatter / Pearson={pearson:.3f}")
    for ax in axes.flat:
        ax.grid(alpha=0.2)
    fig.suptitle(str(result["case_id"]))
    show_figure(fig)
    display_table(
        f"{result['case_id']} prediction preview",
        prediction_preview_table(test_split, pred),
    )


def ensemble_predictions(
    member_rows,
    val_predictions,
    test_predictions,
    splits,
    top_k: int,
    interval_alpha: float,
):
    ordered = sorted(member_rows, key=lambda row: float(row["val_selection_score"]))[:top_k]
    selected_ids = [str(row["case_id"]) for row in ordered]
    raw_weights = np.asarray(
        [1.0 / max(1e-6, float(row["val_selection_score"])) for row in ordered],
        dtype=np.float64,
    )
    weights = raw_weights / raw_weights.sum()
    val_stack = np.stack([val_predictions[case_id] for case_id in selected_ids])
    test_stack = np.stack([test_predictions[case_id] for case_id in selected_ids])
    outputs = {
        "simple_mean": (val_stack.mean(axis=0), test_stack.mean(axis=0)),
        "validation_weighted": (
            np.average(val_stack, axis=0, weights=weights),
            np.average(test_stack, axis=0, weights=weights),
        ),
        "median": (np.median(val_stack, axis=0), np.median(test_stack, axis=0)),
    }
    rows = []
    interval_outputs = {}
    for name, (val_prediction, test_prediction) in outputs.items():
        metrics = base.evaluate_predictions(splits["test"], test_prediction)
        lower, upper, radius = diag.conformal_interval(
            splits["val"]["y"],
            val_prediction,
            test_prediction,
            interval_alpha,
        )
        interval = diag.interval_metrics(splits["test"]["y"], lower, upper)
        interval_outputs[name] = {
            "prediction": test_prediction,
            "lower": lower,
            "upper": upper,
        }
        rows.append(
            {
                "case_id": f"ensemble__{name}",
                "model": "Ensemble",
                "preprocessing": "mixed",
                "objective": name,
                "seed": -1,
                "param_count": np.nan,
                "batch_size": np.nan,
                "val_selection_score": np.nan,
                "members": ", ".join(selected_ids),
                **metrics,
                **regime_metrics(splits["test"], test_prediction),
                "conformal_radius": radius,
                **interval,
            }
        )
    return rows, interval_outputs, ordered


def show_ensemble_interval(split, output: dict[str, np.ndarray], title: str) -> None:
    import matplotlib.pyplot as plt

    prediction = output["prediction"]
    lower = output["lower"]
    upper = output["upper"]
    actual = split["y"]
    n = min(300, len(prediction))
    x = np.arange(n)
    error = np.abs(actual - prediction)
    width = upper - lower
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), dpi=140)
    axes[0].plot(x, actual[-n:], label="actual return")
    axes[0].plot(x, prediction[-n:], label="ensemble prediction")
    axes[0].fill_between(x, lower[-n:], upper[-n:], alpha=0.25, label="conformal interval")
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_xlabel("test time index")
    axes[0].set_ylabel("next log return")
    axes[0].set_title("Validation-calibrated prediction interval")
    axes[0].legend()
    show_candlestick_comparison(
        axes[1],
        split,
        prediction,
        "Ensemble next-candle comparison",
        n=min(60, len(prediction)),
    )
    axes[2].scatter(width, error, s=14, alpha=0.5)
    axes[2].set_xlabel("interval width")
    axes[2].set_ylabel("absolute return error")
    axes[2].set_title("Interval width versus realized error")
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.suptitle(title)
    show_figure(fig)
    display_table(
        f"{title} preview",
        prediction_preview_table(split, prediction, lower=lower, upper=upper),
    )


def show_summary(summary: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    if summary.empty:
        return
    ordered = summary.sort_values(["collapse_score", "copy_risk_ratio", "mae_krw"])
    columns = [
        "model",
        "preprocessing",
        "objective",
        "seed",
        "mae_krw",
        "copy_risk_ratio",
        "direction_accuracy",
        "variance_ratio",
        "near_zero_return_share",
        "collapse_score",
    ]
    print("\n[leaderboard]")
    print(ordered[columns].head(30).to_string(index=False))

    non_ensemble = ordered[ordered["model"] != "Ensemble"]
    pivot = non_ensemble.pivot_table(index="objective", columns="model", values="copy_risk_ratio", aggfunc="mean")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=140)
    image = axes[0].imshow(pivot.to_numpy(), aspect="auto", cmap="coolwarm")
    axes[0].set_xticks(np.arange(len(pivot.columns)), pivot.columns, rotation=35, ha="right")
    axes[0].set_yticks(np.arange(len(pivot.index)), pivot.index)
    axes[0].set_title("Objective × model copy-risk")
    fig.colorbar(image, ax=axes[0])

    axes[1].scatter(
        non_ensemble["variance_ratio"].clip(1e-3, 1e3),
        non_ensemble["copy_risk_ratio"],
        c=non_ensemble["direction_accuracy"],
        cmap="viridis",
    )
    axes[1].axvline(1.0, color="black", linestyle="--")
    axes[1].axhline(1.0, color="black", linestyle="--")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("variance ratio")
    axes[1].set_ylabel("MAE / persistence MAE")
    axes[1].set_title("Collapse map")

    objective_means = (
        non_ensemble.groupby("objective", as_index=False)
        .agg(copy_risk=("copy_risk_ratio", "mean"), val_score=("val_selection_score", "mean"))
        .sort_values("val_score")
    )
    axes[2].barh(objective_means["objective"], objective_means["copy_risk"])
    axes[2].axvline(1.0, color="black", linestyle="--")
    axes[2].set_title("Mean copy-risk by objective")
    for ax in axes:
        ax.grid(alpha=0.2)
    show_figure(fig)


def display_markdown(text: str) -> None:
    diag.display_markdown(text)


def point_branch_gate(summary: pd.DataFrame) -> tuple[str, str]:
    if summary.empty:
        return "NOT_EVALUATED", "완료된 case가 없다."
    candidates = summary[summary["model"] != "Ensemble"].copy()
    candidates["gate_pass"] = (
        (candidates["copy_risk_ratio"] < 1.0)
        & (candidates["direction_accuracy"] >= 0.52)
        & candidates["variance_ratio"].between(0.25, 4.0)
    )
    grouped = (
        candidates.groupby(["model", "preprocessing", "objective"], as_index=False)
        .agg(passed_seeds=("gate_pass", "sum"), tested_seeds=("seed", "nunique"))
        .sort_values(["passed_seeds", "tested_seeds"], ascending=False)
    )
    qualified = grouped[(grouped["passed_seeds"] >= 2) & (grouped["tested_seeds"] >= 2)]
    if qualified.empty:
        return (
            "NO_GO",
            "두 개 이상 seed에서 persistence·방향·분산 기준을 함께 통과한 조합이 없다.",
        )
    winner = qualified.iloc[0]
    return (
        "GO",
        f"{winner['model']} / {winner['preprocessing']} / {winner['objective']}가 "
        f"{int(winner['passed_seeds'])}개 seed에서 통과했다.",
    )


def build_inline_report(args: argparse.Namespace, summary: pd.DataFrame) -> str:
    best = (
        summary.sort_values(["collapse_score", "copy_risk_ratio", "mae_krw"]).head(15)
        if not summary.empty
        else summary
    )
    branch_status, branch_reason = point_branch_gate(summary)
    return "\n".join(
        [
            "# 10번 objective·ensemble 본실험 실행 요약",
            "",
            "## 연구 질문",
            "",
            "9번에서 전처리만 바꾸어도 persistence 미달과 출력 붕괴가 남았다. "
            "10번은 목적함수의 구성과 validation-only ensemble이 이 실패를 실제로 줄이는지 확인한다.",
            "",
            "## 통과 기준",
            "",
            "- test MAE가 persistence보다 작아 copy-risk ratio가 1 미만이어야 한다.",
            "- variance ratio가 1에 가까워야 하며 0수익률 평탄화와 분산 폭주를 모두 피해야 한다.",
            "- direction accuracy가 0.5 부근의 우연 수준을 안정적으로 넘어야 한다.",
            "- 단일 seed 우승이 아니라 여러 seed와 ensemble에서 재현되어야 한다.",
            "- 모델 선택과 ensemble 가중치는 validation 결과만 사용하고 test 결과를 선택에 쓰지 않는다.",
            "",
            "## 점예측 연구선 판정",
            "",
            f"- 상태: `{branch_status}`",
            f"- 근거: {branch_reason}",
            "- 통과 기준: copy-risk ratio < 1, direction accuracy >= 0.52, variance ratio 0.25~4.0을 두 개 이상 seed에서 동시에 만족한다.",
            "",
            "## 실행 설정",
            "",
            "```json",
            json.dumps(vars(args), ensure_ascii=False, indent=2),
            "```",
            "",
            "## 상위 결과",
            "",
            "```text",
            best.to_string(index=False),
            "```",
            "",
            "## 해석",
            "",
            "objective가 train loss만 낮추고 copy-risk ratio, variance ratio, 방향 정확도를 개선하지 못하면 "
            "최적화는 성공한 것이 아니라 다른 형태의 쉬운 해를 찾은 것이다.",
        ]
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="10번 objective·ensemble 본실험")
    parser.add_argument("--db", default=None)
    parser.add_argument("--table", default="btc_15m_advance")
    parser.add_argument("--ticker", default=None)
    parser.add_argument("--profile", default="school_4090_15gb")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument(
        "--parallel-slot",
        choices=sorted(PARALLEL_RESOURCE_SLOTS),
        default="point_primary",
        help="10번과 11번을 동시에 실행할 때 사용할 자원 분할 슬롯",
    )
    parser.add_argument(
        "--suite",
        choices=[
            "parallel_point_probe",
            "objective_screen",
            "failure_mode_control",
            "seed_confirmation",
            "ensemble_confirmation",
            "full",
        ],
        default="parallel_point_probe",
    )
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--preprocessings", default=",".join(DEFAULT_PREPROCESSINGS))
    parser.add_argument("--objectives", default=",".join(DEFAULT_OBJECTIVES))
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--feature-set", choices=sorted(base.FEATURE_SETS), default="wide_stationary")
    parser.add_argument("--normalization", default="window_standard")
    parser.add_argument("--optimizer", default="adamw")
    parser.add_argument("--scheduler", default="cosine")
    parser.add_argument("--gradient-policy", default="clip1")
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--objective-warmup-epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=1e-6)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-rows", type=int, default=40000)
    parser.add_argument("--max-windows", type=int, default=4096)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ensemble-top-k", type=int, default=5)
    parser.add_argument("--interval-alpha", type=float, default=0.10)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-failure", action="store_true")
    return parser.parse_known_args(argv)[0]


def main(argv: list[str] | None = None) -> None:
    configure_inline_matplotlib()
    args = parse_args(argv)
    profile = base.build_resource_profile(args.profile, args.device)
    profile, parallel_settings = apply_parallel_resource_slot(profile, args.parallel_slot)
    base.apply_resource_profile(profile)
    environment = base.log_environment(profile)
    environment["parallel_settings"] = parallel_settings
    print("[environment]", json.dumps(environment, ensure_ascii=False, indent=2))

    cases = build_cases(args)
    print(f"[plan] suite={args.suite} cases={len(cases)}")
    print(pd.DataFrame(cases).head(60).to_string(index=False))
    if args.dry_run:
        return

    db_path = base.resolve_db_path(args.db)
    raw = base.load_price_data(db_path, args.table, args.ticker, args.max_rows)
    features = base.make_features(raw)
    print("[statistics]", json.dumps(base.basic_statistics(features), ensure_ascii=False, indent=2))
    display_table(
        "missingness audit",
        build_missingness_audit(raw, features, "다음 15분 next-return / next-close"),
    )

    rows: list[dict[str, object]] = []
    val_predictions: dict[str, np.ndarray] = {}
    test_predictions: dict[str, np.ndarray] = {}
    latest_splits = None
    for index, case in enumerate(cases, start=1):
        print(f"\n[case {index}/{len(cases)}] {case}")
        try:
            result, val_pred, test_pred, splits = run_case(case, features, profile, args)
            rows.append(result)
            case_id = str(result["case_id"])
            val_predictions[case_id] = val_pred
            test_predictions[case_id] = test_pred
            latest_splits = splits
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and torch.cuda.is_available():
                torch.cuda.empty_cache()
            if not args.continue_on_failure:
                raise
            print(f"[case failed] {case}: {exc}")
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    if args.suite in {"parallel_point_probe", "ensemble_confirmation", "full"} and rows and latest_splits is not None:
        ensemble_rows, interval_outputs, selected = ensemble_predictions(
            rows,
            val_predictions,
            test_predictions,
            latest_splits,
            min(args.ensemble_top_k, len(rows)),
            args.interval_alpha,
        )
        print("\n[ensemble members selected by validation only]")
        print(pd.DataFrame(selected)[["case_id", "val_selection_score"]].to_string(index=False))
        rows.extend(ensemble_rows)
        show_ensemble_interval(
            latest_splits["test"],
            interval_outputs["validation_weighted"],
            "Validation-weighted ensemble",
        )

    summary = pd.DataFrame(rows)
    show_summary(summary)
    display_markdown(build_inline_report(args, summary))
    print("[done] all diagnostics displayed inline; no PNG/CSV/Markdown artifacts were written.")


if __name__ == "__main__":
    main()
