# [FOR COMMIT TRACKING ONLY - DO NOT EXECUTE]
# This file is automatically mirrored from the corresponding .ipynb for git diff purposes.
# Actual research execution should be performed in the Jupyter Notebook (.ipynb)
# or in an approved remote/server environment.

# %% [markdown]
# # 9번 전처리·불확실성·용량 진단 실험
#
# 8번 breadth 결과를 바탕으로 전처리 조합, seed 불확실성, conformal prediction interval, 모델 용량 변화를 함께 진단합니다.
# 모든 그림과 표는 notebook inline output으로만 표시하며 서버에 PNG/CSV/Markdown을 저장하지 않습니다.

# %%
# [FOR COMMIT TRACKING ONLY - DO NOT EXECUTE LOCALLY]
# This file mirrors the corresponding notebook for review and server execution.

"""9번: 전처리 조합, 불확실성, 모델 용량을 함께 진단하는 후속 실험.

실제 연구 실행은 학교 서버 CUDA Jupyter 커널에서 수행한다.
모든 시각화는 notebook inline output으로만 표시하며 파일로 저장하지 않는다.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents, Path.home() / "personal_ai_project" / "quantitative_trading"]:
        if (candidate / "pyproject.toml").exists() and (candidate / "test").exists():
            return candidate
    return start


REPO_ROOT = find_repo_root(Path.cwd())
BACKEND_PATH = REPO_ROOT / "test" / "models" / "8_optimization_breadth_training_test.py"


def load_backend():
    spec = importlib.util.spec_from_file_location("optimization_breadth_backend", BACKEND_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"8번 backend를 불러오지 못했다: {BACKEND_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = load_backend()


VIDEO_BLACK_SCHOLES = "https://www.youtube.com/watch?v=99BHnu64pu8"
VIDEO_DOUBLE_DESCENT = "https://www.youtube.com/watch?v=5ruiphjlOwo"
SHARED_REVIEW = "https://chatgpt.com/share/6a395764-1d2c-83ee-92ac-20ada70cf9db"

PREPROCESSING_PIPELINES = [
    "none",
    "winsor_005",
    "winsor_01",
    "winsor_025",
    "hampel_3",
    "asinh_robust",
    "signed_log1p",
    "first_diff",
    "seasonal_diff4",
    "seasonal_diff16",
    "ema_residual_5",
    "ema_residual_9",
    "ema_residual_17",
    "linear_detrend",
    "median_residual_5",
    "frequency_highpass_1",
    "frequency_highpass_3",
    "frequency_highpass_5",
    "frequency_bandpass",
    "volatility_scale",
    "winsor_01+asinh_robust",
    "hampel_3+asinh_robust",
    "winsor_01+ema_residual_9",
    "asinh_robust+ema_residual_9",
    "winsor_01+frequency_highpass_3",
    "volatility_scale+asinh_robust",
    "linear_detrend+asinh_robust",
    "seasonal_diff4+asinh_robust",
]

DEFAULT_MODELS = ["Linear", "PatchTSTLike", "TimesNetLike", "AutoformerLike"]
UNCERTAINTY_PIPELINES = [
    "none",
    "winsor_01+asinh_robust",
    "asinh_robust+ema_residual_9",
    "volatility_scale+asinh_robust",
]
CAPACITY_WIDTHS = [24, 48, 64, 96, 128, 192, 256]


def configure_inline_matplotlib() -> None:
    """Force Jupyter to retain PNG output instead of silently using a headless backend."""

    try:
        from IPython import get_ipython

        ip = get_ipython()
        if ip is not None:
            ip.run_line_magic("matplotlib", "inline")
            from matplotlib_inline.backend_inline import set_matplotlib_formats

            set_matplotlib_formats("png")
    except Exception as exc:
        print(f"[plot] inline backend 설정을 건너뜀: {exc}")


def show_figure(fig) -> None:
    import matplotlib.pyplot as plt

    fig.tight_layout()
    plt.show()
    plt.close(fig)


def odd_kernel(length: int, requested: int) -> int:
    value = min(requested, length if length % 2 == 1 else length - 1)
    return max(3, value if value % 2 == 1 else value - 1)


def rolling_median_np(window: np.ndarray, kernel: int) -> np.ndarray:
    kernel = odd_kernel(len(window), kernel)
    radius = kernel // 2
    padded = np.pad(window, ((radius, radius), (0, 0)), mode="edge")
    result = np.empty_like(window)
    for index in range(len(window)):
        result[index] = np.nanmedian(padded[index : index + kernel], axis=0)
    return result


def robust_scale(window: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    median = np.nanmedian(window, axis=0, keepdims=True)
    q25 = np.nanquantile(window, 0.25, axis=0, keepdims=True)
    q75 = np.nanquantile(window, 0.75, axis=0, keepdims=True)
    scale = np.where((q75 - q25) < 1e-6, 1.0, q75 - q25)
    return median, scale


def apply_atomic_preprocessing(window: np.ndarray, mode: str) -> np.ndarray:
    x = np.asarray(window, dtype=np.float32)
    if mode == "none":
        return x
    if mode.startswith("winsor_"):
        alpha = {"winsor_005": 0.005, "winsor_01": 0.01, "winsor_025": 0.025}[mode]
        lo = np.nanquantile(x, alpha, axis=0, keepdims=True)
        hi = np.nanquantile(x, 1.0 - alpha, axis=0, keepdims=True)
        return np.clip(x, lo, hi)
    if mode == "hampel_3":
        median = rolling_median_np(x, 7)
        abs_dev = np.abs(x - median)
        mad = rolling_median_np(abs_dev, 7)
        threshold = 3.0 * 1.4826 * np.where(mad < 1e-6, 1.0, mad)
        return np.where(abs_dev > threshold, median, x)
    if mode == "asinh_robust":
        median, scale = robust_scale(x)
        return np.arcsinh((x - median) / scale).astype(np.float32)
    if mode == "signed_log1p":
        median, scale = robust_scale(x)
        centered = (x - median) / scale
        return (np.sign(centered) * np.log1p(np.abs(centered))).astype(np.float32)
    if mode == "first_diff":
        return np.diff(x, axis=0, prepend=x[:1]).astype(np.float32)
    if mode.startswith("seasonal_diff"):
        lag = int(mode.replace("seasonal_diff", ""))
        result = np.zeros_like(x)
        result[lag:] = x[lag:] - x[:-lag]
        return result
    if mode.startswith("ema_residual_"):
        kernel = int(mode.rsplit("_", 1)[1])
        return (x - base.moving_average_np(x, odd_kernel(len(x), kernel))).astype(np.float32)
    if mode == "linear_detrend":
        t = np.linspace(-1.0, 1.0, len(x), dtype=np.float32)
        design = np.column_stack([np.ones_like(t), t])
        beta = np.linalg.lstsq(design, x, rcond=None)[0]
        return (x - design @ beta).astype(np.float32)
    if mode == "median_residual_5":
        return (x - rolling_median_np(x, 5)).astype(np.float32)
    if mode.startswith("frequency_highpass_"):
        cutoff = int(mode.rsplit("_", 1)[1])
        spectrum = np.fft.rfft(x, axis=0)
        spectrum[: min(cutoff + 1, len(spectrum))] = 0
        return np.fft.irfft(spectrum, n=len(x), axis=0).astype(np.float32)
    if mode == "frequency_bandpass":
        spectrum = np.fft.rfft(x, axis=0)
        keep = np.zeros_like(spectrum)
        low = min(2, len(spectrum))
        high = min(max(low + 1, len(spectrum) // 3), len(spectrum))
        keep[low:high] = spectrum[low:high]
        return np.fft.irfft(keep, n=len(x), axis=0).astype(np.float32)
    if mode == "volatility_scale":
        diff = np.diff(x, axis=0, prepend=x[:1])
        local_vol = np.sqrt(base.moving_average_np(diff**2, odd_kernel(len(x), 9)) + 1e-6)
        return (x / local_vol).astype(np.float32)
    raise ValueError(f"지원하지 않는 전처리 원자: {mode}")


def apply_preprocessing_pipeline(window: np.ndarray, pipeline: str) -> np.ndarray:
    result = window.copy()
    for step in pipeline.split("+"):
        result = apply_atomic_preprocessing(result, step)
    return np.nan_to_num(result, nan=0.0, posinf=10.0, neginf=-10.0).astype(np.float32)


def show_raw_data_diagnostics(features: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    recent = features.tail(min(4000, len(features))).copy()
    fig, axes = plt.subplots(3, 2, figsize=(14, 11), dpi=150)
    axes[0, 0].plot(pd.to_datetime(recent["timestamp"]), recent["close"], linewidth=1)
    axes[0, 0].set_title("Raw BTC close")
    axes[0, 0].set_ylabel("KRW")

    axes[0, 1].plot(pd.to_datetime(recent["timestamp"]), recent["log_return_1"], linewidth=0.8)
    axes[0, 1].axhline(0, color="black", linewidth=0.7)
    axes[0, 1].set_title("15-minute log return")

    axes[1, 0].hist(features["log_return_1"], bins=120, density=True, alpha=0.8)
    axes[1, 0].set_title("Return distribution: center and heavy tails")

    ordered = np.sort(features["log_return_1"].to_numpy())
    theoretical = np.sort(np.random.default_rng(42).normal(ordered.mean(), ordered.std(), len(ordered)))
    axes[1, 1].scatter(theoretical[::20], ordered[::20], s=5, alpha=0.45)
    limits = [min(theoretical.min(), ordered.min()), max(theoretical.max(), ordered.max())]
    axes[1, 1].plot(limits, limits, color="black", linewidth=0.8)
    axes[1, 1].set_title("Normal QQ diagnostic")
    axes[1, 1].set_xlabel("Normal-theory quantile")
    axes[1, 1].set_ylabel("Observed return quantile")

    axes[2, 0].plot(pd.to_datetime(recent["timestamp"]), recent["realized_vol_64"], linewidth=1)
    axes[2, 0].set_title("Rolling realized volatility")

    rolling_mean = recent["log_return_1"].rolling(256).mean()
    rolling_std = recent["log_return_1"].rolling(256).std()
    axes[2, 1].plot(pd.to_datetime(recent["timestamp"]), rolling_mean, label="rolling mean")
    axes[2, 1].plot(pd.to_datetime(recent["timestamp"]), rolling_std, label="rolling std")
    axes[2, 1].set_title("Rolling distribution drift")
    axes[2, 1].legend()
    for ax in axes.flat:
        ax.grid(alpha=0.2)
    show_figure(fig)


def show_time_split(data: dict[str, np.ndarray], train_ratio: float, val_ratio: float) -> None:
    import matplotlib.pyplot as plt

    n = len(data["y"])
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    timestamps = pd.to_datetime(data["timestamp"])
    fig, ax = plt.subplots(figsize=(13, 4), dpi=150)
    ax.plot(timestamps, data["y"], color="0.35", linewidth=0.7)
    ax.axvspan(timestamps[0], timestamps[train_end - 1], alpha=0.14, color="tab:blue", label="train")
    ax.axvspan(timestamps[train_end], timestamps[val_end - 1], alpha=0.18, color="tab:orange", label="validation")
    ax.axvspan(timestamps[val_end], timestamps[-1], alpha=0.18, color="tab:green", label="test")
    ax.set_title("Time-ordered split: leakage-free train / validation / test")
    ax.set_ylabel("next log return")
    ax.legend()
    ax.grid(alpha=0.2)
    show_figure(fig)


def show_preprocessing_gallery(window: np.ndarray, feature_names: list[str], pipelines: list[str]) -> None:
    import matplotlib.pyplot as plt

    selected = pipelines[: min(12, len(pipelines))]
    feature_index = 0
    cols = 3
    rows = math.ceil(len(selected) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3.2 * rows), dpi=145, squeeze=False)
    for ax, pipeline in zip(axes.flat, selected):
        transformed = apply_preprocessing_pipeline(window, pipeline)
        ax.plot(window[:, feature_index], label="before", alpha=0.55)
        ax.plot(transformed[:, feature_index], label="after", linewidth=1.4)
        ax.set_title(pipeline)
        ax.grid(alpha=0.2)
    for ax in axes.flat[len(selected) :]:
        ax.axis("off")
    axes.flat[0].legend()
    fig.suptitle(f"Preprocessing gallery: {feature_names[feature_index]}", y=1.01)
    show_figure(fig)


def show_distribution_gallery(window: np.ndarray, pipelines: list[str]) -> None:
    import matplotlib.pyplot as plt

    selected = pipelines[: min(12, len(pipelines))]
    cols = 3
    rows = math.ceil(len(selected) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3.2 * rows), dpi=145, squeeze=False)
    for ax, pipeline in zip(axes.flat, selected):
        transformed = apply_preprocessing_pipeline(window, pipeline).ravel()
        ax.hist(transformed, bins=45, alpha=0.8)
        ax.set_title(pipeline)
        ax.grid(alpha=0.2)
    for ax in axes.flat[len(selected) :]:
        ax.axis("off")
    fig.suptitle("Distribution after each preprocessing pipeline", y=1.01)
    show_figure(fig)


def show_frequency_gallery(window: np.ndarray, pipelines: list[str]) -> None:
    import matplotlib.pyplot as plt

    selected = pipelines[: min(8, len(pipelines))]
    fig, axes = plt.subplots(math.ceil(len(selected) / 2), 2, figsize=(14, 3.3 * math.ceil(len(selected) / 2)), dpi=145)
    axes = np.atleast_1d(axes).ravel()
    for ax, pipeline in zip(axes, selected):
        transformed = apply_preprocessing_pipeline(window, pipeline)[:, 0]
        power = np.abs(np.fft.rfft(transformed)) ** 2
        ax.plot(np.arange(len(power)), power, linewidth=1)
        ax.set_yscale("log")
        ax.set_title(pipeline)
        ax.set_xlabel("frequency bin")
        ax.set_ylabel("power")
        ax.grid(alpha=0.2)
    for ax in axes[len(selected) :]:
        ax.axis("off")
    fig.suptitle("Frequency-domain effect of preprocessing", y=1.01)
    show_figure(fig)


def show_learning_diagnostics(curves: pd.DataFrame, title: str) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), dpi=150)
    axes[0].plot(curves["epoch"], curves["train_loss"], label="train")
    axes[0].plot(curves["epoch"], curves["val_loss"], label="validation")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(curves["epoch"], curves["grad_norm_mean"], label="mean")
    axes[1].plot(curves["epoch"], curves["grad_norm_max"], label="max", alpha=0.75)
    axes[1].set_title("Gradient norm")
    axes[1].legend()

    axes[2].plot(curves["epoch"], curves["lr"])
    axes[2].set_title("Learning rate")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.2)
    fig.suptitle(title)
    show_figure(fig)


def show_prediction_diagnostics(
    split: dict[str, np.ndarray],
    prediction: np.ndarray,
    title: str,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
) -> None:
    import matplotlib.pyplot as plt

    n = min(300, len(prediction))
    actual = split["y"][-n:]
    pred = prediction[-n:]
    prev = split["prev_close"][-n:]
    actual_price = split["target_close"][-n:]
    pred_price = prev * np.exp(pred)
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), dpi=150, sharex=True)
    x = np.arange(n)
    axes[0].plot(x, actual, label="actual return")
    axes[0].plot(x, pred, label="predicted return")
    if lower is not None and upper is not None:
        axes[0].fill_between(x, lower[-n:], upper[-n:], alpha=0.22, label="prediction interval")
    axes[0].axhline(0, color="black", linewidth=0.7)
    axes[0].legend()
    axes[0].set_title("Return prediction")

    axes[1].plot(x, actual_price, label="actual next close")
    axes[1].plot(x, pred_price, label="predicted next close")
    axes[1].plot(x, prev, label="persistence", alpha=0.7)
    axes[1].legend()
    axes[1].set_title("KRW reconstruction")

    axes[2].scatter(actual, pred, s=10, alpha=0.45)
    limit = max(np.max(np.abs(actual)), np.max(np.abs(pred)))
    axes[2].plot([-limit, limit], [-limit, limit], color="black", linewidth=0.7)
    axes[2].set_xlabel("actual return")
    axes[2].set_ylabel("predicted return")
    axes[2].set_title("Calibration scatter")
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.suptitle(title)
    show_figure(fig)


def conformal_interval(
    val_actual: np.ndarray,
    val_prediction: np.ndarray,
    test_prediction: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    scores = np.abs(val_actual - val_prediction)
    level = min(1.0, math.ceil((len(scores) + 1) * (1.0 - alpha)) / len(scores))
    radius = float(np.quantile(scores, level, method="higher"))
    return test_prediction - radius, test_prediction + radius, radius


def interval_metrics(actual: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> dict[str, float]:
    covered = (actual >= lower) & (actual <= upper)
    return {
        "interval_coverage": float(np.mean(covered)),
        "interval_width": float(np.mean(upper - lower)),
        "interval_miss_distance": float(np.mean(np.maximum(lower - actual, 0) + np.maximum(actual - upper, 0))),
    }


def show_uncertainty_diagnostics(
    split: dict[str, np.ndarray],
    predictions: list[np.ndarray],
    val_actual: np.ndarray,
    val_predictions: list[np.ndarray],
    title: str,
    alpha: float,
) -> dict[str, float]:
    import matplotlib.pyplot as plt

    stack = np.stack(predictions)
    val_stack = np.stack(val_predictions)
    mean_pred = stack.mean(axis=0)
    epistemic_std = stack.std(axis=0)
    val_mean = val_stack.mean(axis=0)
    lower, upper, conformal_radius = conformal_interval(val_actual, val_mean, mean_pred, alpha)
    lower = lower - 1.645 * epistemic_std
    upper = upper + 1.645 * epistemic_std
    metrics = interval_metrics(split["y"], lower, upper)
    metrics["conformal_radius"] = conformal_radius
    metrics["epistemic_std_mean"] = float(epistemic_std.mean())

    n = min(300, len(mean_pred))
    x = np.arange(n)
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), dpi=150, sharex=True)
    axes[0].plot(x, split["y"][-n:], label="actual")
    axes[0].plot(x, mean_pred[-n:], label="ensemble mean")
    axes[0].fill_between(x, lower[-n:], upper[-n:], alpha=0.23, label="conformal + ensemble interval")
    axes[0].legend()
    axes[0].set_title("Prediction interval")
    axes[1].plot(x, epistemic_std[-n:], label="seed disagreement")
    axes[1].plot(x, np.abs(split["y"][-n:] - mean_pred[-n:]), label="absolute error", alpha=0.75)
    axes[1].legend()
    axes[1].set_title("Uncertainty versus realized error")
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.suptitle(title)
    show_figure(fig)
    show_prediction_diagnostics(split, mean_pred, title, lower, upper)
    return metrics


def show_summary(summary: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    if summary.empty:
        return
    ordered = summary.sort_values(["collapse_score", "mae_krw"]).reset_index(drop=True)
    display_columns = [
        "model",
        "preprocessing",
        "seed",
        "hidden",
        "mae_krw",
        "copy_risk_ratio",
        "direction_accuracy",
        "variance_ratio",
        "near_zero_return_share",
        "collapse_score",
    ]
    print("\n[leaderboard]\n", ordered[display_columns].head(40).to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=150)
    labels = (ordered["model"] + "\n" + ordered["preprocessing"]).head(25)
    axes[0].bar(np.arange(len(labels)), ordered["copy_risk_ratio"].head(25))
    axes[0].axhline(1.0, color="black", linewidth=0.8)
    axes[0].set_title("Copy-risk ratio")
    axes[0].set_xticks(np.arange(len(labels)), labels, rotation=90, fontsize=7)

    axes[1].scatter(ordered["variance_ratio"], ordered["direction_accuracy"], c=ordered["collapse_score"], cmap="viridis")
    axes[1].axvline(1.0, color="black", linewidth=0.8)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("variance ratio")
    axes[1].set_ylabel("direction accuracy")
    axes[1].set_title("Information preservation")

    axes[2].scatter(ordered["param_count"], ordered["mae_krw"], c=ordered["collapse_score"], cmap="plasma")
    axes[2].set_xscale("log")
    axes[2].set_xlabel("parameter count")
    axes[2].set_ylabel("MAE KRW")
    axes[2].set_title("Capacity versus generalization error")
    for ax in axes:
        ax.grid(alpha=0.2)
    show_figure(fig)

    pivot = ordered.pivot_table(index="preprocessing", columns="model", values="copy_risk_ratio", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(12, max(6, len(pivot) * 0.35)), dpi=150)
    image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="coolwarm", vmin=0.9, vmax=min(2.0, np.nanmax(pivot.to_numpy())))
    ax.set_xticks(np.arange(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)), pivot.index)
    ax.set_title("Preprocessing × model copy-risk heatmap")
    fig.colorbar(image, ax=ax, label="MAE / persistence MAE")
    show_figure(fig)


def build_cases(args: argparse.Namespace) -> list[dict[str, object]]:
    models = [item.strip() for item in args.models.split(",") if item.strip()]
    pipelines = [item.strip() for item in args.preprocessings.split(",") if item.strip()]
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    widths = [int(item) for item in args.widths.split(",") if item.strip()]

    cases: list[dict[str, object]] = []
    if args.suite == "preprocessing_matrix":
        for pipeline in pipelines:
            for model in models:
                cases.append({"model": model, "preprocessing": pipeline, "seed": seeds[0], "hidden": args.hidden})
    elif args.suite == "uncertainty_probe":
        for pipeline in UNCERTAINTY_PIPELINES:
            for model in models[: min(3, len(models))]:
                for seed in seeds:
                    cases.append({"model": model, "preprocessing": pipeline, "seed": seed, "hidden": args.hidden})
    elif args.suite == "capacity_probe":
        for model in models[: min(3, len(models))]:
            for width in widths:
                for seed in seeds:
                    cases.append({"model": model, "preprocessing": pipelines[0], "seed": seed, "hidden": width})
    elif args.suite == "full":
        for pipeline in pipelines:
            for model in models:
                for seed in seeds:
                    cases.append({"model": model, "preprocessing": pipeline, "seed": seed, "hidden": args.hidden})
    else:
        raise ValueError(f"알 수 없는 suite: {args.suite}")
    return cases[: args.max_cases] if args.max_cases > 0 else cases


def run_case(
    case: dict[str, object],
    features: pd.DataFrame,
    profile,
    args: argparse.Namespace,
) -> tuple[dict[str, object], pd.DataFrame, np.ndarray, np.ndarray, dict[str, dict[str, np.ndarray]]]:
    seed = int(case["seed"])
    base.set_seed(seed)
    feature_columns = base.FEATURE_SETS[args.feature_set]
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
            model = base.make_model(str(case["model"]), args.seq_len, len(feature_columns), int(case["hidden"]))
            param_count = sum(parameter.numel() for parameter in model.parameters())
            model, curves = base.train_one_model(
                model,
                splits,
                profile,
                args.epochs,
                batch_size,
                args.lr,
                args.weight_decay,
                args.loss,
                args.optimizer,
                args.scheduler,
                args.gradient_policy,
            )
            val_prediction = base.predict(model, splits["val"], profile, batch_size)
            test_prediction = base.predict(model, splits["test"], profile, batch_size)
            break
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower() or batch_size <= 4:
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            next_batch = max(4, batch_size // 2)
            print(f"[oom-retry] batch_size {batch_size} -> {next_batch}")
            batch_size = next_batch
    metrics = base.evaluate_predictions(splits["test"], test_prediction)
    result = {
        "case_id": f"{case['model']}__{case['preprocessing']}__seed{seed}__h{case['hidden']}",
        "suite": args.suite,
        "model": case["model"],
        "preprocessing": case["preprocessing"],
        "normalization": args.normalization,
        "loss": args.loss,
        "seed": seed,
        "hidden": int(case["hidden"]),
        "param_count": int(param_count),
        "batch_size": batch_size,
        **metrics,
    }
    show_learning_diagnostics(curves, str(result["case_id"]))
    show_prediction_diagnostics(splits["test"], test_prediction, str(result["case_id"]))
    return result, curves, val_prediction, test_prediction, splits


def display_markdown(text: str) -> None:
    try:
        from IPython.display import Markdown, display

        display(Markdown(text))
    except Exception:
        print(text)


def build_inline_report(args: argparse.Namespace, summary: pd.DataFrame, uncertainty_rows: list[dict[str, object]]) -> str:
    best = summary.sort_values(["collapse_score", "mae_krw"]).head(15)
    lines = [
        "# 9번 전처리·불확실성 진단 실행 요약",
        "",
        "## 목적",
        "",
        "8번에서 모든 모델이 persistence를 넘지 못한 원인이 모델 구조보다 전처리·분포 이동·불확실성 처리에 있는지 확인한다.",
        "",
        "## 영상 반영",
        "",
        f"- [Black–Scholes 영상]({VIDEO_BLACK_SCHOLES}): 점예측 대신 분포·변동성·예측구간을 평가한다.",
        f"- [Double Descent 영상]({VIDEO_DOUBLE_DESCENT}): width, parameter count, seed, epoch 변화에 따른 일반화 오차를 평가한다.",
        f"- [공유 정리]({SHARED_REVIEW}): 기울기 문제를 단독 원인으로 단정하지 않고 bias, variance, noise, optimizer가 선택한 해를 분리해 본다.",
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
    ]
    if uncertainty_rows:
        lines.extend(
            [
                "",
                "## 불확실성 결과",
                "",
                "```text",
                pd.DataFrame(uncertainty_rows).to_string(index=False),
                "```",
            ]
        )
    lines.extend(
        [
            "",
            "## 해석 원칙",
            "",
            "- MAE만 낮고 variance ratio가 0에 가까우면 좋은 예측이 아니라 0수익률 collapse다.",
            "- interval coverage가 높아도 폭이 지나치게 넓으면 실용적인 불확실성 모델이 아니다.",
            "- seed 간 예측 분산이 크면 단일 실행의 순위를 신뢰하지 않는다.",
            "- 모델 크기 증가에 따라 test error가 단조롭게 변한다고 가정하지 않는다.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="9번 전처리·불확실성·용량 진단 실험")
    parser.add_argument("--db", default=None)
    parser.add_argument("--table", default="btc_15m_advance")
    parser.add_argument("--ticker", default=None)
    parser.add_argument("--profile", default="school_4090_15gb")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument(
        "--suite",
        choices=["preprocessing_matrix", "uncertainty_probe", "capacity_probe", "full"],
        default="preprocessing_matrix",
    )
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--preprocessings", default=",".join(PREPROCESSING_PIPELINES))
    parser.add_argument("--seeds", default="42,137,2026")
    parser.add_argument("--widths", default=",".join(map(str, CAPACITY_WIDTHS)))
    parser.add_argument("--feature-set", choices=sorted(base.FEATURE_SETS), default="wide_stationary")
    parser.add_argument("--normalization", default="window_standard")
    parser.add_argument("--loss", default="return_huber")
    parser.add_argument("--optimizer", default="adamw")
    parser.add_argument("--scheduler", default="cosine")
    parser.add_argument("--gradient-policy", default="clip1")
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-rows", type=int, default=40000)
    parser.add_argument("--max-windows", type=int, default=4096)
    parser.add_argument("--max-cases", type=int, default=112)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--interval-alpha", type=float, default=0.10)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-failure", action="store_true")
    return parser.parse_known_args(argv)[0]


def main(argv: list[str] | None = None) -> None:
    configure_inline_matplotlib()
    args = parse_args(argv)
    profile = base.build_resource_profile(args.profile, args.device)
    base.apply_resource_profile(profile)
    environment = base.log_environment(profile)
    print("[environment]", json.dumps(environment, ensure_ascii=False, indent=2))

    cases = build_cases(args)
    print(f"[plan] suite={args.suite} cases={len(cases)}")
    print(pd.DataFrame(cases).head(40).to_string(index=False))
    if args.dry_run:
        return

    db_path = base.resolve_db_path(args.db)
    raw = base.load_price_data(db_path, args.table, args.ticker, args.max_rows)
    features = base.make_features(raw)
    print("[statistics]", json.dumps(base.basic_statistics(features), ensure_ascii=False, indent=2))

    base.apply_window_preprocessing = apply_preprocessing_pipeline
    feature_columns = base.FEATURE_SETS[args.feature_set]
    raw_values = features[feature_columns].to_numpy(np.float32)
    example_window = raw_values[-args.seq_len :].copy()

    show_raw_data_diagnostics(features)
    show_preprocessing_gallery(example_window, feature_columns, [str(case["preprocessing"]) for case in cases])
    show_distribution_gallery(example_window, [str(case["preprocessing"]) for case in cases])
    show_frequency_gallery(example_window, [str(case["preprocessing"]) for case in cases])

    preview_data = base.build_windows(
        features,
        feature_columns,
        args.seq_len,
        str(cases[0]["preprocessing"]),
        args.normalization,
        args.max_windows,
        args.stride,
    )
    show_time_split(preview_data, args.train_ratio, args.val_ratio)

    rows: list[dict[str, object]] = []
    predictions: dict[tuple[str, str, int], np.ndarray] = {}
    val_predictions: dict[tuple[str, str, int], np.ndarray] = {}
    split_by_group: dict[tuple[str, str], dict[str, dict[str, np.ndarray]]] = {}

    for index, case in enumerate(cases, start=1):
        print(f"\n[case {index}/{len(cases)}] {case}")
        try:
            result, _, val_pred, test_pred, splits = run_case(case, features, profile, args)
            rows.append(result)
            key = (str(case["model"]), str(case["preprocessing"]), int(case["seed"]))
            predictions[key] = test_pred
            val_predictions[key] = val_pred
            split_by_group[(str(case["model"]), str(case["preprocessing"]))] = splits
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and torch.cuda.is_available():
                torch.cuda.empty_cache()
            if not args.continue_on_failure:
                raise
            print(f"[case failed] {case}: {exc}")

    summary = pd.DataFrame(rows)
    show_summary(summary)

    uncertainty_rows: list[dict[str, object]] = []
    grouped_seeds: dict[tuple[str, str], list[int]] = defaultdict(list)
    for model, pipeline, seed in predictions:
        grouped_seeds[(model, pipeline)].append(seed)
    for group, seeds in grouped_seeds.items():
        if len(seeds) < 2:
            continue
        model, pipeline = group
        splits = split_by_group[group]
        metrics = show_uncertainty_diagnostics(
            splits["test"],
            [predictions[(model, pipeline, seed)] for seed in seeds],
            splits["val"]["y"],
            [val_predictions[(model, pipeline, seed)] for seed in seeds],
            f"{model} / {pipeline}",
            args.interval_alpha,
        )
        uncertainty_rows.append({"model": model, "preprocessing": pipeline, "seeds": seeds, **metrics})

    display_markdown(build_inline_report(args, summary, uncertainty_rows))
    print("[done] all diagnostics displayed inline; no PNG/CSV/Markdown artifacts were written.")


if __name__ == "__main__":
    main()
