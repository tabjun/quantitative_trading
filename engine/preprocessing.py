"""전처리 파이프라인과 conformal interval, inline matplotlib 설정.

- `moving_average_np`, `apply_window_preprocessing`는 8번에서 옮겼다.
- atomic/pipeline 전처리(`apply_atomic_preprocessing`, `apply_preprocessing_pipeline`)와
  `conformal_interval`, `interval_metrics`, `configure_inline_matplotlib`는 9번에서 옮겼다.
- 9번에서 `base.moving_average_np`로 부르던 indirection은 같은 모듈의 `moving_average_np`로 연결한다.
"""

from __future__ import annotations

import math

import numpy as np


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
        return (x - moving_average_np(x, odd_kernel(len(x), kernel))).astype(np.float32)
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
        local_vol = np.sqrt(moving_average_np(diff**2, odd_kernel(len(x), 9)) + 1e-6)
        return (x / local_vol).astype(np.float32)
    raise ValueError(f"지원하지 않는 전처리 원자: {mode}")


def apply_preprocessing_pipeline(window: np.ndarray, pipeline: str) -> np.ndarray:
    result = window.copy()
    for step in pipeline.split("+"):
        result = apply_atomic_preprocessing(result, step)
    return np.nan_to_num(result, nan=0.0, posinf=10.0, neginf=-10.0).astype(np.float32)


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
