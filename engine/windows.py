"""윈도우 생성과 시간 분할.

- `build_windows`, `time_split`는 8번에서 옮겼다.
- `build_risk_windows`, `prepare_event_splits`는 11번에서 옮겼다.

8번 원본은 build_windows가 전역 `apply_window_preprocessing`(기본=atomic)을 호출하고
10/12가 실행 시 이를 full pipeline으로 monkeypatch했다. 패키지에서는 그 전역 mutation이
프로세스 전체로 새므로(fix②), 여기서는 처음부터 full pipeline으로 고정하고 재바인딩하지 않는다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import models, preprocessing

# fix②: 8번 원본은 build_windows가 전역 apply_window_preprocessing(기본=atomic)을 부르고
# 10/12가 매 실행마다 이를 full pipeline으로 monkeypatch했다. load_module 시절엔 모듈이 격리돼
# 안전했지만, 패키지에서는 전역 mutation이 프로세스 전체로 새어 실행 순서가 결과를 바꾼다.
# 엔진의 build_windows 사용자는 point.run_case 한 곳뿐이고 늘 full pipeline을 원하므로,
# 여기서 처음부터 full pipeline으로 고정한다(이후 mutation 없음 -> leak 없음).
apply_window_preprocessing = preprocessing.apply_preprocessing_pipeline


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
    target_open = df["target_open"].to_numpy(np.float32)
    target_high = df["target_high"].to_numpy(np.float32)
    target_low = df["target_low"].to_numpy(np.float32)
    target_close = df["target_close"].to_numpy(np.float32)
    timestamps = df["target_timestamp"].astype(str).to_numpy()
    # fix①: 결정시점 = "진입 봉"의 raw timestamp.
    # point 샘플은 y=target_return[end]=log(close[end+1]/close[end]) 이고 prev_close[end]=close[end] 이므로,
    # 진입 봉 = end (close[end]에 진입해 다음 봉까지 보유). 따라서 decision_timestamp = timestamp[end].
    # risk 빌더는 base_close=close[end-1]에서 출발하므로 진입 봉 = end-1 -> timestamp[end-1]을 쓴다.
    # 두 키가 같은 진입 봉을 가리킬 때만 inner-join 되어, 점 신호와 위험 gate가 동일 시점에 짝지어진다.
    decision_timestamps = df["timestamp"].astype(str).to_numpy()

    end_indices = list(range(seq_len, len(df), max(1, stride)))
    if max_windows and len(end_indices) > max_windows:
        end_indices = end_indices[-max_windows:]

    xs, ys, prevs, opens, highs, lows, closes, times = [], [], [], [], [], [], [], []
    decisions: list[str] = []
    for end in end_indices:
        window = values[end - seq_len : end].copy()
        window = apply_window_preprocessing(window, preprocessing)
        window = apply_window_normalization(window, normalization)
        xs.append(window)
        ys.append(targets[end])
        prevs.append(prev_close[end])
        opens.append(target_open[end])
        highs.append(target_high[end])
        lows.append(target_low[end])
        closes.append(target_close[end])
        times.append(timestamps[end])
        decisions.append(decision_timestamps[end])

    x = np.stack(xs).astype(np.float32)
    y = np.asarray(ys, dtype=np.float32)
    prev = np.asarray(prevs, dtype=np.float32)
    open_next = np.asarray(opens, dtype=np.float32)
    high_next = np.asarray(highs, dtype=np.float32)
    low_next = np.asarray(lows, dtype=np.float32)
    close = np.asarray(closes, dtype=np.float32)
    if normalization == "global_standard":
        flat = x.reshape(-1, x.shape[-1])
        mean = flat.mean(axis=0, keepdims=True)
        std = flat.std(axis=0, keepdims=True)
        x = (x - mean.reshape(1, 1, -1)) / np.where(std.reshape(1, 1, -1) < 1e-6, 1.0, std.reshape(1, 1, -1))
    return {
        "x": x,
        "y": y,
        "prev_close": prev,
        "target_open": open_next,
        "target_high": high_next,
        "target_low": low_next,
        "target_close": close,
        "timestamp": np.asarray(times),
        "decision_timestamp": np.asarray(decisions),
    }


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


def build_risk_windows(
    features: pd.DataFrame,
    feature_columns: list[str],
    seq_len: int,
    horizon: int,
    preprocessing: str,
    normalization: str,
    max_windows: int | None,
    stride: int,
) -> dict[str, np.ndarray]:
    values = features[feature_columns].to_numpy(np.float32)
    close = features["close"].to_numpy(np.float64)
    timestamps = features["timestamp"].astype(str).to_numpy()
    end_indices = list(range(seq_len, len(features) - horizon + 1, max(1, stride)))
    if max_windows and len(end_indices) > max_windows:
        end_indices = end_indices[-max_windows:]

    xs: list[np.ndarray] = []
    future_returns: list[float] = []
    realized_volatility: list[float] = []
    downside_scores: list[float] = []
    absolute_move_scores: list[float] = []
    prev_closes: list[float] = []
    target_closes: list[float] = []
    times: list[str] = []
    decisions: list[str] = []

    for end in end_indices:
        window = values[end - seq_len : end].copy()
        window = _apply_preprocessing_pipeline(window, preprocessing)
        window = _apply_window_normalization(window, normalization)
        base_close = close[end - 1]
        path = close[end : end + horizon]
        path_log_returns = np.diff(np.log(np.concatenate([[base_close], path])))
        cumulative_path = np.cumsum(path_log_returns)
        xs.append(window)
        future_returns.append(float(cumulative_path[-1]))
        realized_volatility.append(float(np.sqrt(np.sum(path_log_returns**2))))
        downside_scores.append(float(max(0.0, -np.min(cumulative_path))))
        absolute_move_scores.append(float(np.max(np.abs(cumulative_path))))
        prev_closes.append(float(base_close))
        target_closes.append(float(path[-1]))
        times.append(str(timestamps[end + horizon - 1]))
        # fix①: 위험 윈도우의 진입/base 봉 = end-1 (base_close=close[end-1]에서 horizon 전개).
        # 그래서 decision_timestamp = timestamp[end-1]. point는 진입 봉 end -> timestamp[end]을 쓰므로,
        # 두 키가 같은 진입 봉 close[T]를 가리킬 때 join된다(point end_p == risk end_r-1).
        # 기존 "timestamp"는 horizon 끝(end+horizon-1)이라 정렬키로 쓰면 안 된다.
        decisions.append(str(timestamps[end - 1]))

    return {
        "x": np.stack(xs).astype(np.float32),
        "y": np.asarray(future_returns, dtype=np.float32),
        "future_return": np.asarray(future_returns, dtype=np.float32),
        "realized_volatility": np.asarray(realized_volatility, dtype=np.float32),
        "downside_score": np.asarray(downside_scores, dtype=np.float32),
        "absolute_move_score": np.asarray(absolute_move_scores, dtype=np.float32),
        "prev_close": np.asarray(prev_closes, dtype=np.float32),
        "target_close": np.asarray(target_closes, dtype=np.float32),
        "timestamp": np.asarray(times),
        "decision_timestamp": np.asarray(decisions),
    }


def prepare_event_splits(
    data: dict[str, np.ndarray],
    train_ratio: float,
    val_ratio: float,
    event_kind: str,
    event_quantile: float,
) -> tuple[dict[str, dict[str, np.ndarray]], float]:
    splits = time_split(data, train_ratio, val_ratio)
    score_name = f"{event_kind}_score"
    threshold = float(np.quantile(splits["train"][score_name], event_quantile))
    for split in splits.values():
        split["event_score"] = split[score_name].copy()
        split["y"] = (split[score_name] >= threshold).astype(np.float32)
    return splits, threshold


# build_windows의 `apply_window_normalization` 호출과, build_risk_windows가 11번에서
# 직접 부르던 `base.apply_window_normalization` / `diag.apply_preprocessing_pipeline`을
# 같은 엔진 함수로 연결한다.
apply_window_normalization = models.apply_window_normalization
_apply_window_normalization = models.apply_window_normalization
_apply_preprocessing_pipeline = preprocessing.apply_preprocessing_pipeline
