"""데이터 로딩, 컬럼 정규화, 기초 통계, feature 생성.

8번 실험(`8_optimization_breadth_training_test.py`)에서 그대로 옮겼다.
`resolve_db_path`/`load_price_data`/`normalize_columns`/`make_features`/
`basic_statistics`와 보조 helper `rolling_z`를 담는다.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


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


def make_features(raw: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(raw)
    open_price = df["open"].astype(float) if "open" in df.columns else df["close"].astype(float)
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
    df["target_open"] = open_price.shift(-1)
    df["target_high"] = high.shift(-1)
    df["target_low"] = low.shift(-1)
    df["target_close"] = close.shift(-1)
    df["target_timestamp"] = df["timestamp"].shift(-1)
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
