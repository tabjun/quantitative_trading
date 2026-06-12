# [FOR COMMIT TRACKING ONLY - DO NOT EXECUTE]
# This file is automatically mirrored from the corresponding .ipynb for git diff purposes.
# Actual research execution should be performed in the Jupyter Notebook (.ipynb)
# or in an approved remote/server environment.

# %% [markdown]
# # 5. 최적화 경로 진단 실험
#
# [연구용 스크립트 - Codex 로컬 세션에서 자동 실행 금지]
# 이번 실험은 성능 리더보드가 아니라, 비정상 금융 시계열에서 어떤 objective / head / architecture 조합이 쉬운 해(0 수익률, lag-1 복사)에 붕괴하는지를 빠르게 확인하기 위한 경량 진단용 pass입니다.
#
# 핵심 질문:
# - raw next-close 회귀가 실제로 copy-risk를 키우는가?
# - return target만으로 충분한가, 아니면 Huber / 방향성 penalty / volatility weighting이 필요한가?
# - 같은 objective를 두었을 때 Linear / LSTM / GRU 중 어느 쪽이 더 안정적인가?
#
# 기본 실행 예시:
# - `uv run test/models/5_optimization_diagnostics_test.py`
# - `uv run test/models/5_optimization_diagnostics_test.py --suite objective_probe`
# - `uv run test/models/5_optimization_diagnostics_test.py --suite architecture_probe --max-windows 768 --epochs 6`
# - `uv run test/models/5_optimization_diagnostics_test.py --suite full_matrix --feature-set market_only --max-rows 4000`
#
# 산출물:
# - epoch curve CSV
# - case summary CSV
# - learning curve figure PNG
# - collapse 진단용 Markdown report
#
# 읽는 기준:
# - `collapse_score`가 낮을수록 좋음
# - `variance_ratio`가 0에 가까우면 flat prediction 붕괴 위험
# - `zero_share`가 높으면 0-return shortcut 위험
# - `persistence_gap`이 계속 양수이면 naive copy보다도 못한 상태
#
#
#

# %%
from __future__ import annotations

import sys
from pathlib import Path


def bootstrap_repo_root() -> Path:
    """Notebook-first bootstrap so remote kernels can import repo-local modules."""
    candidates = [Path.cwd(), *Path.cwd().parents]
    try:
        current_file = Path(__file__).resolve()
        candidates.extend([current_file.parent, *current_file.parents])
    except NameError:
        pass

    for candidate in candidates:
        if (candidate / "database" / "paths.py").exists():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            return candidate
    raise ModuleNotFoundError(
        "Could not locate the repository root containing 'database/paths.py'. "
        "Open the notebook from inside the quantitative_trading repository or add the repo root to PYTHONPATH."
    )


REPO_ROOT = bootstrap_repo_root()
print(f"[env-check] repo root: {REPO_ROOT}")
print(f"[env-check] python path head: {sys.path[0]}")

# %%
"""Optimization diagnostics for research-time training curve analysis.

This module is designed for `/test/models` notebooks that compare
architecture and objective-function behavior without performing local heavy
research runs by default. Actual long runs should happen on the approved
remote/server environment.
"""

import argparse
import math
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def ensure_repo_root_on_path() -> Path:
    """Allow notebook/server kernels to import repo-local packages reliably."""
    if "REPO_ROOT" in globals():
        candidate = Path(REPO_ROOT)
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)
        return candidate

    candidates = [Path.cwd(), *Path.cwd().parents]
    try:
        current_file = Path(__file__).resolve()
        candidates.extend([current_file.parent, *current_file.parents])
    except NameError:
        pass

    for candidate in candidates:
        if (candidate / "database" / "paths.py").exists():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            return candidate
    raise ModuleNotFoundError(
        "Could not locate the repository root containing 'database/paths.py'. "
        "Run the notebook from inside the quantitative_trading repository or add the repo root to PYTHONPATH."
    )


REPO_ROOT = ensure_repo_root_on_path()

from database.paths import resolve_db_path


TEXT_FEATURE_COLUMNS = [
    "text_event_count",
    "text_sentiment_mean",
    "text_sentiment_sum",
    "text_sentiment_abs_mean",
    "text_positive_hits",
    "text_negative_hits",
    "text_macro_count",
    "text_risk_count",
    "text_crypto_count",
    "text_regulation_count",
    "text_liquidity_count",
    "text_news_count",
    "text_report_count",
    "text_sns_count",
    "text_shock_z",
    "text_sentiment_momentum_1h",
]

MARKET_FEATURE_COLUMNS = [
    "log_return_1",
    "return_4",
    "return_16",
    "realized_vol_16",
    "hl_range_pct",
    "gk_volatility",
    "rsi_14",
    "roc_12",
    "bb_width_20",
    "sma_spread_5_20",
    "breakout_distance",
    "volume_z_96",
    "value_z_96",
    "amihud_illiq",
    "turnover_proxy",
    "spread_proxy",
]

OPTIMIZATION_PROBE_FEATURE_COLUMNS = [
    "log_return_1",
    "return_4",
    "realized_vol_16",
    "hl_range_pct",
    "volume_z_96",
    "spread_proxy",
]

FEATURE_SETS = {
    "optimization_probe": OPTIMIZATION_PROBE_FEATURE_COLUMNS,
    "market_only": MARKET_FEATURE_COLUMNS,
    "text_aware": MARKET_FEATURE_COLUMNS + TEXT_FEATURE_COLUMNS,
}


@dataclass(frozen=True)
class ExperimentConfig:
    db_path: str = "data/upbit_data.db"
    price_table: str = "btc_15m_advance"
    output_dir: str = "test/results"
    ticker: str | None = None
    feature_set: str = "optimization_probe"
    suite: str = "quick_probe"
    seq_len: int = 32
    max_rows: int | None = 2500
    max_windows: int | None = 512
    window_stride: int = 4
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    epochs: int = 5
    batch_size: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    hidden_dim: int = 32
    n_heads: int = 2
    grad_clip_norm: float = 1.0
    seed: int = 42
    device: str = "auto"
    bootstrap_if_missing: bool = True
    bootstrap_ticker: str = "KRW-BTC"
    bootstrap_days: int = 180


@dataclass(frozen=True)
class CaseSpec:
    name: str
    suite: str
    algorithm: str
    target_mode: str
    objective_mode: str
    description: str


QUICK_PROBE_CASES = [
    CaseSpec(
        name="lstm_level_mse",
        suite="quick_probe",
        algorithm="lstm",
        target_mode="next_close_level",
        objective_mode="mse",
        description="Direct next-close regression as the fastest copy-risk control case.",
    ),
    CaseSpec(
        name="lstm_return_huber",
        suite="quick_probe",
        algorithm="lstm",
        target_mode="next_log_return",
        objective_mode="huber",
        description="Compact robust-return probe for a fast collapse check.",
    ),
    CaseSpec(
        name="lstm_return_directional_hybrid",
        suite="quick_probe",
        algorithm="lstm",
        target_mode="next_log_return",
        objective_mode="directional_hybrid",
        description="Return regression plus direction penalty to test shortcut suppression quickly.",
    ),
]

OBJECTIVE_PROBE_CASES = [
    CaseSpec(
        name="lstm_level_mse",
        suite="objective_probe",
        algorithm="lstm",
        target_mode="next_close_level",
        objective_mode="mse",
        description="Direct next-close regression. Useful as the copy-risk control case.",
    ),
    CaseSpec(
        name="lstm_return_mse",
        suite="objective_probe",
        algorithm="lstm",
        target_mode="next_log_return",
        objective_mode="mse",
        description="One-step log-return regression with plain MSE.",
    ),
    CaseSpec(
        name="lstm_return_huber",
        suite="objective_probe",
        algorithm="lstm",
        target_mode="next_log_return",
        objective_mode="huber",
        description="Robust return regression using SmoothL1/Huber loss.",
    ),
    CaseSpec(
        name="lstm_return_directional_hybrid",
        suite="objective_probe",
        algorithm="lstm",
        target_mode="next_log_return",
        objective_mode="directional_hybrid",
        description="Regression plus sign-consistency penalty to reduce zero-return collapse.",
    ),
]

ARCHITECTURE_PROBE_CASES = [
    CaseSpec(
        name=f"{algorithm}_directional_hybrid",
        suite="architecture_probe",
        algorithm=algorithm,
        target_mode="next_log_return",
        objective_mode="directional_hybrid",
        description="Shared objective to isolate architectural differences in optimization behavior.",
    )
    for algorithm in ("linear", "lstm", "gru")
]

FULL_MATRIX_CASES = [
    CaseSpec(
        name=f"{algorithm}_{objective_mode}",
        suite="full_matrix",
        algorithm=algorithm,
        target_mode="next_log_return" if objective_mode != "level_mse" else "next_close_level",
        objective_mode="mse" if objective_mode == "level_mse" else objective_mode,
        description="Crossed architecture/objective probe.",
    )
    for algorithm in ("linear", "lstm", "gru")
    for objective_mode in ("level_mse", "huber", "directional_hybrid")
]

SUITE_CASES = {
    "quick_probe": QUICK_PROBE_CASES,
    "objective_probe": OBJECTIVE_PROBE_CASES,
    "architecture_probe": ARCHITECTURE_PROBE_CASES,
    "full_matrix": FULL_MATRIX_CASES,
}

PRICE_TABLE_CANDIDATES = [
    "btc_15m_advance",
    "upbit_krw_candle",
    "btc_15m",
]


def compatible_num_heads(hidden_dim: int, requested: int) -> int:
    for candidate in range(min(hidden_dim, requested), 0, -1):
        if hidden_dim % candidate == 0:
            return candidate
    return 1


class SequenceDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        close_now: np.ndarray,
        close_next: np.ndarray,
        return_next: np.ndarray,
        volatility_next: np.ndarray,
        direction_next: np.ndarray,
        seq_len: int,
    ) -> None:
        self.features = torch.tensor(features, dtype=torch.float32)
        self.close_now = torch.tensor(close_now, dtype=torch.float32)
        self.close_next = torch.tensor(close_next, dtype=torch.float32)
        self.return_next = torch.tensor(return_next, dtype=torch.float32)
        self.volatility_next = torch.tensor(volatility_next, dtype=torch.float32)
        self.direction_next = torch.tensor(direction_next, dtype=torch.float32)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "x": self.features[index],
            "close_now": self.close_now[index],
            "close_next": self.close_next[index],
            "return_next": self.return_next[index],
            "volatility_next": self.volatility_next[index],
            "direction_next": self.direction_next[index],
        }


class LinearProbe(nn.Module):
    def __init__(self, input_dim: int, seq_len: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim * seq_len, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RecurrentProbe(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, kind: str) -> None:
        super().__init__()
        rnn_cls = nn.LSTM if kind == "lstm" else nn.GRU
        self.rnn = rnn_cls(input_dim, hidden_dim, batch_first=True, num_layers=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, hidden = self.rnn(x)
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        return hidden[-1]


class TCNProbe(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=2, dilation=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=4, dilation=2),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=8, dilation=4),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x.transpose(1, 2))
        return y[..., -1]


class TransformerProbe(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_heads: int) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=compatible_num_heads(hidden_dim, n_heads),
            dim_feedforward=hidden_dim * 2,
            batch_first=True,
            dropout=0.0,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.input_proj(x)
        y = self.encoder(y)
        return self.norm(y[:, -1, :])


class ForecastHead(nn.Module):
    def __init__(self, hidden_dim: int, directional: bool) -> None:
        super().__init__()
        self.value_head = nn.Linear(hidden_dim, 1)
        self.directional = directional
        self.direction_head = nn.Linear(hidden_dim, 1) if directional else None

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        out = {"value": self.value_head(z).squeeze(-1)}
        if self.directional and self.direction_head is not None:
            out["direction_logit"] = self.direction_head(z).squeeze(-1)
        return out


class ForecastModel(nn.Module):
    def __init__(self, config: ExperimentConfig, algorithm: str, input_dim: int) -> None:
        super().__init__()
        hidden_dim = config.hidden_dim
        if algorithm == "linear":
            self.backbone = LinearProbe(input_dim=input_dim, seq_len=config.seq_len, hidden_dim=hidden_dim)
        elif algorithm == "lstm":
            self.backbone = RecurrentProbe(input_dim=input_dim, hidden_dim=hidden_dim, kind="lstm")
        elif algorithm == "gru":
            self.backbone = RecurrentProbe(input_dim=input_dim, hidden_dim=hidden_dim, kind="gru")
        elif algorithm == "tcn":
            self.backbone = TCNProbe(input_dim=input_dim, hidden_dim=hidden_dim)
        elif algorithm == "transformer":
            self.backbone = TransformerProbe(input_dim=input_dim, hidden_dim=hidden_dim, n_heads=config.n_heads)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        self.head = ForecastHead(hidden_dim=hidden_dim, directional=True)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.backbone(x)
        return self.head(z)


def table_exists(con: duckdb.DuckDBPyConnection, table_name: str) -> bool:
    return bool(
        con.execute(
            """
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_name = ?
            """,
            [table_name],
        ).fetchone()[0]
    )


def list_main_tables(con: duckdb.DuckDBPyConnection) -> list[str]:
    rows = con.execute(
        """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'main'
        ORDER BY table_name
        """
    ).fetchall()
    return [row[0] for row in rows]


def resolve_price_table(con: duckdb.DuckDBPyConnection, requested: str) -> str:
    if table_exists(con, requested):
        return requested

    available = list_main_tables(con)
    for candidate in PRICE_TABLE_CANDIDATES:
        if candidate in available:
            print(
                f"[table-resolve] Requested price table '{requested}' was not found. "
                f"Using fallback table '{candidate}'."
            )
            return candidate

    raise RuntimeError(
        "Price table not found in DuckDB. "
        f"requested='{requested}', available_tables={available}"
    )


def bootstrap_price_table_if_missing(config: ExperimentConfig) -> str | None:
    if not config.bootstrap_if_missing:
        return None

    try:
        import pyupbit
    except ImportError as exc:
        raise RuntimeError(
            "No reusable price table exists in DuckDB and pyupbit is not available for bootstrap."
        ) from exc

    db_path = Path(config.db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        "[table-bootstrap] No reusable price table was found. "
        f"Building '{config.price_table}' from {config.bootstrap_ticker} 15-minute candles "
        f"for the last {config.bootstrap_days} days."
    )

    df_list: list[pd.DataFrame] = []
    end_date = datetime.now()
    start_date = end_date - timedelta(days=config.bootstrap_days)
    current_to = end_date

    while current_to > start_date:
        df = pyupbit.get_ohlcv(
            config.bootstrap_ticker,
            interval="minute15",
            to=current_to.strftime("%Y-%m-%d %H:%M:%S"),
            count=200,
        )
        if df is None or df.empty:
            break
        df_list.append(df)
        current_to = df.index[0]
        time.sleep(0.12)

    if not df_list:
        raise RuntimeError(
            "Price table bootstrap failed because pyupbit returned no candle data."
        )

    full_df = pd.concat(df_list).sort_index()
    full_df = full_df[~full_df.index.duplicated(keep="first")]
    full_df = full_df.loc[full_df.index >= pd.Timestamp(start_date)].copy()
    full_df.reset_index(inplace=True)
    full_df.rename(columns={"index": "timestamp"}, inplace=True)
    full_df["ticker"] = config.bootstrap_ticker
    if "value" not in full_df.columns:
        full_df["value"] = full_df["close"] * full_df["volume"]

    with duckdb.connect(config.db_path) as con:
        con.register("bootstrap_df", full_df)
        con.execute(f"CREATE OR REPLACE TABLE {config.price_table} AS SELECT * FROM bootstrap_df")

    print(
        f"[table-bootstrap] Created '{config.price_table}' with {len(full_df)} rows "
        f"at {config.db_path}."
    )
    return config.price_table


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    rolling_mean = series.rolling(window).mean()
    rolling_std = series.rolling(window).std().replace(0, np.nan)
    return (series - rolling_mean) / rolling_std


def garman_klass_volatility(df: pd.DataFrame) -> pd.Series:
    log_hl = np.log(df["high"] / df["low"]).replace([np.inf, -np.inf], np.nan)
    log_co = np.log(df["close"] / df["open"]).replace([np.inf, -np.inf], np.nan)
    return np.sqrt(0.5 * log_hl.pow(2) - (2 * np.log(2) - 1) * log_co.pow(2))


def load_price_with_text_features(config: ExperimentConfig) -> pd.DataFrame:
    limit_sql = f"LIMIT {int(config.max_rows)}" if config.max_rows else ""
    with duckdb.connect(config.db_path) as con:
        try:
            price_table = resolve_price_table(con, config.price_table)
        except RuntimeError:
            con.close()
            bootstrapped_table = bootstrap_price_table_if_missing(config)
            with duckdb.connect(config.db_path) as rebuilt_con:
                price_table = resolve_price_table(rebuilt_con, bootstrapped_table or config.price_table)
                table_columns = {row[1] for row in rebuilt_con.execute(f"PRAGMA table_info('{price_table}')").fetchall()}
                ticker_expr = "ticker" if "ticker" in table_columns else "'KRW-BTC' AS ticker"
                value_expr = "value" if "value" in table_columns else "close * volume AS value"

                params: list[object] = []
                where_sql = ""
                if config.ticker:
                    if "ticker" in table_columns:
                        where_sql = "WHERE ticker = ?"
                        params.append(config.ticker)
                    else:
                        where_sql = "WHERE 'KRW-BTC' = ?"
                        params.append(config.ticker)

                price_df = rebuilt_con.execute(
                    f"""
                    SELECT timestamp, open, high, low, close, volume, {value_expr}, {ticker_expr}
                    FROM {price_table}
                    {where_sql}
                    ORDER BY timestamp DESC
                    {limit_sql}
                    """,
                    params,
                ).df()

                text_exists = table_exists(rebuilt_con, "text_features_15m")
                if text_exists:
                    text_df = rebuilt_con.execute(
                        f"""
                        SELECT timestamp, {", ".join(TEXT_FEATURE_COLUMNS)}
                        FROM text_features_15m
                        ORDER BY timestamp
                        """
                    ).df()
                else:
                    text_df = pd.DataFrame(columns=["timestamp", *TEXT_FEATURE_COLUMNS])

            if price_df.empty:
                raise RuntimeError("No rows loaded after table bootstrap. Check ticker or bootstrap window.")

            price_df = price_df.sort_values("timestamp").reset_index(drop=True)
            price_df["timestamp"] = pd.to_datetime(price_df["timestamp"]).dt.floor("15min")
            text_df["timestamp"] = pd.to_datetime(text_df["timestamp"]).dt.floor("15min")
            merged = price_df.merge(text_df, on="timestamp", how="left")
            for column in TEXT_FEATURE_COLUMNS:
                if column not in merged.columns:
                    merged[column] = 0.0
            merged[TEXT_FEATURE_COLUMNS] = merged[TEXT_FEATURE_COLUMNS].fillna(0.0)
            return merged.sort_values(["ticker", "timestamp"]).reset_index(drop=True)

        table_columns = {row[1] for row in con.execute(f"PRAGMA table_info('{price_table}')").fetchall()}
        ticker_expr = "ticker" if "ticker" in table_columns else "'KRW-BTC' AS ticker"
        value_expr = "value" if "value" in table_columns else "close * volume AS value"

        params: list[object] = []
        where_sql = ""
        if config.ticker:
            if "ticker" in table_columns:
                where_sql = "WHERE ticker = ?"
                params.append(config.ticker)
            else:
                where_sql = "WHERE 'KRW-BTC' = ?"
                params.append(config.ticker)

        price_df = con.execute(
            f"""
            SELECT timestamp, open, high, low, close, volume, {value_expr}, {ticker_expr}
            FROM {price_table}
            {where_sql}
            ORDER BY timestamp DESC
            {limit_sql}
            """,
            params,
        ).df()

        text_exists = table_exists(con, "text_features_15m")
        if text_exists:
            text_df = con.execute(
                f"""
                SELECT timestamp, {", ".join(TEXT_FEATURE_COLUMNS)}
                FROM text_features_15m
                ORDER BY timestamp
                """
            ).df()
        else:
            text_df = pd.DataFrame(columns=["timestamp", *TEXT_FEATURE_COLUMNS])

    if price_df.empty:
        raise RuntimeError("No rows loaded. Check table, ticker, or max_rows.")

    price_df = price_df.sort_values("timestamp").reset_index(drop=True)
    price_df["timestamp"] = pd.to_datetime(price_df["timestamp"]).dt.floor("15min")
    text_df["timestamp"] = pd.to_datetime(text_df["timestamp"]).dt.floor("15min")
    merged = price_df.merge(text_df, on="timestamp", how="left")
    for column in TEXT_FEATURE_COLUMNS:
        if column not in merged.columns:
            merged[column] = 0.0
    merged[TEXT_FEATURE_COLUMNS] = merged[TEXT_FEATURE_COLUMNS].fillna(0.0)
    return merged.sort_values(["ticker", "timestamp"]).reset_index(drop=True)


def add_market_features(df: pd.DataFrame) -> pd.DataFrame:
    out_frames: list[pd.DataFrame] = []
    for ticker, ticker_df in df.groupby("ticker", sort=False):
        tdf = ticker_df.sort_values("timestamp").copy()
        close = tdf["close"].replace(0, np.nan)
        value = tdf["value"].replace(0, np.nan)
        volume = tdf["volume"].replace(0, np.nan)

        tdf["log_return_1"] = np.log(close).diff()
        tdf["return_4"] = close.pct_change(4)
        tdf["return_16"] = close.pct_change(16)
        tdf["realized_vol_16"] = tdf["log_return_1"].rolling(16).std()
        tdf["hl_range_pct"] = (tdf["high"] - tdf["low"]) / close
        tdf["gk_volatility"] = garman_klass_volatility(tdf)

        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        tdf["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))
        tdf["roc_12"] = close.pct_change(12)

        sma_5 = close.rolling(5).mean()
        sma_20 = close.rolling(20).mean()
        std_20 = close.rolling(20).std()
        tdf["bb_width_20"] = ((sma_20 + 2 * std_20) - (sma_20 - 2 * std_20)) / close
        tdf["sma_spread_5_20"] = (sma_5 - sma_20) / close
        rolling_high_96 = tdf["high"].rolling(96).max()
        tdf["breakout_distance"] = (close - rolling_high_96) / close

        tdf["volume_z_96"] = rolling_zscore(volume, 96)
        tdf["value_z_96"] = rolling_zscore(value, 96)
        tdf["amihud_illiq"] = close.pct_change().abs() / value.replace(0, np.nan)
        tdf["turnover_proxy"] = value / value.rolling(96).mean()
        tdf["spread_proxy"] = (tdf["high"] - tdf["low"]) / close
        out_frames.append(tdf)
    return pd.concat(out_frames, axis=0).sort_values(["ticker", "timestamp"]).reset_index(drop=True)


def prepare_feature_frame(config: ExperimentConfig) -> pd.DataFrame:
    df = load_price_with_text_features(config)
    df = add_market_features(df)
    feature_columns = FEATURE_SETS[config.feature_set]
    required_columns = ["timestamp", "ticker", "close", *feature_columns]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    df["next_close"] = df.groupby("ticker")["close"].shift(-1)
    df["next_log_return"] = np.log(df["next_close"] / df["close"])
    df["next_direction"] = (df["next_log_return"] > 0).astype(float)
    df["next_volatility"] = df.groupby("ticker")["realized_vol_16"].shift(-1)

    clean_columns = ["close", "next_close", "next_log_return", "next_direction", "next_volatility", *feature_columns]
    df = df.dropna(subset=clean_columns).reset_index(drop=True)
    return df


def standardize_train_applied(train: np.ndarray, other: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return (train - mean) / std, (other - mean) / std


def build_splits(config: ExperimentConfig, df: pd.DataFrame) -> tuple[dict[str, SequenceDataset], list[str]]:
    feature_columns = FEATURE_SETS[config.feature_set]
    features = df[feature_columns].to_numpy(dtype=np.float32)
    close = df["close"].to_numpy(dtype=np.float32)
    next_close = df["next_close"].to_numpy(dtype=np.float32)
    next_return = df["next_log_return"].to_numpy(dtype=np.float32)
    next_vol = df["next_volatility"].to_numpy(dtype=np.float32)
    next_dir = df["next_direction"].to_numpy(dtype=np.float32)

    sequences: list[np.ndarray] = []
    close_now_list: list[float] = []
    close_next_list: list[float] = []
    return_next_list: list[float] = []
    vol_next_list: list[float] = []
    dir_next_list: list[float] = []

    for end_idx in range(config.seq_len - 1, len(df), max(1, config.window_stride)):
        start_idx = end_idx - config.seq_len + 1
        sequences.append(features[start_idx : end_idx + 1])
        close_now_list.append(close[end_idx])
        close_next_list.append(next_close[end_idx])
        return_next_list.append(next_return[end_idx])
        vol_next_list.append(next_vol[end_idx])
        dir_next_list.append(next_dir[end_idx])

    X = np.stack(sequences)
    close_now_arr = np.asarray(close_now_list, dtype=np.float32)
    close_next_arr = np.asarray(close_next_list, dtype=np.float32)
    return_next_arr = np.asarray(return_next_list, dtype=np.float32)
    vol_next_arr = np.asarray(vol_next_list, dtype=np.float32)
    dir_next_arr = np.asarray(dir_next_list, dtype=np.float32)

    if config.max_windows and len(X) > config.max_windows:
        sampled_idx = np.linspace(0, len(X) - 1, num=config.max_windows, dtype=int)
        X = X[sampled_idx]
        close_now_arr = close_now_arr[sampled_idx]
        close_next_arr = close_next_arr[sampled_idx]
        return_next_arr = return_next_arr[sampled_idx]
        vol_next_arr = vol_next_arr[sampled_idx]
        dir_next_arr = dir_next_arr[sampled_idx]

    total = len(X)
    if total < 64:
        raise RuntimeError(
            f"Not enough sequence windows for diagnostics: {total}. Increase max_rows or reduce seq_len."
        )
    train_end = max(config.seq_len + 10, int(total * config.train_ratio))
    val_end = max(train_end + 10, int(total * (config.train_ratio + config.val_ratio)))
    val_end = min(val_end, total - 1)
    if train_end >= val_end or val_end >= total:
        raise RuntimeError(
            f"Invalid split sizes derived from {total} windows. Adjust train_ratio/val_ratio/max_rows."
        )

    X_train = X[:train_end]
    X_val = X[train_end:val_end]
    X_test = X[val_end:]
    X_train_scaled, X_val_scaled = standardize_train_applied(X_train, X_val)
    _, X_test_scaled = standardize_train_applied(X_train, X_test)

    splits = {
        "train": SequenceDataset(
            features=X_train_scaled,
            close_now=close_now_arr[:train_end],
            close_next=close_next_arr[:train_end],
            return_next=return_next_arr[:train_end],
            volatility_next=vol_next_arr[:train_end],
            direction_next=dir_next_arr[:train_end],
            seq_len=config.seq_len,
        ),
        "val": SequenceDataset(
            features=X_val_scaled,
            close_now=close_now_arr[train_end:val_end],
            close_next=close_next_arr[train_end:val_end],
            return_next=return_next_arr[train_end:val_end],
            volatility_next=vol_next_arr[train_end:val_end],
            direction_next=dir_next_arr[train_end:val_end],
            seq_len=config.seq_len,
        ),
        "test": SequenceDataset(
            features=X_test_scaled,
            close_now=close_now_arr[val_end:],
            close_next=close_next_arr[val_end:],
            return_next=return_next_arr[val_end:],
            volatility_next=vol_next_arr[val_end:],
            direction_next=dir_next_arr[val_end:],
            seq_len=config.seq_len,
        ),
    }
    return splits, feature_columns


def resolve_device(config: ExperimentConfig) -> torch.device:
    if config.device != "auto":
        return torch.device(config.device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_case_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    case: CaseSpec,
) -> torch.Tensor:
    pred_value = outputs["value"]
    direction_logit = outputs["direction_logit"]
    if case.target_mode == "next_close_level":
        target = batch["close_next"]
    else:
        target = batch["return_next"]

    if case.objective_mode == "mse":
        return F.mse_loss(pred_value, target)
    if case.objective_mode == "huber":
        return F.smooth_l1_loss(pred_value, target, beta=0.02)
    if case.objective_mode == "vol_weighted_mse":
        weights = 1.0 + torch.nan_to_num(batch["volatility_next"], nan=0.0).abs() * 10.0
        return torch.mean(weights * torch.square(pred_value - target))
    if case.objective_mode == "directional_hybrid":
        reg = F.smooth_l1_loss(pred_value, target, beta=0.02)
        cls = F.binary_cross_entropy_with_logits(direction_logit, batch["direction_next"])
        return reg + 0.25 * cls
    if case.objective_mode == "tail_focus":
        weights = 1.0 + torch.abs(batch["return_next"]) * 50.0
        reg = torch.mean(weights * torch.square(pred_value - target))
        cls = F.binary_cross_entropy_with_logits(direction_logit, batch["direction_next"])
        return reg + 0.20 * cls
    raise ValueError(f"Unsupported objective mode: {case.objective_mode}")


def infer_close_prediction(case: CaseSpec, batch: dict[str, torch.Tensor], pred_value: torch.Tensor) -> torch.Tensor:
    if case.target_mode == "next_close_level":
        return pred_value
    return batch["close_now"] * torch.exp(pred_value)


def infer_delta_prediction(case: CaseSpec, batch: dict[str, torch.Tensor], pred_value: torch.Tensor) -> torch.Tensor:
    pred_close = infer_close_prediction(case, batch, pred_value)
    return pred_close - batch["close_now"]


def evaluate_epoch(
    model: ForecastModel,
    loader: DataLoader,
    case: CaseSpec,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    pred_close_all: list[np.ndarray] = []
    true_close_all: list[np.ndarray] = []
    current_close_all: list[np.ndarray] = []
    pred_delta_all: list[np.ndarray] = []
    true_delta_all: list[np.ndarray] = []
    pred_return_all: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(batch["x"])
            loss = compute_case_loss(outputs, batch, case)
            losses.append(float(loss.item()))

            pred_close = infer_close_prediction(case, batch, outputs["value"])
            true_close = batch["close_next"]
            pred_delta = pred_close - batch["close_now"]
            true_delta = true_close - batch["close_now"]
            pred_return = torch.log(torch.clamp(pred_close / batch["close_now"], min=1e-8))

            pred_close_all.append(pred_close.detach().cpu().numpy())
            true_close_all.append(true_close.detach().cpu().numpy())
            current_close_all.append(batch["close_now"].detach().cpu().numpy())
            pred_delta_all.append(pred_delta.detach().cpu().numpy())
            true_delta_all.append(true_delta.detach().cpu().numpy())
            pred_return_all.append(pred_return.detach().cpu().numpy())

    pred_close_np = np.concatenate(pred_close_all)
    true_close_np = np.concatenate(true_close_all)
    current_close_np = np.concatenate(current_close_all)
    pred_delta_np = np.concatenate(pred_delta_all)
    true_delta_np = np.concatenate(true_delta_all)
    pred_return_np = np.concatenate(pred_return_all)

    persistence_mae = float(np.mean(np.abs(current_close_np - true_close_np)))
    model_mae = float(np.mean(np.abs(pred_close_np - true_close_np)))
    pred_std = float(np.std(pred_delta_np))
    true_std = float(np.std(true_delta_np))
    variance_ratio = pred_std / true_std if true_std > 1e-8 else math.nan
    zero_share = float(np.mean(np.abs(pred_return_np) < 1e-4))
    sign_agreement = float(np.mean(np.sign(pred_delta_np) == np.sign(true_delta_np)))
    copy_alignment = float(np.corrcoef(pred_close_np, current_close_np)[0, 1]) if len(pred_close_np) > 3 else math.nan
    collapse_score = float(
        0.45 * max(0.0, 1.0 - min(variance_ratio if not math.isnan(variance_ratio) else 0.0, 1.0))
        + 0.35 * zero_share
        + 0.20 * max(0.0, copy_alignment if not math.isnan(copy_alignment) else 0.0)
    )

    return {
        "loss": float(np.mean(losses)) if losses else math.nan,
        "model_mae": model_mae,
        "persistence_mae": persistence_mae,
        "persistence_gap": model_mae - persistence_mae,
        "variance_ratio": variance_ratio,
        "zero_share": zero_share,
        "sign_agreement": sign_agreement,
        "copy_alignment": copy_alignment,
        "collapse_score": collapse_score,
    }


def train_case(
    config: ExperimentConfig,
    case: CaseSpec,
    datasets: dict[str, SequenceDataset],
    input_dim: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    device = resolve_device(config)
    train_loader = DataLoader(datasets["train"], batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(datasets["val"], batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(datasets["test"], batch_size=config.batch_size, shuffle=False)

    model = ForecastModel(config=config, algorithm=case.algorithm, input_dim=input_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    history_rows: list[dict[str, float | int | str]] = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        train_losses: list[float] = []
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch["x"])
            loss = compute_case_loss(outputs, batch, case)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            optimizer.step()
            train_losses.append(float(loss.item()))

        train_diag = evaluate_epoch(model, train_loader, case, device)
        val_diag = evaluate_epoch(model, val_loader, case, device)
        history_rows.append(
            {
                "case_name": case.name,
                "algorithm": case.algorithm,
                "objective_mode": case.objective_mode,
                "target_mode": case.target_mode,
                "epoch": epoch,
                "train_step_loss": float(np.mean(train_losses)) if train_losses else math.nan,
                "train_diag_loss": train_diag["loss"],
                "val_loss": val_diag["loss"],
                "val_persistence_gap": val_diag["persistence_gap"],
                "val_variance_ratio": val_diag["variance_ratio"],
                "val_zero_share": val_diag["zero_share"],
                "val_sign_agreement": val_diag["sign_agreement"],
                "val_copy_alignment": val_diag["copy_alignment"],
                "val_collapse_score": val_diag["collapse_score"],
            }
        )

    test_diag = evaluate_epoch(model, test_loader, case, device)
    summary = {
        "case_name": case.name,
        "algorithm": case.algorithm,
        "objective_mode": case.objective_mode,
        "target_mode": case.target_mode,
        "description": case.description,
        **test_diag,
    }
    return pd.DataFrame(history_rows), summary


def plot_suite_curves(history_df: pd.DataFrame, output_path: Path) -> None:
    if history_df.empty:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    metric_map = [
        ("val_loss", "Validation Loss"),
        ("val_collapse_score", "Collapse Score"),
        ("val_variance_ratio", "Variance Ratio"),
        ("val_zero_share", "Near-Zero Return Share"),
    ]

    for axis, (metric, title) in zip(axes.flat, metric_map):
        for case_name, frame in history_df.groupby("case_name"):
            axis.plot(frame["epoch"], frame[metric], marker="o", label=case_name)
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
    axes[0, 0].legend(loc="best", fontsize=8)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def render_suite_report(
    config: ExperimentConfig,
    history_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    feature_columns: list[str],
    output_path: Path,
    figure_path: Path,
) -> None:
    if summary_df.empty:
        output_path.write_text("# Optimization diagnostics\n\nNo summary rows were produced.\n", encoding="utf-8")
        return

    ranked = summary_df.sort_values(["collapse_score", "persistence_gap", "model_mae"], ascending=[True, True, True])
    top = ranked.iloc[0]
    lines = [
        "# Optimization Diagnostics Report",
        "",
        "## Purpose",
        "",
        "This experiment is not a leaderboard run.",
        "It is a lightweight training-curve diagnostic suite for checking whether the objective function and model family drift into an easy solution such as near-zero return prediction or lag-1 price copying.",
        "The defaults intentionally avoid a wide independent-variable setup so that optimization collapse can be checked quickly before the larger research runs are attempted.",
        "",
        "## Configuration",
        "",
        f"- Suite: `{config.suite}`",
        f"- Feature set: `{config.feature_set}`",
        f"- Feature count: `{len(feature_columns)}`",
        f"- Sequence length: `{config.seq_len}`",
        f"- Epochs: `{config.epochs}`",
        f"- Max rows: `{config.max_rows}`",
        f"- Max windows: `{config.max_windows}`",
        f"- Window stride: `{config.window_stride}`",
        f"- Figure: `{figure_path.as_posix()}`",
        "",
        "## Reading guide",
        "",
        "- `collapse_score`: lower is better. It penalizes low prediction variance, excessive near-zero predictions, and overly strong alignment with the current price level.",
        "- `persistence_gap`: model MAE minus naive persistence MAE on the held-out split. Negative values mean the model improved on the simplest copy baseline.",
        "- `variance_ratio`: predicted delta standard deviation divided by actual delta standard deviation. Values near zero suggest collapse toward flat predictions.",
        "- `zero_share`: fraction of predicted one-step returns whose absolute value is smaller than `1e-4`.",
        "",
        "## Best current diagnostic profile",
        "",
        f"- Case: `{top['case_name']}`",
        f"- Algorithm: `{top['algorithm']}`",
        f"- Objective: `{top['objective_mode']}`",
        f"- Collapse score: `{top['collapse_score']:.4f}`",
        f"- Persistence gap: `{top['persistence_gap']:.6f}`",
        f"- Variance ratio: `{top['variance_ratio']:.4f}`",
        "",
        "## Case summary",
        "",
    ]

    for row in ranked.itertuples(index=False):
        lines.extend(
            [
                f"### {row.case_name}",
                f"- Description: {row.description}",
                f"- Algorithm / objective: `{row.algorithm}` / `{row.objective_mode}`",
                f"- Collapse score: `{row.collapse_score:.4f}`",
                f"- Persistence gap: `{row.persistence_gap:.6f}`",
                f"- Variance ratio: `{row.variance_ratio:.4f}`",
                f"- Near-zero return share: `{row.zero_share:.4f}`",
                f"- Sign agreement: `{row.sign_agreement:.4f}`",
                "",
            ]
        )

    lines.extend(
        [
            "## Recommendation for this research direction",
            "",
            "Start from the case family with the lowest collapse score, not from the lowest raw loss alone.",
            "For non-stationary crypto forecasting, the fastest useful first check is whether the objective maintains non-trivial predictive variance and whether it beats naive persistence without collapsing into near-zero returns.",
            "If the quick probe already collapses, increasing independent variables or widening the data matrix usually adds cost before it adds insight.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_suite(config: ExperimentConfig) -> dict[str, Path]:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    df = prepare_feature_frame(config)
    datasets, feature_columns = build_splits(config, df)
    input_dim = len(feature_columns)
    cases = SUITE_CASES[config.suite]

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    history_frames: list[pd.DataFrame] = []
    summaries: list[dict[str, float | str]] = []
    for case in cases:
        history_df, summary = train_case(config=config, case=case, datasets=datasets, input_dim=input_dim)
        history_frames.append(history_df)
        summaries.append(summary)

    combined_history = pd.concat(history_frames, axis=0, ignore_index=True)
    summary_df = pd.DataFrame(summaries)

    history_path = output_dir / f"5_optimization_diagnostics_{config.suite}_{stamp}_curves.csv"
    summary_path = output_dir / f"5_optimization_diagnostics_{config.suite}_{stamp}_summary.csv"
    figure_path = output_dir / f"5_optimization_diagnostics_{config.suite}_{stamp}.png"
    report_path = output_dir / f"5_optimization_diagnostics_{config.suite}_{stamp}.md"

    combined_history.to_csv(history_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    plot_suite_curves(combined_history, figure_path)
    render_suite_report(config, combined_history, summary_df, feature_columns, report_path, figure_path)
    return {
        "history_csv": history_path,
        "summary_csv": summary_path,
        "figure_png": figure_path,
        "report_md": report_path,
    }


def parse_args(argv: Iterable[str] | None = None) -> ExperimentConfig:
    parser = argparse.ArgumentParser(description="Optimization diagnostics for training-curve analysis.")
    parser.add_argument("--db", default="data/upbit_data.db")
    parser.add_argument("--table", default="btc_15m_advance")
    parser.add_argument("--output-dir", default="test/results")
    parser.add_argument("--ticker", default=None)
    parser.add_argument("--feature-set", choices=sorted(FEATURE_SETS.keys()), default="optimization_probe")
    parser.add_argument("--suite", choices=sorted(SUITE_CASES.keys()), default="quick_probe")
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--max-rows", type=int, default=2500)
    parser.add_argument("--max-windows", type=int, default=512)
    parser.add_argument("--window-stride", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--n-heads", type=int, default=2)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--bootstrap-if-missing", action="store_true", default=True)
    parser.add_argument("--no-bootstrap-if-missing", dest="bootstrap_if_missing", action="store_false")
    parser.add_argument("--bootstrap-ticker", default="KRW-BTC")
    parser.add_argument("--bootstrap-days", type=int, default=180)
    if argv is None and "ipykernel" in sys.modules:
        argv = []
    args = parser.parse_args(list(argv) if argv is not None else None)
    return ExperimentConfig(
        db_path=str(resolve_db_path(args.db)),
        price_table=args.table,
        output_dir=args.output_dir,
        ticker=args.ticker,
        feature_set=args.feature_set,
        suite=args.suite,
        seq_len=args.seq_len,
        max_rows=args.max_rows,
        max_windows=args.max_windows,
        window_stride=args.window_stride,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        n_heads=args.n_heads,
        grad_clip_norm=args.grad_clip_norm,
        seed=args.seed,
        device=args.device,
        bootstrap_if_missing=args.bootstrap_if_missing,
        bootstrap_ticker=args.bootstrap_ticker,
        bootstrap_days=args.bootstrap_days,
    )


def main(argv: Iterable[str] | None = None) -> None:
    config = parse_args(argv)
    artifacts = run_suite(config)
    print("Optimization diagnostics finished.")
    for name, path in artifacts.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
