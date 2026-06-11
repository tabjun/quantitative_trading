"""Optimization diagnostics for research-time training curve analysis.

This module is designed for `/test/models` notebooks that compare
architecture and objective-function behavior without performing local heavy
research runs by default. Actual long runs should happen on the approved
remote/server environment.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from datetime import datetime
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

FEATURE_SETS = {
    "market_only": MARKET_FEATURE_COLUMNS,
    "text_aware": MARKET_FEATURE_COLUMNS + TEXT_FEATURE_COLUMNS,
}


@dataclass(frozen=True)
class ExperimentConfig:
    db_path: str = "data/upbit_data.db"
    price_table: str = "btc_15m_advance"
    output_dir: str = "test/results"
    ticker: str | None = None
    feature_set: str = "text_aware"
    suite: str = "objective_probe"
    seq_len: int = 64
    max_rows: int | None = 5000
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    epochs: int = 8
    batch_size: int = 128
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    hidden_dim: int = 64
    n_heads: int = 4
    grad_clip_norm: float = 1.0
    seed: int = 42
    device: str = "auto"


@dataclass(frozen=True)
class CaseSpec:
    name: str
    suite: str
    algorithm: str
    target_mode: str
    objective_mode: str
    description: str


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
        name="lstm_return_vol_weighted",
        suite="objective_probe",
        algorithm="lstm",
        target_mode="next_log_return",
        objective_mode="vol_weighted_mse",
        description="Weights larger moves more heavily using realized volatility.",
    ),
    CaseSpec(
        name="lstm_return_directional_hybrid",
        suite="objective_probe",
        algorithm="lstm",
        target_mode="next_log_return",
        objective_mode="directional_hybrid",
        description="Regression plus sign-consistency penalty to reduce zero-return collapse.",
    ),
    CaseSpec(
        name="lstm_return_tail_focus",
        suite="objective_probe",
        algorithm="lstm",
        target_mode="next_log_return",
        objective_mode="tail_focus",
        description="Upweights turning points and large absolute returns.",
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
    for algorithm in ("linear", "lstm", "gru", "tcn", "transformer")
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
    for algorithm in ("linear", "lstm", "gru", "tcn", "transformer")
    for objective_mode in ("level_mse", "huber", "vol_weighted_mse", "directional_hybrid")
]

SUITE_CASES = {
    "objective_probe": OBJECTIVE_PROBE_CASES,
    "architecture_probe": ARCHITECTURE_PROBE_CASES,
    "full_matrix": FULL_MATRIX_CASES,
}


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
        if not table_exists(con, config.price_table):
            raise RuntimeError(f"Price table not found in DuckDB: {config.price_table}")

        table_columns = {row[1] for row in con.execute(f"PRAGMA table_info('{config.price_table}')").fetchall()}
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
            FROM {config.price_table}
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

    for end_idx in range(config.seq_len - 1, len(df)):
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
    train_loader = DataLoader(datasets["train"], batch_size=config.batch_size, shuffle=False)
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
        "It is a training-curve diagnostic suite for checking whether the objective function and model family drift into an easy solution such as near-zero return prediction or lag-1 price copying.",
        "",
        "## Configuration",
        "",
        f"- Suite: `{config.suite}`",
        f"- Feature set: `{config.feature_set}`",
        f"- Feature count: `{len(feature_columns)}`",
        f"- Sequence length: `{config.seq_len}`",
        f"- Epochs: `{config.epochs}`",
        f"- Max rows: `{config.max_rows}`",
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
            "For non-stationary crypto forecasting, the most useful checks are whether the objective maintains non-trivial predictive variance and whether it beats naive persistence without collapsing into near-zero returns.",
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
    parser.add_argument("--feature-set", choices=sorted(FEATURE_SETS.keys()), default="text_aware")
    parser.add_argument("--suite", choices=sorted(SUITE_CASES.keys()), default="objective_probe")
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--max-rows", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
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
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        n_heads=args.n_heads,
        grad_clip_norm=args.grad_clip_norm,
        seed=args.seed,
        device=args.device,
    )


def main(argv: Iterable[str] | None = None) -> None:
    config = parse_args(argv)
    artifacts = run_suite(config)
    print("Optimization diagnostics finished.")
    for name, path in artifacts.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
