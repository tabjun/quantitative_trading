# [FOR COMMIT TRACKING ONLY - DO NOT EXECUTE]
# This file is automatically mirrored from the corresponding .ipynb for git diff purposes.
# Actual research execution should be performed in the Jupyter Notebook (.ipynb)
# or in an approved remote/server environment.

# %% [markdown]
# # 4. 텍스트 독립변수 분석
#
# [연구용 스크립트 - Codex 로컬 세션에서 자동 실행 금지]
# Phase 2.2 확인용 실험입니다.
# 가격 기반 기준선과 텍스트 반영 변수셋을 나란히 비교합니다.
#
# - `price_ohlcv_technical_baseline`: 정상성 진단을 통과한 가격·거래량 기술지표만 사용
# - `text_context_independent_variables`: 위 기준선에 뉴스/리포트/SNS 텍스트 변수 추가
#
# 이번 실험에서 보는 변수 예시는 다음과 같습니다.
#
# - 텍스트 변수: `text_event_count`, `text_sentiment_mean`, `text_sentiment_sum`, `text_sentiment_abs_mean`, `text_positive_hits`, `text_negative_hits`, `text_macro_count`, `text_risk_count`, `text_crypto_count`, `text_regulation_count`, `text_liquidity_count`, `text_news_count`, `text_report_count`, `text_sns_count`, `text_shock_z`, `text_sentiment_momentum_1h`
# - 시장 변수: `log_return_1`, `return_4`, `return_16`, `realized_vol_16`, `hl_range_pct`, `gk_volatility`, `rsi_14`, `roc_12`, `bb_width_20`, `sma_spread_5_20`, `breakout_distance`, `volume_z_96`, `value_z_96`, `amihud_illiq`, `turnover_proxy`, `spread_proxy`
# - 가격 상태 후보: `close_level_z`, `log_close_level_z`, `close_diff_z`, `log_return_z`, `price_rolling_z_96`, `ema_diff_z`
#
# 왜 `adaptive_stationarity`를 쓰는가:
# - 원시 가격 수준을 무조건 쓰면 Lag-1 shift 같은 복사 예측으로 끌릴 수 있다.
# - 그래서 ADF/드리프트 검정으로 안전한 가격 상태만 고르고, 필요한 경우에만 수준 변수를 쓴다.
#
# 실행 예시:
# - `uv run test/models/4_text_independent_variable_analysis.py --db data/upbit_data.db --table btc_15m_advance`
# - `uv run test/models/4_text_independent_variable_analysis.py --max-rows 10000 --epochs 3 --seq-len 64`
# - `uv run test/models/4_text_independent_variable_analysis.py --optuna-trials 5 --max-rows 2000`
#
# 실제 수치 산출은 학교 서버나 승인된 원격 환경에서만 수행합니다.
#

# %%
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import duckdb
import numpy as np
import pandas as pd
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

STATIONARY_MARKET_FEATURE_COLUMNS = [
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

PRICE_STATE_CANDIDATE_COLUMNS = [
    "close_level_z",
    "log_close_level_z",
    "close_diff_z",
    "log_return_z",
    "price_rolling_z_96",
    "ema_diff_z",
]

# Default inputs are selected by `resolve_feature_columns`: the code diagnoses
# price-state stationarity on the train split and chooses a safe representation.
BASE_FEATURE_COLUMNS = STATIONARY_MARKET_FEATURE_COLUMNS
TEXT_AWARE_FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + TEXT_FEATURE_COLUMNS

MODEL_SPECS = [
    {
        "name": "LSTMRepresentative",
        "family": "RNN/LSTM",
        "reason": "One compact recurrent baseline from the older 15-model set.",
    },
    {
        "name": "TransformerRepresentative",
        "family": "Transformer encoder",
        "reason": "One compact attention baseline from the older 15-model set.",
    },
    {
        "name": "MambaLite",
        "family": "Mamba/SSM-inspired",
        "reason": "Light gated causal-conv proxy for Mamba-style time-series models such as Mamba4Cast.",
    },
    {
        "name": "TimeXerLite",
        "family": "Exogenous-variable Transformer",
        "reason": "Inspired by TimeXer, NeurIPS 2024, which focuses on forecasting with exogenous variables.",
    },
    {
        "name": "ITransformerLite",
        "family": "Inverted variable-token Transformer",
        "reason": "Inspired by iTransformer, ICLR 2024, useful for multivariate variables as tokens.",
    },
]


@dataclass(frozen=True)
class ExperimentConfig:
    db_path: str = "data/upbit_data.db"
    price_table: str = "btc_15m_advance"
    output_dir: str = "test/results"
    seq_len: int = 32
    horizon: int = 1
    train_ratio: float = 0.75
    ticker: str | None = None
    max_rows: int | None = 1200
    max_train_windows: int = 512
    max_test_windows: int = 128
    epochs: int = 1
    batch_size: int = 128
    min_batch_size: int = 8
    hidden_dim: int = 64
    learning_rate: float = 0.001
    grad_clip_norm: float = 1.0
    optuna_trials: int = 0
    optuna_timeout: int | None = None
    preprocessing_mode: str = "adaptive_stationarity"
    adf_pvalue_threshold: float = 0.05
    drift_ratio_threshold: float = 0.25
    seed: int = 42
    device: str = "auto"


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


def load_price_with_text_features(config: ExperimentConfig) -> pd.DataFrame:
    """Load candle rows and left-join already-built 15-minute text features."""

    limit_sql = f"LIMIT {int(config.max_rows)}" if config.max_rows else ""
    with duckdb.connect(config.db_path) as con:
        if not table_exists(con, config.price_table):
            raise RuntimeError(f"Price table not found in DuckDB: {config.price_table}")

        text_exists = table_exists(con, "text_features_15m")
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
        raise RuntimeError("No price rows loaded. Check table name, ticker, and max_rows.")

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
    out_frames = []
    for ticker, ticker_df in df.groupby("ticker", sort=False):
        tdf = ticker_df.sort_values("timestamp").copy()
        close = tdf["close"].replace(0, np.nan)
        volume = tdf["volume"].replace(0, np.nan)
        value = tdf["value"].replace(0, np.nan)

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
        tdf["amihud_illiq"] = tdf["log_return_1"].abs() / value
        tdf["turnover_proxy"] = volume / volume.rolling(96).mean()
        tdf["spread_proxy"] = tdf["hl_range_pct"]
        tdf["close_level_z"] = expanding_zscore(close)
        tdf["log_close_level_z"] = expanding_zscore(np.log(close))
        tdf["close_diff_z"] = expanding_zscore(close.diff())
        tdf["log_return_z"] = expanding_zscore(tdf["log_return_1"])
        tdf["price_rolling_z_96"] = rolling_zscore(close, 96)
        tdf["ema_diff_z"] = expanding_zscore(close.diff().ewm(span=8, adjust=False).mean())
        tdf["ticker"] = ticker
        out_frames.append(tdf)

    enriched = pd.concat(out_frames, ignore_index=True)
    feature_cols = STATIONARY_MARKET_FEATURE_COLUMNS + PRICE_STATE_CANDIDATE_COLUMNS + TEXT_FEATURE_COLUMNS
    enriched[feature_cols] = enriched[feature_cols].replace([np.inf, -np.inf], np.nan)
    enriched[feature_cols] = enriched.groupby("ticker", group_keys=False)[feature_cols].apply(
        lambda frame: frame.ffill().fillna(0.0)
    )
    return enriched


def garman_klass_volatility(df: pd.DataFrame) -> pd.Series:
    high = df["high"].replace(0, np.nan)
    low = df["low"].replace(0, np.nan)
    open_ = df["open"].replace(0, np.nan)
    close = df["close"].replace(0, np.nan)
    term = 0.5 * np.log(high / low) ** 2 - (2 * np.log(2) - 1) * np.log(close / open_) ** 2
    return np.sqrt(term.clip(lower=0)).rolling(16).mean()


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std.replace(0, np.nan)


def expanding_zscore(series: pd.Series, min_periods: int = 32) -> pd.Series:
    mean = series.expanding(min_periods=min_periods).mean()
    std = series.expanding(min_periods=min_periods).std()
    return (series - mean) / std.replace(0, np.nan)


def split_by_time(df: pd.DataFrame, train_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    timestamps = pd.Series(pd.to_datetime(df["timestamp"]).sort_values().unique())
    split_idx = max(1, int(len(timestamps) * train_ratio))
    split_timestamp = timestamps.iloc[min(split_idx, len(timestamps) - 1)]
    return df[df["timestamp"] < split_timestamp].copy(), df[df["timestamp"] >= split_timestamp].copy()


def resolve_feature_columns(
    train_df: pd.DataFrame,
    base_columns: list[str],
    config: ExperimentConfig,
) -> tuple[list[str], dict[str, object]]:
    """Choose price-state representation without forcing one preprocessing forever.

    Original Phase-2 work compared several preprocessing choices. This keeps that
    spirit: diagnose the train split, then select the first representation that
    looks stationary enough. If none passes, use stationary-only derived factors
    rather than raw level shortcut learning.
    """

    mode = config.preprocessing_mode
    diagnostics = {
        column: stationarity_diagnostics(
            train_df[column],
            adf_pvalue_threshold=config.adf_pvalue_threshold,
            drift_ratio_threshold=config.drift_ratio_threshold,
        )
        for column in PRICE_STATE_CANDIDATE_COLUMNS
    }
    selected: list[str] = []

    if mode == "stationary_only":
        selected = []
    elif mode == "level_plus_stationary":
        selected = ["close_level_z"]
    elif mode == "all_price_state_candidates":
        selected = list(PRICE_STATE_CANDIDATE_COLUMNS)
    elif mode == "adaptive_stationarity":
        for column in PRICE_STATE_CANDIDATE_COLUMNS:
            diag = diagnostics[column]
            if diag["adf_pass"] and diag["drift_pass"]:
                selected = [column]
                break
    else:
        raise ValueError(
            "Unknown preprocessing_mode. Use adaptive_stationarity, stationary_only, "
            "level_plus_stationary, or all_price_state_candidates."
        )

    resolved = list(dict.fromkeys(base_columns + selected))
    metadata = {
        "preprocessing_mode": mode,
        "selected_price_state_columns": selected,
        "stationarity_diagnostics": diagnostics,
    }
    return resolved, metadata


def stationarity_diagnostics(
    series: pd.Series,
    adf_pvalue_threshold: float = 0.05,
    drift_ratio_threshold: float = 0.25,
) -> dict[str, object]:
    values = pd.Series(series).replace([np.inf, -np.inf], np.nan).dropna()
    values = values.iloc[-min(len(values), 3000) :]
    if len(values) < 32 or values.std() == 0:
        return {
            "adf_pvalue": None,
            "adf_pass": False,
            "drift_ratio": None,
            "drift_pass": False,
            "n": int(len(values)),
        }

    adf_pvalue = None
    try:
        from statsmodels.tsa.stattools import adfuller

        adf_pvalue = float(adfuller(values.to_numpy(dtype=float), autolag="AIC")[1])
    except Exception:
        adf_pvalue = None

    drift_ratio = rolling_drift_ratio(values)
    adf_pass = adf_pvalue is not None and adf_pvalue <= adf_pvalue_threshold
    drift_pass = drift_ratio is not None and drift_ratio <= drift_ratio_threshold
    return {
        "adf_pvalue": adf_pvalue,
        "adf_pass": bool(adf_pass),
        "drift_ratio": drift_ratio,
        "drift_pass": bool(drift_pass),
        "n": int(len(values)),
    }


def rolling_drift_ratio(series: pd.Series, window: int = 64) -> float | None:
    if len(series) < window * 2:
        return None
    rolling_mean = series.rolling(window).mean().dropna()
    if rolling_mean.empty:
        return None
    drift = float(rolling_mean.diff().abs().mean())
    scale = float(series.std())
    if scale <= 0:
        return None
    return drift / scale


def fit_feature_scaler(train_df: pd.DataFrame, feature_columns: list[str]) -> dict[str, pd.Series]:
    values = train_df[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return {
        "mean": values.mean(),
        "std": values.std().replace(0, 1.0).fillna(1.0),
    }


def build_windows(
    df: pd.DataFrame,
    feature_columns: list[str],
    seq_len: int,
    horizon: int,
    scaler: dict[str, pd.Series],
    max_windows: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X, y_return, prev_close, target_close = [], [], [], []
    for _, ticker_df in df.groupby("ticker", sort=False):
        tdf = ticker_df.sort_values("timestamp").copy()
        if len(tdf) < seq_len + horizon + 1:
            continue
        features = (tdf[feature_columns].fillna(0.0) - scaler["mean"]) / scaler["std"]
        closes = tdf["close"].to_numpy(dtype=float)
        feature_values = features.to_numpy(dtype=float)
        for i in range(seq_len, len(tdf) - horizon + 1):
            target_idx = i + horizon - 1
            previous = closes[i - 1]
            target = closes[target_idx]
            if previous <= 0 or target <= 0:
                continue
            X.append(feature_values[i - seq_len : i])
            y_return.append(np.log(target / previous))
            prev_close.append(previous)
            target_close.append(target)
    if not X:
        empty_x = np.empty((0, seq_len, len(feature_columns)), dtype=np.float32)
        return empty_x, np.array([]), np.array([]), np.array([])

    X_arr = np.asarray(X, dtype=np.float32)
    y_arr = np.asarray(y_return, dtype=np.float32)
    prev_arr = np.asarray(prev_close, dtype=np.float64)
    target_arr = np.asarray(target_close, dtype=np.float64)
    if max_windows and len(X_arr) > max_windows:
        X_arr = X_arr[-max_windows:]
        y_arr = y_arr[-max_windows:]
        prev_arr = prev_arr[-max_windows:]
        target_arr = target_arr[-max_windows:]
    return X_arr, y_arr, prev_arr, target_arr


def prepare_datasets(
    feature_df: pd.DataFrame,
    base_feature_columns: list[str],
    config: ExperimentConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], dict[str, object]]:
    train_df, test_df = split_by_time(feature_df, config.train_ratio)
    feature_columns, preprocessing_metadata = resolve_feature_columns(train_df, base_feature_columns, config)
    scaler = fit_feature_scaler(train_df, feature_columns)
    X_train, y_train, _, _ = build_windows(
        train_df, feature_columns, config.seq_len, config.horizon, scaler, config.max_train_windows
    )
    X_test, y_test_return, prev_test, target_test = build_windows(
        test_df, feature_columns, config.seq_len, config.horizon, scaler, config.max_test_windows
    )
    return X_train, y_train, X_test, y_test_return, prev_test, target_test, feature_columns, preprocessing_metadata


def run_representative_experiment(config: ExperimentConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_df = load_price_with_text_features(config)
    feature_df = add_market_features(raw_df)
    feature_sets = {
        "price_ohlcv_technical_baseline": BASE_FEATURE_COLUMNS,
        "text_context_independent_variables": TEXT_AWARE_FEATURE_COLUMNS,
    }

    metrics_rows = []
    prediction_rows = []
    for feature_set_name, feature_columns in feature_sets.items():
        (
            X_train,
            y_train,
            X_test,
            y_test_return,
            prev_test,
            target_test,
            resolved_feature_columns,
            preprocessing_metadata,
        ) = prepare_datasets(
            feature_df, feature_columns, config
        )
        if len(X_train) == 0 or len(X_test) == 0:
            raise RuntimeError(
                f"Not enough rows for {feature_set_name}. "
                "Lower --seq-len or increase --max-rows in the approved execution environment."
            )

        for model_name in [spec["name"] for spec in MODEL_SPECS]:
            run_config = tune_with_optuna_if_requested(model_name, X_train, y_train, config)
            pred_return = fit_predict_torch_model(model_name, X_train, y_train, X_test, run_config)
            pred_close = prev_test * np.exp(pred_return.astype(np.float64))
            metrics = evaluate_forecast(target_test, pred_close, prev_test)
            metrics_rows.append(
                {
                    "feature_set": feature_set_name,
                    "model": model_name,
                    "seq_len": config.seq_len,
                    "horizon": config.horizon,
                    "train_windows": int(len(X_train)),
                    "test_windows": int(len(X_test)),
                    "feature_count": int(len(resolved_feature_columns)),
                    "preprocessing_mode": str(preprocessing_metadata["preprocessing_mode"]),
                    "selected_price_state": ",".join(preprocessing_metadata["selected_price_state_columns"]) or "none",
                    "batch_size": int(run_config.batch_size),
                    "hidden_dim": int(run_config.hidden_dim),
                    "learning_rate": float(run_config.learning_rate),
                    "grad_clip_norm": float(run_config.grad_clip_norm),
                    **metrics,
                }
            )
            prediction_rows.extend(
                {
                    "feature_set": feature_set_name,
                    "model": model_name,
                    "target_close": float(target_test[i]),
                    "pred_close": float(pred_close[i]),
                    "naive_pred": float(prev_test[i]),
                    "target_return": float(y_test_return[i]),
                    "pred_return": float(pred_return[i]),
                }
                for i in range(len(pred_return))
            )

    return pd.DataFrame(metrics_rows), pd.DataFrame(prediction_rows)


def tune_with_optuna_if_requested(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: ExperimentConfig,
) -> ExperimentConfig:
    if config.optuna_trials <= 0:
        return config
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError("Optuna tuning requested but optuna is not installed in this environment.") from exc

    split_idx = max(1, int(len(X_train) * 0.8))
    if len(X_train) - split_idx < 4:
        return config
    X_fit, y_fit = X_train[:split_idx], y_train[:split_idx]
    X_val, y_val = X_train[split_idx:], y_train[split_idx:]

    def objective(trial):
        trial_config = ExperimentConfig(
            db_path=config.db_path,
            price_table=config.price_table,
            output_dir=config.output_dir,
            seq_len=config.seq_len,
            horizon=config.horizon,
            train_ratio=config.train_ratio,
            ticker=config.ticker,
            max_rows=config.max_rows,
            max_train_windows=min(config.max_train_windows, 384),
            max_test_windows=config.max_test_windows,
            epochs=max(1, min(config.epochs, 2)),
            batch_size=trial.suggest_categorical("batch_size", [32, 64, 128]),
            min_batch_size=config.min_batch_size,
            hidden_dim=trial.suggest_categorical("hidden_dim", [32, 64, 128]),
            learning_rate=trial.suggest_float("learning_rate", 3e-4, 3e-3, log=True),
            grad_clip_norm=trial.suggest_categorical("grad_clip_norm", [0.5, 1.0, 2.0]),
            optuna_trials=0,
            optuna_timeout=None,
            preprocessing_mode=config.preprocessing_mode,
            adf_pvalue_threshold=config.adf_pvalue_threshold,
            drift_ratio_threshold=config.drift_ratio_threshold,
            seed=config.seed,
            device=config.device,
        )
        pred = fit_predict_torch_model(model_name, X_fit, y_fit, X_val, trial_config)
        score = float(np.mean(np.abs(pred - y_val)))
        trial.report(score, step=0)
        return score

    sampler = optuna.samplers.TPESampler(seed=config.seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=0)
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=config.optuna_trials, timeout=config.optuna_timeout, gc_after_trial=True)
    params = study.best_params
    return ExperimentConfig(
        db_path=config.db_path,
        price_table=config.price_table,
        output_dir=config.output_dir,
        seq_len=config.seq_len,
        horizon=config.horizon,
        train_ratio=config.train_ratio,
        ticker=config.ticker,
        max_rows=config.max_rows,
        max_train_windows=config.max_train_windows,
        max_test_windows=config.max_test_windows,
        epochs=config.epochs,
        batch_size=int(params.get("batch_size", config.batch_size)),
        min_batch_size=config.min_batch_size,
        hidden_dim=int(params.get("hidden_dim", config.hidden_dim)),
        learning_rate=float(params.get("learning_rate", config.learning_rate)),
        grad_clip_norm=float(params.get("grad_clip_norm", config.grad_clip_norm)),
        optuna_trials=0,
        optuna_timeout=None,
        preprocessing_mode=config.preprocessing_mode,
        adf_pvalue_threshold=config.adf_pvalue_threshold,
        drift_ratio_threshold=config.drift_ratio_threshold,
        seed=config.seed,
        device=config.device,
    )


def fit_predict_torch_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    config: ExperimentConfig,
) -> np.ndarray:
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise RuntimeError(
            "This representative model smoke test requires PyTorch in the execution environment. "
            "Install/use an environment that already supports the earlier Torch notebooks."
        ) from exc

    torch.manual_seed(config.seed)
    device = resolve_torch_device(torch, config.device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    input_dim = X_train.shape[-1]
    text_start_idx = len(BASE_FEATURE_COLUMNS) if input_dim > len(BASE_FEATURE_COLUMNS) else None

    batch_size = max(config.min_batch_size, config.batch_size)
    last_error: Exception | None = None
    while batch_size >= config.min_batch_size:
        try:
            model = build_model(
                model_name=model_name,
                input_dim=input_dim,
                seq_len=X_train.shape[1],
                hidden_dim=config.hidden_dim,
                text_start_idx=text_start_idx,
                nn=nn,
            ).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
            criterion = nn.SmoothL1Loss()

            dataset = TensorDataset(
                torch.tensor(X_train, dtype=torch.float32),
                torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32),
            )
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            model.train()
            for _ in range(config.epochs):
                for bx, by in loader:
                    bx, by = bx.to(device), by.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = criterion(model(bx), by)
                    loss.backward()
                    clip_grad_norm(model.parameters(), config.grad_clip_norm, torch)
                    optimizer.step()

            preds = predict_in_batches(model, X_test, batch_size=batch_size, device=device, torch=torch)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            return preds
        except RuntimeError as exc:
            if not is_torch_oom(exc):
                raise
            last_error = exc
            if device.type == "cuda":
                torch.cuda.empty_cache()
            batch_size = batch_size // 2
    raise RuntimeError(
        f"OOM even after reducing batch size to {config.min_batch_size}. "
        "Lower --seq-len, --hidden-dim, --max-train-windows, or --max-test-windows."
    ) from last_error


def is_torch_oom(exc: RuntimeError) -> bool:
    message = str(exc).lower()
    return "out of memory" in message or "cuda error: out of memory" in message


def clip_grad_norm(parameters, max_norm: float, torch) -> None:
    if max_norm and max_norm > 0:
        torch.nn.utils.clip_grad_norm_(list(parameters), max_norm=max_norm)


def predict_in_batches(model, X_test: np.ndarray, batch_size: int, device, torch) -> np.ndarray:
    preds = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(X_test), batch_size):
            bx = torch.tensor(X_test[start : start + batch_size], dtype=torch.float32).to(device)
            preds.append(model(bx).cpu().numpy().reshape(-1))
    return np.concatenate(preds) if preds else np.array([])


def resolve_torch_device(torch_module, requested: str):
    if requested == "auto":
        return torch_module.device("cuda" if torch_module.cuda.is_available() else "cpu")
    return torch_module.device(requested)


def build_model(model_name: str, input_dim: int, seq_len: int, hidden_dim: int, text_start_idx: int | None, nn):
    if model_name == "LSTMRepresentative":
        return LSTMRepresentative(input_dim, hidden_dim, nn)
    if model_name == "TransformerRepresentative":
        return TransformerRepresentative(input_dim, hidden_dim, nn)
    if model_name == "MambaLite":
        return MambaLite(input_dim, hidden_dim, nn)
    if model_name == "TimeXerLite":
        return TimeXerLite(input_dim, hidden_dim, text_start_idx, nn)
    if model_name == "ITransformerLite":
        return ITransformerLite(seq_len, input_dim, hidden_dim, nn)
    raise ValueError(f"Unknown model: {model_name}")


class LSTMRepresentative:
    def __init__(self, input_dim: int, hidden_dim: int, nn):
        self.module = nn.Sequential()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=1)
        self.head = nn.Linear(hidden_dim, 1)

    def to(self, device):
        self.lstm.to(device)
        self.head.to(device)
        return self

    def parameters(self):
        return list(self.lstm.parameters()) + list(self.head.parameters())

    def train(self):
        self.lstm.train()
        self.head.train()

    def eval(self):
        self.lstm.eval()
        self.head.eval()

    def __call__(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


class TransformerRepresentative:
    def __init__(self, input_dim: int, hidden_dim: int, nn):
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        layer = nn.TransformerEncoderLayer(hidden_dim, nhead=4, dim_feedforward=hidden_dim * 2, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.head = nn.Linear(hidden_dim, 1)

    def to(self, device):
        self.in_proj.to(device)
        self.encoder.to(device)
        self.head.to(device)
        return self

    def parameters(self):
        return list(self.in_proj.parameters()) + list(self.encoder.parameters()) + list(self.head.parameters())

    def train(self):
        self.in_proj.train()
        self.encoder.train()
        self.head.train()

    def eval(self):
        self.in_proj.eval()
        self.encoder.eval()
        self.head.eval()

    def __call__(self, x):
        encoded = self.encoder(self.in_proj(x))
        return self.head(encoded[:, -1, :])


class MambaLite:
    """Small gated causal-conv SSM proxy, not a full Mamba implementation."""

    def __init__(self, input_dim: int, hidden_dim: int, nn):
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.filter = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=2, groups=1)
        self.gate = nn.Linear(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)

    def to(self, device):
        self.in_proj.to(device)
        self.filter.to(device)
        self.gate.to(device)
        self.head.to(device)
        return self

    def parameters(self):
        return (
            list(self.in_proj.parameters())
            + list(self.filter.parameters())
            + list(self.gate.parameters())
            + list(self.head.parameters())
        )

    def train(self):
        self.in_proj.train()
        self.filter.train()
        self.gate.train()
        self.head.train()

    def eval(self):
        self.in_proj.eval()
        self.filter.eval()
        self.gate.eval()
        self.head.eval()

    def __call__(self, x):
        import torch

        h = self.in_proj(x)
        conv = self.filter(h.transpose(1, 2))[:, :, : x.shape[1]].transpose(1, 2)
        gated = torch.tanh(conv) * torch.sigmoid(self.gate(h))
        return self.head(gated[:, -1, :])


class TimeXerLite:
    """Tiny exogenous cross-attention prototype inspired by TimeXer."""

    def __init__(self, input_dim: int, hidden_dim: int, text_start_idx: int | None, nn):
        self.text_start_idx = text_start_idx
        self.market_dim = text_start_idx or input_dim
        self.text_dim = max(input_dim - self.market_dim, 1)
        self.market_proj = nn.Linear(self.market_dim, hidden_dim)
        self.text_proj = nn.Linear(self.text_dim, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.head = nn.Linear(hidden_dim, 1)

    def to(self, device):
        self.market_proj.to(device)
        self.text_proj.to(device)
        self.cross_attn.to(device)
        self.head.to(device)
        return self

    def parameters(self):
        return (
            list(self.market_proj.parameters())
            + list(self.text_proj.parameters())
            + list(self.cross_attn.parameters())
            + list(self.head.parameters())
        )

    def train(self):
        self.market_proj.train()
        self.text_proj.train()
        self.cross_attn.train()
        self.head.train()

    def eval(self):
        self.market_proj.eval()
        self.text_proj.eval()
        self.cross_attn.eval()
        self.head.eval()

    def __call__(self, x):
        if self.text_start_idx is None:
            market_x = x
            text_x = x[:, :, -1:].clone()
        else:
            market_x = x[:, :, : self.text_start_idx]
            text_x = x[:, :, self.text_start_idx :]
        q = self.market_proj(market_x)
        k = self.text_proj(text_x)
        attended, _ = self.cross_attn(q, k, k)
        return self.head(attended[:, -1, :])


class ITransformerLite:
    """Variables-as-tokens prototype inspired by iTransformer."""

    def __init__(self, seq_len: int, input_dim: int, hidden_dim: int, nn):
        self.var_proj = nn.Linear(seq_len, hidden_dim)
        layer = nn.TransformerEncoderLayer(hidden_dim, nhead=4, dim_feedforward=hidden_dim * 2, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.head = nn.Linear(hidden_dim * input_dim, 1)

    def to(self, device):
        self.var_proj.to(device)
        self.encoder.to(device)
        self.head.to(device)
        return self

    def parameters(self):
        return list(self.var_proj.parameters()) + list(self.encoder.parameters()) + list(self.head.parameters())

    def train(self):
        self.var_proj.train()
        self.encoder.train()
        self.head.train()

    def eval(self):
        self.var_proj.eval()
        self.encoder.eval()
        self.head.eval()

    def __call__(self, x):
        tokens = x.transpose(1, 2)
        encoded = self.encoder(self.var_proj(tokens))
        return self.head(encoded.flatten(start_dim=1))


def evaluate_forecast(y_true: np.ndarray, y_pred: np.ndarray, prev_close: np.ndarray) -> dict[str, float]:
    residual = y_true - y_pred
    naive_mae = float(np.mean(np.abs(y_true - prev_close)))
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual**2)))
    direction_true = np.sign(y_true - prev_close)
    direction_pred = np.sign(y_pred - prev_close)
    da = float(np.mean(direction_true == direction_pred) * 100.0)
    mase = float(mae / naive_mae) if naive_mae > 0 else float("nan")
    return {
        "RMSE_KRW": rmse,
        "MAE_KRW": mae,
        "DA_percent": da,
        "MASE": mase,
        "Naive_MAE_KRW": naive_mae,
    }


def write_experiment_report(metrics_df: pd.DataFrame, predictions_df: pd.DataFrame, config: ExperimentConfig) -> Path:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_path = output_dir / f"4_text_independent_variable_metrics_{timestamp}.csv"
    predictions_path = output_dir / f"4_text_independent_variable_predictions_{timestamp}.csv"
    report_path = output_dir / f"4_text_independent_variable_report_{timestamp}.md"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    predictions_df.to_csv(predictions_path, index=False, encoding="utf-8-sig")

    best_row = metrics_df.sort_values(["MASE", "RMSE_KRW"], ascending=True).iloc[0].to_dict()
    report = f"""# 4. 텍스트 독립변수 대표 모델 점검

생성 시각: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 목적
이 실험은 기존 15개 모델 전면 비교를 대체하는 것이 아니라, 대표 구조별로 하나씩만 뽑아 빠르게 점검하는 소규모 실험입니다.
가격 기반 기준선과 텍스트 반영 변수셋이 실제로 차이를 만드는지 확인합니다.

## 모델 구성
{markdown_table(pd.DataFrame(MODEL_SPECS))}

## 실행 설정
- DuckDB 경로: `{config.db_path}`
- 가격 테이블: `{config.price_table}`
- 티커 필터: `{config.ticker or "all available"}`
- 최대 원본 행 수: `{config.max_rows}`
- 시퀀스 길이: `{config.seq_len}` (15분 단위)
- 최대 학습/평가 윈도우: `{config.max_train_windows}` / `{config.max_test_windows}`
- 에폭 수: `{config.epochs}`
- 은닉 차원: `{config.hidden_dim}`
- 배치 크기 / 최소 배치 크기: `{config.batch_size}` / `{config.min_batch_size}`
- 그래디언트 클리핑 노름: `{config.grad_clip_norm}`
- Optuna trial 수: `{config.optuna_trials}`
- 전처리 모드: `{config.preprocessing_mode}`
- ADF p-value 임계값: `{config.adf_pvalue_threshold}`
- 롤링 드리프트 비율 임계값: `{config.drift_ratio_threshold}`
- 로컬 Codex 메모: 여기서는 문법 점검만 수행했으며, 실제 모델 실행은 학교 서버, CI, 스케줄러, 또는 명시적으로 승인된 로컬 실행에서만 진행한다.

## 변수 구성
- `price_ohlcv_technical_baseline`: 정상성 진단을 통과한 가격·거래량 기술지표만 사용한 기준선이다.
- `text_context_independent_variables`: 위 기준선에 뉴스, 리포트, SNS 텍스트 변수들을 더한 확장 집합이다.
- 가격 상태 후보: `close_level_z`, `log_close_level_z`, `close_diff_z`, `log_return_z`, `price_rolling_z_96`, `ema_diff_z`.
- `adaptive_stationarity`: 원시 가격 수준을 무조건 쓰지 않고, ADF/드리프트 검정으로 안전한 상태 변수를 고르는 절차다.
- 이번 실험의 핵심은 “이번에 쓰는 변수가 무엇인지”를 분명히 보여 주는 데 있다.

## 지표
{markdown_table(metrics_df)}

## 최우수 행
```json
{json.dumps(best_row, ensure_ascii=False, indent=2)}
```

## 해석
- `price_ohlcv_technical_baseline`은 가격·거래량 변수만으로 어디까지 갈 수 있는지 보여 준다.
- `text_context_independent_variables`는 실시간 뉴스/리포트/SNS 요인을 더한 결과다.
- 가격 수준은 무조건 제거하지도, 무조건 유지하지도 않는다. 학습 분할을 점검한 뒤 가장 안전한 가격 상태 신호를 고른다. `--preprocessing-mode level_plus_stationary`는 의도적으로 비정상 수준 동작과 lag-copy 위험을 비교할 때만 사용한다.
- `MASE < 1.0`이면 naive persistence보다 낫고, `DA_percent > 50`이면 무작위 방향 추정보다 낫다.
- 이 점검용 실험은 일부러 작게 잡았다. 기본 변수 정렬과 지표 일관성이 확인된 뒤에만 `--max-rows`, `--seq-len`, `--epochs`를 늘린다.

## 최근 연구 단서
- TimeXer, NeurIPS 2024: 외생 변수를 포함한 Transformer 예측.
- iTransformer, ICLR 2024: 변수 토큰을 뒤집는 multivariate time series Transformer.
- Mamba4Cast, 2024: 효율적인 시계열 예측을 위한 state-space/Mamba 계열.
- 2024~2026의 뉴스/감성 기반 주가 예측 연구들은 텍스트 변수군의 근거가 된다.

## 산출 파일
- 지표 CSV: `{metrics_path.as_posix()}`
- 예측 CSV: `{predictions_path.as_posix()}`
"""
    report_path.write_text(report, encoding="utf-8")
    return report_path


def markdown_table(df: pd.DataFrame) -> str:
    headers = [str(column) for column in df.columns]
    rows = []
    for _, row in df.iterrows():
        formatted = []
        for value in row:
            if isinstance(value, float):
                formatted.append(f"{value:.6f}")
            else:
                formatted.append(str(value))
        rows.append(formatted)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def parse_args(argv: Iterable[str] | None = None) -> ExperimentConfig:
    parser = argparse.ArgumentParser(description="Run a tiny representative text-factor model smoke test.")
    parser.add_argument("--db", default="data/upbit_data.db", help="DuckDB path.")
    parser.add_argument("--table", default="btc_15m_advance", help="Candle table name.")
    parser.add_argument("--output-dir", default="test/results", help="Output directory.")
    parser.add_argument("--seq-len", type=int, default=32, help="Input window length in 15-minute steps.")
    parser.add_argument("--horizon", type=int, default=1, help="Forecast horizon in 15-minute steps.")
    parser.add_argument("--train-ratio", type=float, default=0.75, help="Chronological train split ratio.")
    parser.add_argument("--ticker", default=None, help="Optional ticker filter, e.g. KRW-BTC.")
    parser.add_argument("--max-rows", type=int, default=1200, help="Small default row cap for rough smoke tests.")
    parser.add_argument("--max-train-windows", type=int, default=512, help="Training window cap.")
    parser.add_argument("--max-test-windows", type=int, default=128, help="Test window cap.")
    parser.add_argument("--epochs", type=int, default=1, help="Tiny default epoch count.")
    parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size.")
    parser.add_argument("--min-batch-size", type=int, default=8, help="Smallest batch size after OOM backoff.")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Shallow-but-wide hidden width.")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Optimizer learning rate.")
    parser.add_argument("--grad-clip-norm", type=float, default=1.0, help="Gradient clipping max norm.")
    parser.add_argument("--optuna-trials", type=int, default=0, help="Optional tiny Optuna trial count.")
    parser.add_argument("--optuna-timeout", type=int, default=None, help="Optional Optuna timeout in seconds.")
    parser.add_argument(
        "--preprocessing-mode",
        default="adaptive_stationarity",
        choices=["adaptive_stationarity", "stationary_only", "level_plus_stationary", "all_price_state_candidates"],
        help="How to include transformed price-state features.",
    )
    parser.add_argument("--adf-pvalue-threshold", type=float, default=0.05, help="ADF pass threshold.")
    parser.add_argument("--drift-ratio-threshold", type=float, default=0.25, help="Rolling mean drift pass threshold.")
    parser.add_argument("--seed", type=int, default=42, help="Torch seed.")
    parser.add_argument("--device", default="auto", help="Torch device: auto, cpu, cuda.")
    args = parser.parse_args(list(argv) if argv is not None else None)
    return ExperimentConfig(
        db_path=resolve_db_path(args.db),
        price_table=args.table,
        output_dir=args.output_dir,
        seq_len=args.seq_len,
        horizon=args.horizon,
        train_ratio=args.train_ratio,
        ticker=args.ticker,
        max_rows=args.max_rows,
        max_train_windows=args.max_train_windows,
        max_test_windows=args.max_test_windows,
        epochs=args.epochs,
        batch_size=args.batch_size,
        min_batch_size=args.min_batch_size,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        grad_clip_norm=args.grad_clip_norm,
        optuna_trials=args.optuna_trials,
        optuna_timeout=args.optuna_timeout,
        preprocessing_mode=args.preprocessing_mode,
        adf_pvalue_threshold=args.adf_pvalue_threshold,
        drift_ratio_threshold=args.drift_ratio_threshold,
        seed=args.seed,
        device=args.device,
    )


def main(argv: Iterable[str] | None = None) -> None:
    config = parse_args(argv)
    metrics_df, predictions_df = run_representative_experiment(config)
    report_path = write_experiment_report(metrics_df, predictions_df, config)
    print(metrics_df.to_string(index=False))
    print(f"[SUCCESS] Report saved to {report_path}")


if __name__ == "__main__":
    main()
