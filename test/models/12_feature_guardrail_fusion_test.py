# [FOR COMMIT TRACKING ONLY - DO NOT EXECUTE]
# This file is automatically mirrored from the corresponding .ipynb for git diff purposes.
# Actual research execution should be performed in the Jupyter Notebook (.ipynb)
# or in an approved remote/server environment.

# %% [markdown]
# # 12번 feature guardrail fusion 실험
#
# 10번에서 살아남은 `balanced_composite` 점예측 branch를 기본 토대로 두고,
# 11번에서 살아남은 `absolute_move` 위험 확률 branch를 guardrail로 결합합니다.
#
# 12번의 주 실험 축은 새 알고리즘 추가가 아니라 **코인 독립변수 조합**입니다.
# 즉 어떤 feature group이 점예측 collapse를 줄이고, 위험 gate와 결합했을 때
# MDD/거래비용 관점의 의사결정 품질을 높이는지 확인합니다.
#
# 모든 표와 그림은 notebook inline output으로만 표시하며 서버에 PNG/CSV/Markdown 파일을 저장하지 않습니다.
#

# %%
"""12번: 10번 point objective와 11번 risk gate를 feature group별로 결합한다."""

from __future__ import annotations

import argparse
import copy
import gc
import importlib.util
import json
import math
import sys
from dataclasses import replace
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
OBJECTIVE_BACKEND_PATH = REPO_ROOT / "test" / "models" / "10_objective_ensemble_confirmation_test.py"
RISK_BACKEND_PATH = REPO_ROOT / "test" / "models" / "11_distributional_capacity_diagnostics_test.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"backend를 불러오지 못했다: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


obj = load_module(OBJECTIVE_BACKEND_PATH, "objective_backend_for_12")
risk = load_module(RISK_BACKEND_PATH, "risk_backend_for_12")
diag = obj.diag
base = obj.base


DEFAULT_FEATURE_GROUPS = [
    "ohlcv_core",
    "coin_liquidity_micro",
    "coin_volatility_regime",
    "coin_momentum_reversal",
    "coin_orderflow_proxy",
    "coin_multitimeframe_structure",
    "coin_shock_event",
    "coin_attention_proxy",
    "coin_calendar_cycle",
    "coin_text_context",
    "coin_cross_market",
    "coin_macro_proxy",
    "coin_onchain_proxy",
    "coin_derivatives_proxy",
    "coin_search_social_dev",
    "coin_full_available",
]
DEFAULT_POINT_MODELS = ["Linear", "PatchTSTLike"]
DEFAULT_RISK_MODELS = ["PatchTSTLike"]
DEFAULT_PREPROCESSINGS = ["seasonal_diff16", "winsor_025"]
DEFAULT_SEEDS = [42, 2026]

FEATURE_GROUP_EXPLANATIONS = {
    "ohlcv_core": "가격 수익률, 단기/중기 수익률, 실현 변동성, 거래량 z-score만 쓰는 기준 feature group이다.",
    "coin_liquidity_micro": "코인 시장에서 중요한 거래대금, turnover, range, Amihud-style illiquidity proxy를 포함한다.",
    "coin_volatility_regime": "급변과 저변동 국면을 나누기 위해 변동성 비율, 상승/하락 변동성, 꼬리 위험 proxy를 포함한다.",
    "coin_momentum_reversal": "RSI, MACD gap, EMA gap, 추세 강도와 단기 되돌림을 포함해 방향성 신호를 확인한다.",
    "coin_orderflow_proxy": "실제 order book이 없을 때 캔들 몸통, 위/아래 꼬리, 종가 위치, signed volume/value로 매수·매도 압력 proxy를 만든다.",
    "coin_multitimeframe_structure": "15분 단일 봉만 보지 않고 1시간, 4시간, 16시간, 2일 근처의 수익률·변동성·추세 구조를 함께 본다.",
    "coin_shock_event": "급등락, 거래량 폭증, range 확장, 점프 후 되돌림을 묶어 이벤트성 시장 충격을 확인한다.",
    "coin_attention_proxy": "검색량·소셜 데이터가 없더라도 거래대금/거래량/range shock으로 시장 관심도 proxy를 만든다.",
    "coin_calendar_cycle": "코인은 24시간 거래되므로 hour/day 주기와 한국/미국 시간대 proxy를 포함한다.",
    "coin_text_context": "뉴스/텍스트 context mart가 있을 때만 감성, shock, topic count를 붙인다. 없으면 core와 동일한 fallback으로 기록한다.",
    "coin_cross_market": "다중 ticker 테이블이 있을 때 ETH/XRP/SOL 등 cross-asset return을 붙인다. 없으면 자동 제외된다.",
    "coin_macro_proxy": "DXY, VIX, 금리, 주가지수, 원/달러, 금·유가 같은 macro/cross-asset 컬럼이 있을 때만 위험자산 레짐 proxy로 쓴다.",
    "coin_onchain_proxy": "active address, exchange flow, whale, SOPR, MVRV, NVT 같은 on-chain 컬럼이 있을 때만 코인 고유 수급 proxy로 쓴다.",
    "coin_derivatives_proxy": "funding rate, open interest, basis, liquidation, long/short ratio 컬럼이 있을 때만 레버리지·청산 압력 proxy로 쓴다.",
    "coin_search_social_dev": "Google Trends, YouTube, Twitter/X, Reddit, Telegram, GitHub activity 같은 관심도·커뮤니티·개발자 활동 컬럼이 있을 때만 쓴다.",
    "coin_full_available": "현재 데이터에서 쓸 수 있는 모든 코인 특화 feature를 합친 상한선 후보이다.",
}

LITERATURE_NOTES = {
    "technical_liquidity": "기술지표와 유동성 변수는 코인 예측 논문에서 가장 기본적인 입력군이다.",
    "sentiment_interest": "뉴스, SNS, 검색 관심도는 가격 자체보다 regime과 이벤트 확률을 보조하는 변수로 쓰는 편이 안전하다.",
    "macro_cross_market": "달러, VIX, 금리, 전통시장 지수와 cross-asset 변수는 BTC가 위험자산처럼 움직이는 구간을 설명할 수 있다.",
    "onchain_orderflow": "온체인/오더북 변수는 코인 고유 정보지만 현재 로컬 mart에 없으면 proxy feature로 먼저 검증한다.",
    "practitioner_orderflow": "실무·영상 자료에서 자주 강조되는 order flow, liquidity sweep, volume spike는 현재 OHLCV proxy로 먼저 테스트한다.",
    "multi_timeframe": "단기 캔들 하나보다 여러 시간축의 추세·변동성·거래량을 함께 보는 multi-timeframe 설계가 단기 코인 예측에 자주 쓰인다.",
}


def configure_inline_matplotlib() -> None:
    diag.configure_inline_matplotlib()


def show_figure(fig) -> None:
    diag.show_figure(fig)


def display_table(title: str, frame: pd.DataFrame, max_rows: int = 40) -> None:
    print(f"\n[{title}]")
    try:
        from IPython.display import display

        display(frame.head(max_rows))
    except Exception:
        print(frame.head(max_rows).to_string(index=False))


def display_markdown(text: str) -> None:
    diag.display_markdown(text)


def safe_divide(numerator: pd.Series | np.ndarray, denominator: pd.Series | np.ndarray, fill: float = 0.0):
    out = np.asarray(numerator, dtype=float) / np.where(np.abs(np.asarray(denominator, dtype=float)) < 1e-12, np.nan, denominator)
    return pd.Series(out).replace([np.inf, -np.inf], np.nan).fillna(fill)


def columns_by_keywords(frame: pd.DataFrame, keywords: list[str], exclude: set[str] | None = None) -> list[str]:
    exclude = exclude or set()
    columns: list[str] = []
    for column in frame.columns:
        lowered = column.lower()
        if column in exclude:
            continue
        if any(keyword in lowered for keyword in keywords):
            columns.append(column)
    return columns


def add_optional_text_features(db_path: Path, features: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """`text_features_15m`가 있으면 timestamp 기준으로 결합한다.

    텍스트 데이터마트가 아직 비어 있거나 없는 서버에서도 12번이 깨지지 않아야 하므로,
    실패는 hard error가 아니라 feature group fallback으로 처리한다.
    """

    text_columns: list[str] = []
    try:
        import duckdb

        with duckdb.connect(str(db_path), read_only=True) as con:
            tables = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
            if "text_features_15m" not in tables:
                return features, text_columns
            columns = [row[1] for row in con.execute("PRAGMA table_info('text_features_15m')").fetchall()]
            candidate_columns = [
                column
                for column in columns
                if column != "timestamp"
                and any(
                    key in column.lower()
                    for key in [
                        "sentiment",
                        "shock",
                        "topic",
                        "event",
                        "risk",
                        "macro",
                        "regulation",
                        "liquidity",
                        "google",
                        "trend",
                        "search",
                        "youtube",
                        "twitter",
                        "tweet",
                        "reddit",
                        "telegram",
                        "discord",
                        "github",
                        "commit",
                        "developer",
                        "social",
                    ]
                )
            ]
            if not candidate_columns:
                return features, text_columns
            selected = ", ".join(["timestamp", *candidate_columns])
            text = con.execute(f"SELECT {selected} FROM text_features_15m ORDER BY timestamp").fetchdf()
    except Exception as exc:
        print(f"[feature-mart] text_features_15m 결합을 건너뜀: {exc}")
        return features, text_columns

    text["timestamp"] = pd.to_datetime(text["timestamp"])
    merged = features.merge(text, on="timestamp", how="left")
    for column in candidate_columns:
        if column in merged.columns:
            merged[column] = merged[column].fillna(0.0)
            text_columns.append(column)
    return merged, text_columns


def add_cross_market_features(
    db_path: Path,
    table: str,
    raw: pd.DataFrame,
    features: pd.DataFrame,
    max_rows: int | None,
    cross_tickers: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """다중 ticker 테이블일 때 cross-asset return을 붙인다.

    현재 12번의 기본 source가 BTC 단일 테이블이면 자동으로 건너뛴다.
    """

    cross_columns: list[str] = []
    if not cross_tickers:
        return features, cross_columns
    if "ticker" not in raw.columns:
        return features, cross_columns

    try:
        import duckdb

        placeholders = ", ".join(["?"] * len(cross_tickers))
        limit_sql = f" LIMIT {int(max_rows) * max(1, len(cross_tickers))}" if max_rows else ""
        query = f"""
        SELECT timestamp, ticker, close
        FROM {table}
        WHERE ticker IN ({placeholders})
        ORDER BY timestamp
        {limit_sql}
        """
        with duckdb.connect(str(db_path), read_only=True) as con:
            cross = con.execute(query, cross_tickers).fetchdf()
    except Exception as exc:
        print(f"[feature-mart] cross-market feature 결합을 건너뜀: {exc}")
        return features, cross_columns

    if cross.empty:
        return features, cross_columns
    cross["timestamp"] = pd.to_datetime(cross["timestamp"])
    cross = cross.sort_values(["ticker", "timestamp"])
    cross["cross_log_return"] = np.log(cross["close"].clip(lower=1e-9)).groupby(cross["ticker"]).diff()
    pivot = cross.pivot_table(index="timestamp", columns="ticker", values="cross_log_return", aggfunc="last")
    pivot = pivot.add_prefix("cross_return_").reset_index()
    merged = features.merge(pivot, on="timestamp", how="left")
    for column in pivot.columns:
        if column != "timestamp" and column in merged.columns:
            merged[column] = merged[column].fillna(0.0)
            cross_columns.append(column)
    return merged, cross_columns


def add_coin_specific_features(features: pd.DataFrame) -> pd.DataFrame:
    df = features.copy()
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    open_ = df["open"].astype(float)
    volume = df["volume"].astype(float)
    value = df["value"].astype(float) if "value" in df.columns else close * volume
    ret = df["log_return_1"].astype(float)
    candle_range = (high - low).replace(0, np.nan)

    # Liquidity / microstructure proxies from OHLCV.
    df["range_z_96"] = base.rolling_z((high - low) / close.replace(0, np.nan), 96)
    df["amihud_illiquidity_proxy"] = (ret.abs() / np.log1p(value).replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    df["volume_acceleration"] = np.log1p(volume).diff().diff()
    df["value_acceleration"] = np.log1p(value).diff().diff()
    df["range_volume_interaction"] = df["hl_range_pct"] * df["volume_z_96"]
    df["turnover_absorption_proxy"] = safe_divide(df["hl_range_pct"], np.log1p(value), fill=0.0)

    # Order-flow proxies. These are not true order-book features; they are candle-derived
    # pressure indicators used until depth/imbalance data is available in the mart.
    df["candle_body_pct"] = (close - open_) / open_.replace(0, np.nan)
    df["upper_wick_pct"] = (high - np.maximum(open_, close)) / close.replace(0, np.nan)
    df["lower_wick_pct"] = (np.minimum(open_, close) - low) / close.replace(0, np.nan)
    df["close_location_value"] = ((close - low) - (high - close)) / candle_range
    df["signed_volume_proxy"] = np.sign(close - open_) * np.log1p(volume)
    df["signed_value_proxy"] = np.sign(close - open_) * np.log1p(value)
    df["volume_pressure_16"] = df["signed_volume_proxy"].rolling(16, min_periods=4).sum()
    df["value_pressure_16"] = df["signed_value_proxy"].rolling(16, min_periods=4).sum()
    df["range_expansion_16"] = safe_divide(df["hl_range_pct"], df["hl_range_pct"].rolling(16, min_periods=4).mean(), fill=1.0)
    df["intrabar_reversal_proxy"] = np.sign(close - open_) * df["close_location_value"]

    # Volatility and regime proxies.
    df["vol_ratio_16_64"] = safe_divide(df["realized_vol_16"], df["realized_vol_64"], fill=1.0)
    df["downside_vol_64"] = ret.clip(upper=0.0).rolling(64, min_periods=16).std()
    df["upside_vol_64"] = ret.clip(lower=0.0).rolling(64, min_periods=16).std()
    df["tail_abs_return_96"] = ret.abs().rolling(96, min_periods=24).quantile(0.90)
    df["vol_of_vol_64"] = df["realized_vol_16"].rolling(64, min_periods=16).std()
    df["drawdown_96"] = close / close.rolling(96, min_periods=24).max() - 1.0

    # Multi-timeframe and shock/attention proxies.
    df["return_192"] = ret.rolling(192, min_periods=48).sum()
    df["realized_vol_192"] = ret.rolling(192, min_periods=48).std()
    df["vol_ratio_64_192"] = safe_divide(df["realized_vol_64"], df["realized_vol_192"], fill=1.0)
    df["price_z_192"] = base.rolling_z(close, 192)
    df["volume_z_192"] = base.rolling_z(np.log1p(volume), 192)
    df["value_z_192"] = base.rolling_z(np.log1p(value), 192)
    df["range_mean_192"] = df["hl_range_pct"].rolling(192, min_periods=48).mean()
    df["trend_strength_192"] = close / close.rolling(192, min_periods=48).mean() - 1.0
    df["abs_return_z_96"] = base.rolling_z(ret.abs(), 96)
    df["volume_shock_z_96"] = base.rolling_z(np.log1p(volume), 96)
    df["value_shock_z_96"] = base.rolling_z(np.log1p(value), 96)
    df["range_shock_z_96"] = base.rolling_z(df["hl_range_pct"], 96)
    df["jump_reversal_4"] = ret.rolling(4, min_periods=2).sum() * -np.sign(ret.shift(4).rolling(4, min_periods=2).sum())
    df["attention_pressure_proxy"] = (
        df["volume_shock_z_96"].clip(lower=0.0)
        + df["value_shock_z_96"].clip(lower=0.0)
        + df["range_shock_z_96"].clip(lower=0.0)
    )
    df["attention_direction_proxy"] = np.sign(ret.rolling(4, min_periods=2).sum()) * df["attention_pressure_proxy"]

    # Momentum and reversal proxies.
    delta = close.diff()
    gain = delta.clip(lower=0.0).rolling(14, min_periods=7).mean()
    loss = (-delta.clip(upper=0.0)).rolling(14, min_periods=7).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi_14_scaled"] = (100.0 - 100.0 / (1.0 + rs)) / 100.0 - 0.5
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    df["macd_gap_pct"] = macd / close.replace(0, np.nan)
    df["macd_signal_gap_pct"] = (macd - macd.ewm(span=9, adjust=False).mean()) / close.replace(0, np.nan)
    df["trend_strength_64"] = close / close.rolling(64, min_periods=16).mean() - 1.0
    df["reversal_4"] = -df["return_4"]

    # Crypto trades 24/7; intraday calendar cycles can proxy regional liquidity.
    ts = pd.to_datetime(df["timestamp"])
    hour = ts.dt.hour + ts.dt.minute / 60.0
    day = ts.dt.dayofweek
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    df["day_sin"] = np.sin(2 * np.pi * day / 7.0)
    df["day_cos"] = np.cos(2 * np.pi * day / 7.0)
    df["korea_active_session"] = ((hour >= 9) & (hour < 18)).astype(float)
    df["us_active_session"] = ((hour >= 22) | (hour < 6)).astype(float)
    return df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)


def register_feature_groups(features: pd.DataFrame, text_columns: list[str], cross_columns: list[str]) -> dict[str, list[str]]:
    base_columns = [
        "log_return_1",
        "return_4",
        "return_16",
        "realized_vol_16",
        "volume_z_96",
        "value_z_96",
    ]
    macro_columns = columns_by_keywords(
        features,
        ["dxy", "vix", "nasdaq", "spx", "s&p", "rate", "yield", "cpi", "fed", "gold", "oil", "kospi", "usdkrw", "usdt"],
        exclude={"timestamp", "ticker"},
    )
    onchain_columns = columns_by_keywords(
        features,
        ["onchain", "active_address", "address", "hash", "miner", "exchange_flow", "reserve", "sopr", "mvrv", "nvt", "utxo", "whale"],
        exclude={"timestamp", "ticker"},
    )
    derivatives_columns = columns_by_keywords(
        features,
        ["funding", "open_interest", "basis", "liquidation", "long_short", "longshort", "perpetual", "futures"],
        exclude={"timestamp", "ticker"},
    )
    search_social_dev_columns = columns_by_keywords(
        features,
        ["google", "trend", "search", "twitter", "tweet", "reddit", "youtube", "telegram", "discord", "github", "commit", "developer", "social"],
        exclude={"timestamp", "ticker"},
    )
    groups = {
        "ohlcv_core": [
            "log_return_1",
            "return_4",
            "return_16",
            "return_64",
            "realized_vol_16",
            "realized_vol_64",
            "hl_range_pct",
            "volume_z_96",
            "value_z_96",
        ],
        "coin_liquidity_micro": [
            "log_return_1",
            "return_4",
            "return_16",
            "realized_vol_16",
            "hl_range_pct",
            "volume_z_96",
            "value_z_96",
            "turnover_proxy",
            "range_z_96",
            "amihud_illiquidity_proxy",
            "volume_acceleration",
            "value_acceleration",
            "range_volume_interaction",
            "turnover_absorption_proxy",
        ],
        "coin_volatility_regime": [
            "log_return_1",
            "return_4",
            "return_16",
            "return_64",
            "realized_vol_16",
            "realized_vol_64",
            "vol_ratio_16_64",
            "downside_vol_64",
            "upside_vol_64",
            "tail_abs_return_96",
            "vol_of_vol_64",
            "drawdown_96",
        ],
        "coin_momentum_reversal": [
            "log_return_1",
            "return_4",
            "return_16",
            "return_64",
            "ema_gap_16",
            "ema_gap_64",
            "rsi_14_scaled",
            "macd_gap_pct",
            "macd_signal_gap_pct",
            "trend_strength_64",
            "reversal_4",
        ],
        "coin_orderflow_proxy": [
            "log_return_1",
            "return_4",
            "realized_vol_16",
            "hl_range_pct",
            "candle_body_pct",
            "upper_wick_pct",
            "lower_wick_pct",
            "close_location_value",
            "signed_volume_proxy",
            "signed_value_proxy",
            "volume_pressure_16",
            "value_pressure_16",
            "range_expansion_16",
            "intrabar_reversal_proxy",
        ],
        "coin_multitimeframe_structure": [
            "log_return_1",
            "return_4",
            "return_16",
            "return_64",
            "return_192",
            "realized_vol_16",
            "realized_vol_64",
            "realized_vol_192",
            "vol_ratio_16_64",
            "vol_ratio_64_192",
            "price_z_192",
            "volume_z_192",
            "value_z_192",
            "range_mean_192",
            "trend_strength_64",
            "trend_strength_192",
        ],
        "coin_shock_event": [
            "log_return_1",
            "return_4",
            "realized_vol_16",
            "abs_return_z_96",
            "volume_shock_z_96",
            "value_shock_z_96",
            "range_shock_z_96",
            "tail_abs_return_96",
            "vol_of_vol_64",
            "drawdown_96",
            "jump_reversal_4",
        ],
        "coin_attention_proxy": [
            "log_return_1",
            "return_4",
            "realized_vol_16",
            "volume_shock_z_96",
            "value_shock_z_96",
            "range_shock_z_96",
            "attention_pressure_proxy",
            "attention_direction_proxy",
            "korea_active_session",
            "us_active_session",
        ],
        "coin_calendar_cycle": [
            "log_return_1",
            "return_4",
            "return_16",
            "realized_vol_16",
            "volume_z_96",
            "value_z_96",
            "hour_sin",
            "hour_cos",
            "day_sin",
            "day_cos",
            "korea_active_session",
            "us_active_session",
        ],
        "coin_text_context": [
            "log_return_1",
            "return_4",
            "return_16",
            "realized_vol_16",
            "volume_z_96",
            "value_z_96",
            *text_columns,
        ],
        "coin_cross_market": [
            "log_return_1",
            "return_4",
            "return_16",
            "realized_vol_16",
            "volume_z_96",
            "value_z_96",
            *cross_columns,
        ],
        "coin_macro_proxy": [
            *base_columns,
            *macro_columns,
        ],
        "coin_onchain_proxy": [
            *base_columns,
            *onchain_columns,
        ],
        "coin_derivatives_proxy": [
            *base_columns,
            *derivatives_columns,
        ],
        "coin_search_social_dev": [
            *base_columns,
            *search_social_dev_columns,
        ],
    }
    all_candidate_columns = []
    for columns in groups.values():
        all_candidate_columns.extend(columns)
    groups["coin_full_available"] = sorted(set(all_candidate_columns))

    available_groups: dict[str, list[str]] = {}
    for name, columns in groups.items():
        available = [column for column in columns if column in features.columns]
        if name == "coin_text_context" and not text_columns:
            available = groups["ohlcv_core"]
        if name == "coin_cross_market" and not cross_columns:
            continue
        if name == "coin_macro_proxy" and not macro_columns:
            continue
        if name == "coin_onchain_proxy" and not onchain_columns:
            continue
        if name == "coin_derivatives_proxy" and not derivatives_columns:
            continue
        if name == "coin_search_social_dev" and not search_social_dev_columns:
            continue
        if len(available) >= 4:
            available_groups[name] = available
            base.FEATURE_SETS[name] = available
    return available_groups


def load_feature_frame(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, list[str]], pd.DataFrame]:
    db_path = base.resolve_db_path(args.db)
    raw = base.load_price_data(db_path, args.table, args.ticker, args.max_rows)
    features = base.make_features(raw)
    features = add_coin_specific_features(features)
    features, text_columns = add_optional_text_features(db_path, features)
    cross_tickers = [ticker.strip() for ticker in args.cross_tickers.split(",") if ticker.strip()]
    features, cross_columns = add_cross_market_features(db_path, args.table, raw, features, args.max_rows, cross_tickers)
    feature_groups = register_feature_groups(features, text_columns, cross_columns)
    return features, feature_groups, raw


def feature_group_table(feature_groups: dict[str, list[str]]) -> pd.DataFrame:
    rows = []
    for name, columns in feature_groups.items():
        rows.append(
            {
                "feature_group": name,
                "n_features": len(columns),
                "why_used": FEATURE_GROUP_EXPLANATIONS.get(name, "사용 가능한 코인 독립변수 묶음이다."),
                "columns": ", ".join(columns),
            }
        )
    return pd.DataFrame(rows)


def build_cases(args: argparse.Namespace, available_groups: dict[str, list[str]]) -> list[dict[str, object]]:
    requested_groups = [item.strip() for item in args.feature_groups.split(",") if item.strip()]
    groups = [group for group in requested_groups if group in available_groups]
    if not groups:
        raise ValueError(f"사용 가능한 feature group이 없다. requested={requested_groups}, available={sorted(available_groups)}")
    point_models = [item.strip() for item in args.point_models.split(",") if item.strip()]
    risk_models = [item.strip() for item in args.risk_models.split(",") if item.strip()]
    preprocessings = [item.strip() for item in args.preprocessings.split(",") if item.strip()]
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]

    cases: list[dict[str, object]] = []
    for feature_group in groups:
        for preprocessing in preprocessings:
            for point_model in point_models:
                for risk_model in risk_models:
                    for seed in seeds:
                        cases.append(
                            {
                                "feature_group": feature_group,
                                "preprocessing": preprocessing,
                                "point_model": point_model,
                                "risk_model": risk_model,
                                "seed": seed,
                            }
                        )
    return cases[: args.max_cases] if args.max_cases > 0 else cases


def make_case_args(args: argparse.Namespace, feature_group: str) -> argparse.Namespace:
    case_args = copy.copy(args)
    case_args.feature_set = feature_group
    return case_args


def run_point_branch(case: dict[str, object], features: pd.DataFrame, profile, args: argparse.Namespace):
    point_case = {
        "model": case["point_model"],
        "preprocessing": case["preprocessing"],
        "objective": args.point_objective,
        "seed": case["seed"],
    }
    point_args = make_case_args(args, str(case["feature_group"]))
    result, val_pred, test_pred, splits = obj.run_case(point_case, features, profile, point_args)
    result["branch"] = "point"
    result["feature_group"] = case["feature_group"]
    result["risk_model"] = case["risk_model"]
    return result, val_pred, test_pred, splits


def run_risk_branch(case: dict[str, object], features: pd.DataFrame, profile, args: argparse.Namespace):
    base.set_seed(int(case["seed"]))
    feature_columns = base.FEATURE_SETS[str(case["feature_group"])]
    risk_args = make_case_args(args, str(case["feature_group"]))
    data = risk.build_risk_windows(
        features,
        feature_columns,
        args.seq_len,
        args.risk_horizon,
        str(case["preprocessing"]),
        args.normalization,
        args.max_windows,
        args.stride,
    )
    splits, event_score_threshold = risk.prepare_event_splits(
        data,
        args.train_ratio,
        args.val_ratio,
        args.risk_event_kind,
        args.event_quantile,
    )
    batch_size = args.batch_size or profile.batch_size or 32
    while True:
        try:
            model = base.make_model(
                str(case["risk_model"]),
                args.seq_len,
                len(feature_columns),
                int(args.hidden),
            )
            param_count = sum(parameter.numel() for parameter in model.parameters())
            model, curves = risk.train_event_model(model, splits, profile, risk_args, batch_size)
            val_probability = risk.predict_event_probability(model, splits["val"], profile, batch_size)
            test_probability = risk.predict_event_probability(model, splits["test"], profile, batch_size)
            break
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower() or batch_size <= 4:
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            next_batch = max(4, batch_size // 2)
            print(f"[oom-retry] risk batch_size {batch_size} -> {next_batch}")
            batch_size = next_batch

    validation_threshold = risk.select_validation_threshold(splits["val"]["y"], val_probability)
    gate_cutoff = float(np.quantile(val_probability, args.risk_allow_quantile))
    metrics = risk.classification_metrics(splits["test"]["y"], test_probability, validation_threshold)
    calibration = risk.calibration_table(splits["test"]["y"], test_probability)
    selective = risk.selective_risk_curve(splits["test"]["y"], test_probability)
    train_event_rate = float(np.mean(splits["train"]["y"]))
    climatology_brier = float(np.mean((train_event_rate - splits["test"]["y"]) ** 2))
    brier_skill = 1.0 - metrics["brier_score"] / max(1e-12, climatology_brier)
    average_precision_lift = (
        metrics["average_precision"] / metrics["event_rate"]
        if metrics["event_rate"] > 0 and np.isfinite(metrics["average_precision"])
        else float("nan")
    )
    result = {
        "case_id": (
            f"{case['risk_model']}__{case['feature_group']}__{case['preprocessing']}"
            f"__{args.risk_event_kind}__horizon{args.risk_horizon}__seed{case['seed']}"
        ),
        "branch": "risk",
        "feature_group": case["feature_group"],
        "preprocessing": case["preprocessing"],
        "risk_model": case["risk_model"],
        "point_model": case["point_model"],
        "seed": int(case["seed"]),
        "risk_event_kind": args.risk_event_kind,
        "risk_horizon": int(args.risk_horizon),
        "event_score_threshold": event_score_threshold,
        "validation_probability_threshold": validation_threshold,
        "risk_gate_cutoff": gate_cutoff,
        "train_event_rate": train_event_rate,
        "param_count": int(param_count),
        "batch_size": int(batch_size),
        "expected_calibration_error": risk.expected_calibration_error(calibration),
        "climatology_brier_score": climatology_brier,
        "brier_skill_score": brier_skill,
        "average_precision_lift": average_precision_lift,
        "selective_error_at_50pct": float(
            selective.iloc[(selective["coverage"] - 0.5).abs().argmin()]["classification_error"]
        ),
        **metrics,
    }
    if args.case_plots:
        risk.show_event_diagnostics(
            curves,
            splits["test"],
            test_probability,
            validation_threshold,
            str(result["case_id"]),
        )
    return result, val_probability, test_probability, splits, gate_cutoff


def max_drawdown(equity: np.ndarray) -> float:
    if len(equity) == 0:
        return 0.0
    running_max = np.maximum.accumulate(equity)
    drawdown = equity / np.where(running_max == 0, 1.0, running_max) - 1.0
    return float(np.min(drawdown))


def position_metrics(actual_return: np.ndarray, position: np.ndarray, cost: float) -> dict[str, float]:
    position = position.astype(float)
    turnover = np.abs(np.diff(np.r_[0.0, position]))
    pnl = position * actual_return - turnover * cost
    equity = np.exp(np.cumsum(pnl))
    active_share = float(np.mean(position > 0))
    trade_count = int(np.sum(turnover > 0))
    mean_return = float(np.mean(pnl))
    downside = pnl[pnl < 0]
    sortino = mean_return / max(1e-12, float(np.std(downside)) if len(downside) else 1e-12)
    return {
        "cumulative_return": float(equity[-1] - 1.0) if len(equity) else 0.0,
        "mdd": max_drawdown(equity),
        "sortino_proxy": sortino,
        "active_share": active_share,
        "trade_count": trade_count,
        "turnover": float(np.sum(turnover)),
        "mean_step_pnl": mean_return,
    }


def evaluate_fusion_policy(
    case: dict[str, object],
    point_split,
    point_pred: np.ndarray,
    risk_probability: np.ndarray,
    risk_gate_cutoff: float,
    args: argparse.Namespace,
) -> tuple[dict[str, object], pd.DataFrame]:
    n = min(len(point_pred), len(risk_probability), len(point_split["y"]))
    actual = point_split["y"][-n:]
    pred = point_pred[-n:]
    risk_prob = risk_probability[-n:]
    cost = args.cost_bps / 10000.0
    signal_threshold = max(args.min_signal_bps / 10000.0, cost)
    risk_allowed = risk_prob <= risk_gate_cutoff
    point_long = pred > signal_threshold
    risk_only_hold = risk_allowed
    fusion_long = point_long & risk_allowed

    policy_rows = []
    for policy_name, position in [
        ("buy_hold", np.ones(n, dtype=bool)),
        ("point_only", point_long),
        ("risk_gate_only", risk_only_hold),
        ("point_plus_risk_gate", fusion_long),
        ("no_trade", np.zeros(n, dtype=bool)),
    ]:
        metrics = position_metrics(actual, position, cost)
        policy_rows.append(
            {
                "policy": policy_name,
                **metrics,
            }
        )
    policy_frame = pd.DataFrame(policy_rows)
    point_metrics = base.evaluate_predictions({key: value[-n:] for key, value in point_split.items()}, pred)
    fusion_metrics = {
        "case_id": (
            f"{case['feature_group']}__{case['preprocessing']}__{case['point_model']}"
            f"__{case['risk_model']}__seed{case['seed']}"
        ),
        **case,
        "n_aligned": int(n),
        "risk_gate_cutoff": float(risk_gate_cutoff),
        "risk_allowed_share": float(np.mean(risk_allowed)),
        "point_signal_share": float(np.mean(point_long)),
        "fusion_signal_share": float(np.mean(fusion_long)),
        "point_mae_krw": point_metrics["mae_krw"],
        "point_copy_risk_ratio": point_metrics["copy_risk_ratio"],
        "point_direction_accuracy": point_metrics["direction_accuracy"],
        "point_variance_ratio": point_metrics["variance_ratio"],
    }
    for _, row in policy_frame.iterrows():
        prefix = str(row["policy"])
        for metric in ["cumulative_return", "mdd", "sortino_proxy", "active_share", "trade_count", "turnover"]:
            fusion_metrics[f"{prefix}_{metric}"] = float(row[metric])
    return fusion_metrics, policy_frame


def show_fusion_diagnostics(
    case: dict[str, object],
    point_split,
    point_pred: np.ndarray,
    risk_probability: np.ndarray,
    risk_gate_cutoff: float,
    policy_frame: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    import matplotlib.pyplot as plt

    n = min(320, len(point_pred), len(risk_probability), len(point_split["y"]))
    actual = point_split["y"][-n:]
    pred = point_pred[-n:]
    risk_prob = risk_probability[-n:]
    cost = args.cost_bps / 10000.0
    signal_threshold = max(args.min_signal_bps / 10000.0, cost)
    risk_allowed = risk_prob <= risk_gate_cutoff
    point_long = pred > signal_threshold
    fusion_long = point_long & risk_allowed
    x = np.arange(n)

    fig, axes = plt.subplots(3, 2, figsize=(17, 12), dpi=140)
    axes[0, 0].plot(x, actual, label="actual next return")
    axes[0, 0].plot(x, pred, label="point prediction")
    axes[0, 0].axhline(0.0, color="black", linewidth=0.8)
    axes[0, 0].axhline(signal_threshold, color="tab:orange", linestyle="--", label="cost/signal threshold")
    axes[0, 0].set_xlabel("test time index")
    axes[0, 0].set_ylabel("next log return")
    axes[0, 0].set_title("Point forecast from 10번 branch")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(x, risk_prob, label="absolute-move risk probability")
    axes[0, 1].axhline(risk_gate_cutoff, color="black", linestyle="--", label="risk gate cutoff")
    axes[0, 1].fill_between(x, 0, 1, where=risk_allowed, alpha=0.12, label="allowed zone")
    axes[0, 1].set_xlabel("test time index")
    axes[0, 1].set_ylabel("risk probability")
    axes[0, 1].set_title("Risk guardrail from 11번 branch")
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].plot(x, point_long.astype(float), label="point-only long")
    axes[1, 0].plot(x, fusion_long.astype(float), label="point + risk gate long")
    axes[1, 0].set_xlabel("test time index")
    axes[1, 0].set_ylabel("position")
    axes[1, 0].set_title("Decision mask")
    axes[1, 0].legend(fontsize=8)

    for policy_name, position in [
        ("buy_hold", np.ones(n, dtype=bool)),
        ("point_only", point_long),
        ("risk_gate_only", risk_allowed),
        ("point_plus_risk_gate", fusion_long),
    ]:
        turnover = np.abs(np.diff(np.r_[0.0, position.astype(float)]))
        pnl = position.astype(float) * actual - turnover * cost
        axes[1, 1].plot(x, np.exp(np.cumsum(pnl)) - 1.0, label=policy_name)
    axes[1, 1].set_xlabel("test time index")
    axes[1, 1].set_ylabel("cumulative return proxy")
    axes[1, 1].set_title("Transaction-cost-aware policy comparison")
    axes[1, 1].legend(fontsize=8)

    axes[2, 0].scatter(risk_prob, np.abs(actual), s=15, alpha=0.45)
    axes[2, 0].axvline(risk_gate_cutoff, color="black", linestyle="--")
    axes[2, 0].set_xlabel("predicted risk probability")
    axes[2, 0].set_ylabel("realized absolute return")
    axes[2, 0].set_title("Risk probability versus realized movement")

    mdd_rows = policy_frame[policy_frame["policy"] != "no_trade"].copy()
    axes[2, 1].barh(mdd_rows["policy"], mdd_rows["mdd"])
    axes[2, 1].set_xlabel("MDD")
    axes[2, 1].set_title("MDD by policy")
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    fig.suptitle(
        f"{case['feature_group']} / {case['preprocessing']} / "
        f"{case['point_model']} + {case['risk_model']} / seed{case['seed']}"
    )
    show_figure(fig)
    display_table("policy preview", policy_frame)


def show_feature_summary(summary: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    if summary.empty:
        return
    ordered = summary.sort_values(
        ["point_plus_risk_gate_mdd", "point_copy_risk_ratio", "point_plus_risk_gate_cumulative_return"],
        ascending=[False, True, False],
    )
    columns = [
        "feature_group",
        "preprocessing",
        "point_model",
        "risk_model",
        "seed",
        "point_copy_risk_ratio",
        "point_direction_accuracy",
        "point_variance_ratio",
        "risk_allowed_share",
        "fusion_signal_share",
        "point_only_mdd",
        "point_plus_risk_gate_mdd",
        "point_only_cumulative_return",
        "point_plus_risk_gate_cumulative_return",
    ]
    print("\n[12번 feature guardrail leaderboard]")
    print(ordered[columns].head(40).to_string(index=False))

    grouped = (
        summary.groupby("feature_group", as_index=False)
        .agg(
            mean_copy_risk=("point_copy_risk_ratio", "mean"),
            mean_direction=("point_direction_accuracy", "mean"),
            mean_variance_ratio=("point_variance_ratio", "mean"),
            mean_point_mdd=("point_only_mdd", "mean"),
            mean_fusion_mdd=("point_plus_risk_gate_mdd", "mean"),
            mean_fusion_return=("point_plus_risk_gate_cumulative_return", "mean"),
            mean_signal_share=("fusion_signal_share", "mean"),
        )
        .sort_values(["mean_fusion_mdd", "mean_copy_risk"], ascending=[False, True])
    )
    display_table("feature group aggregate", grouped, max_rows=80)

    fig, axes = plt.subplots(1, 3, figsize=(19, 5), dpi=140)
    axes[0].barh(grouped["feature_group"], grouped["mean_copy_risk"])
    axes[0].axvline(1.0, color="black", linestyle="--")
    axes[0].set_xlabel("MAE / persistence MAE")
    axes[0].set_title("Point branch copy-risk by feature group")

    axes[1].barh(grouped["feature_group"], grouped["mean_fusion_mdd"] - grouped["mean_point_mdd"])
    axes[1].axvline(0.0, color="black", linestyle="--")
    axes[1].set_xlabel("fusion MDD - point-only MDD")
    axes[1].set_title("Risk gate MDD improvement")

    axes[2].scatter(
        grouped["mean_signal_share"],
        grouped["mean_fusion_return"],
        s=80,
        alpha=0.75,
    )
    for _, row in grouped.iterrows():
        axes[2].annotate(row["feature_group"], (row["mean_signal_share"], row["mean_fusion_return"]), fontsize=8)
    axes[2].set_xlabel("fusion active share")
    axes[2].set_ylabel("fusion cumulative return proxy")
    axes[2].set_title("Opportunity versus return proxy")
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.suptitle("12번 feature group decision summary")
    show_figure(fig)


def build_inline_report(args: argparse.Namespace, summary: pd.DataFrame, feature_groups: dict[str, list[str]]) -> str:
    lines = [
        "# 12번 feature guardrail fusion 실행 요약",
        "",
        "## 목적",
        "",
        "10번의 `balanced_composite` 점예측을 기본 토대로 쓰되, 11번의 `absolute_move` 위험 확률을 guardrail로 붙인다.",
        "이번 단계의 핵심은 모델 구조 추가가 아니라 코인 독립변수 조합을 바꾸었을 때 collapse와 MDD가 개선되는지 확인하는 것이다.",
        "",
        "## 왜 이 방향인가",
        "",
        "- 10번: 점예측은 아직 persistence를 이기지 못했지만, Huber보다 평탄화가 줄어든 안정 objective를 찾았다.",
        "- 11번: absolute-move 위험 확률은 AP/Brier 기준선보다 나아, 진입 제한이나 포지션 축소 guardrail로 쓸 수 있다.",
        "- 따라서 12번은 `feature group -> point branch -> risk branch -> decision policy` 순서로 어느 독립변수군을 데이터마트에 올릴지 고른다.",
        "",
        "## 실행 설정",
        "",
        "```json",
        json.dumps(vars(args), ensure_ascii=False, indent=2),
        "```",
        "",
        "## 사용 가능한 feature group",
        "",
        "```text",
        feature_group_table(feature_groups).to_string(index=False),
        "```",
        "",
    ]
    if not summary.empty:
        best = summary.sort_values(
            ["point_plus_risk_gate_mdd", "point_copy_risk_ratio"],
            ascending=[False, True],
        ).head(15)
        lines.extend(
            [
                "## 상위 후보",
                "",
                "```text",
                best.to_string(index=False),
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "## 해석 기준",
            "",
            "- `point_copy_risk_ratio < 1`이면 persistence보다 나은 점예측이다. 1보다 크더라도 10번 대비 낮아지면 독립변수 추가 효과가 있다.",
            "- `point_plus_risk_gate_mdd`가 `point_only_mdd`보다 덜 음수이면 위험 gate가 하방 방어에 도움을 준 것이다.",
            "- `fusion_signal_share`가 거의 0이면 MDD가 좋아도 거래를 안 해서 좋아진 착시일 수 있다.",
            "- 좋은 feature group은 copy-risk를 낮추거나, 같은 copy-risk에서도 risk gate 결합 후 MDD를 줄여야 한다.",
            "- 이 조건을 통과한 feature group만 이후 데이터마트 정식 스키마 후보로 승격한다.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="12번 feature guardrail fusion 실험")
    parser.add_argument("--db", default=None)
    parser.add_argument("--table", default="btc_15m_advance")
    parser.add_argument("--ticker", default=None)
    parser.add_argument("--profile", default="school_4090_15gb")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--parallel-slot", choices=sorted(obj.PARALLEL_RESOURCE_SLOTS), default="exclusive")
    parser.add_argument("--suite", choices=["feature_guardrail_matrix", "feature_ablation_quick"], default="feature_guardrail_matrix")
    parser.add_argument("--feature-groups", default=",".join(DEFAULT_FEATURE_GROUPS))
    parser.add_argument("--point-models", default=",".join(DEFAULT_POINT_MODELS))
    parser.add_argument("--risk-models", default=",".join(DEFAULT_RISK_MODELS))
    parser.add_argument("--preprocessings", default=",".join(DEFAULT_PREPROCESSINGS))
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--cross-tickers", default="KRW-ETH,KRW-XRP,KRW-SOL")
    parser.add_argument("--point-objective", default="balanced_composite")
    parser.add_argument("--risk-event-kind", choices=["absolute_move", "downside"], default="absolute_move")
    parser.add_argument("--risk-horizon", type=int, default=16)
    parser.add_argument("--risk-allow-quantile", type=float, default=0.55)
    parser.add_argument("--event-quantile", type=float, default=0.70)
    parser.add_argument("--normalization", default="window_standard")
    parser.add_argument("--optimizer", default="adamw")
    parser.add_argument("--scheduler", default="cosine")
    parser.add_argument("--gradient-policy", default="clip1")
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--objective-warmup-epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-rows", type=int, default=40000)
    parser.add_argument("--max-windows", type=int, default=4096)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--cost-bps", type=float, default=14.0)
    parser.add_argument("--min-signal-bps", type=float, default=3.0)
    parser.add_argument("--case-plots", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-failure", action="store_true")
    return parser.parse_known_args(argv)[0]


def main(argv: list[str] | None = None) -> None:
    configure_inline_matplotlib()
    args = parse_args(argv)
    profile = base.build_resource_profile(args.profile, args.device)
    profile, slot_details = obj.apply_parallel_resource_slot(profile, args.parallel_slot)
    base.apply_resource_profile(profile)
    environment = base.log_environment(profile)
    print("[environment]", json.dumps(environment, ensure_ascii=False, indent=2))
    print("[slot]", json.dumps(slot_details, ensure_ascii=False, indent=2))

    features, feature_groups, raw = load_feature_frame(args)
    cases = build_cases(args, feature_groups)
    if args.suite == "feature_ablation_quick":
        cases = cases[: min(len(cases), 12)]

    display_table("feature groups", feature_group_table(feature_groups), max_rows=100)
    print(f"[plan] suite={args.suite} cases={len(cases)}")
    print(pd.DataFrame(cases).head(80).to_string(index=False))
    print("[statistics]", json.dumps(base.basic_statistics(features), ensure_ascii=False, indent=2))
    display_table("missingness audit", obj.build_missingness_audit(raw, features, "1-step point / 4-hour risk labels"))

    if args.dry_run:
        print("[dry-run] 데이터 로드와 feature group 등록까지만 확인하고 종료한다.")
        return

    rows: list[dict[str, object]] = []
    base.apply_window_preprocessing = diag.apply_preprocessing_pipeline

    for index, case in enumerate(cases, start=1):
        print(f"\n[case {index}/{len(cases)}] {case}")
        try:
            point_result, val_pred, test_pred, point_splits = run_point_branch(case, features, profile, args)
            risk_result, val_probability, test_probability, risk_splits, risk_gate_cutoff = run_risk_branch(
                case,
                features,
                profile,
                args,
            )
            fusion_result, policy_frame = evaluate_fusion_policy(
                case,
                point_splits["test"],
                test_pred,
                test_probability,
                risk_gate_cutoff,
                args,
            )
            fusion_result.update(
                {
                    "risk_average_precision": risk_result["average_precision"],
                    "risk_average_precision_lift": risk_result["average_precision_lift"],
                    "risk_brier_skill_score": risk_result["brier_skill_score"],
                    "risk_expected_calibration_error": risk_result["expected_calibration_error"],
                    "risk_event_rate": risk_result["event_rate"],
                    "risk_train_event_rate": risk_result["train_event_rate"],
                }
            )
            rows.append(fusion_result)
            if args.case_plots:
                show_fusion_diagnostics(case, point_splits["test"], test_pred, test_probability, risk_gate_cutoff, policy_frame, args)
        except Exception as exc:
            if not args.continue_on_failure:
                raise
            print(f"[case failed] {case}: {exc}")
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    summary = pd.DataFrame(rows)
    show_feature_summary(summary)
    display_markdown(build_inline_report(args, summary, feature_groups))
    print("[done] all diagnostics displayed inline; no PNG/CSV/Markdown artifacts were written.")


if __name__ == "__main__":
    main()
