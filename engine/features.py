"""코인 특화 feature 엔지니어링과 feature group 등록.

12번에서 옮겼다. `add_coin_specific_features`/`add_optional_text_features`/
`add_cross_market_features`/`register_feature_groups`/`load_feature_frame`/
`feature_group_table`와 보조 helper `safe_divide`/`columns_by_keywords`를 담는다.

12번의 `base.*` indirection은 engine.data / engine.resources로 연결한다.
`register_feature_groups`는 12번처럼 공유 `resources.FEATURE_SETS` dict를 mutate한다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import data, resources

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
    "coin_text_context": "뉴스/텍스트 context mart가 있을 때만 감성, shock, topic count를 붙여 등록한다. 텍스트 컬럼이 없으면 이 그룹은 등록하지 않는다(OHLCV fallback을 텍스트로 둔갑시키지 않음).",
    "coin_cross_market": "다중 ticker 테이블이 있을 때 ETH/XRP/SOL 등 cross-asset return을 붙인다. 없으면 자동 제외된다.",
    "coin_macro_proxy": "DXY, VIX, 금리, 주가지수, 원/달러, 금·유가 같은 macro/cross-asset 컬럼이 있을 때만 위험자산 레짐 proxy로 쓴다.",
    "coin_onchain_proxy": "active address, exchange flow, whale, SOPR, MVRV, NVT 같은 on-chain 컬럼이 있을 때만 코인 고유 수급 proxy로 쓴다.",
    "coin_derivatives_proxy": "funding rate, open interest, basis, liquidation, long/short ratio 컬럼이 있을 때만 레버리지·청산 압력 proxy로 쓴다.",
    "coin_search_social_dev": "Google Trends, YouTube, Twitter/X, Reddit, Telegram, GitHub activity 같은 관심도·커뮤니티·개발자 활동 컬럼이 있을 때만 쓴다.",
    "coin_full_available": "현재 데이터에서 쓸 수 있는 모든 코인 특화 feature를 합친 상한선 후보이다.",
}


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
    df["range_z_96"] = data.rolling_z((high - low) / close.replace(0, np.nan), 96)
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
    df["price_z_192"] = data.rolling_z(close, 192)
    df["volume_z_192"] = data.rolling_z(np.log1p(volume), 192)
    df["value_z_192"] = data.rolling_z(np.log1p(value), 192)
    df["range_mean_192"] = df["hl_range_pct"].rolling(192, min_periods=48).mean()
    df["trend_strength_192"] = close / close.rolling(192, min_periods=48).mean() - 1.0
    df["abs_return_z_96"] = data.rolling_z(ret.abs(), 96)
    df["volume_shock_z_96"] = data.rolling_z(np.log1p(volume), 96)
    df["value_shock_z_96"] = data.rolling_z(np.log1p(value), 96)
    df["range_shock_z_96"] = data.rolling_z(df["hl_range_pct"], 96)
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
        # fix②: 텍스트 컬럼이 없으면 coin_text_context를 OHLCV로 둔갑시켜 등록하던 것을 막는다.
        # OHLCV fallback이 "텍스트 포함" 그룹으로 비교되면 변수 기여도 분석이 오염된다.
        # 텍스트 mart가 실제로 있을 때만 이 그룹을 인정한다(다른 optional 그룹과 동일 규칙).
        if name == "coin_text_context" and not text_columns:
            continue
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
            resources.FEATURE_SETS[name] = available
    return available_groups


def load_feature_frame(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, list[str]], pd.DataFrame]:
    db_path = data.resolve_db_path(args.db)
    raw = data.load_price_data(db_path, args.table, args.ticker, args.max_rows)
    features = data.make_features(raw)
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
