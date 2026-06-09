"""Historical flow data mart for KRW-market analog retrieval.

This module builds reusable historical pattern statistics from price, volume,
independent-variable state, and optional context factors. It is designed for
Upbit KRW-wide candles, not a BTC-only workflow. Heavy full-market builds should
run on the intended server/automation environment; local Codex checks should
stay limited to syntax and tiny synthetic data.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import duckdb
import numpy as np
import pandas as pd
from fastdtw import fastdtw
from database.paths import resolve_db_path


SOURCE_TABLE = "upbit_krw_candle"
BTC_FALLBACK_TABLE = "btc_15m_advance"
WINDOW_TABLE = "historical_flow_windows"
SHAPE_TABLE = "historical_flow_shape_features"
FACTOR_TABLE = "historical_flow_factor_features"
CONTEXT_TABLE = "historical_flow_context_features"
FEATURE_TABLE = "historical_flow_features"
NEIGHBOR_TABLE = "historical_flow_neighbors"
EVENT_STATS_TABLE = "historical_flow_event_stats"
REGIME_TABLE = "historical_regime_stats"
RUN_LOG_TABLE = "historical_flow_run_log"

CONTEXT_COLUMNS = [
    "text_event_count",
    "text_sentiment_mean",
    "text_shock_z",
    "text_sentiment_momentum_1h",
    "text_risk_count",
    "text_macro_count",
    "text_crypto_count",
    "text_regulation_count",
    "text_liquidity_count",
]

FACTOR_VECTOR_COLUMNS = [
    "return_total",
    "realized_vol",
    "mdd_in_window",
    "trend_slope_per_hour",
    "value_z_last",
    "volume_z_last",
    "rsi_14_last",
    "roc_16_last",
    "bb_width_20_last",
    "amihud_illiq_mean",
]

CONTEXT_VECTOR_COLUMNS = [
    "context_event_count_mean",
    "context_sentiment_mean",
    "context_shock_z_max",
    "context_sentiment_momentum_1h",
    "context_risk_count_sum",
    "context_macro_count_sum",
    "context_crypto_count_sum",
    "context_regulation_count_sum",
    "context_liquidity_count_sum",
]


@dataclass(frozen=True)
class HistoricalFlowConfig:
    db_path: str = "data/upbit_data.db"
    source_table: str = SOURCE_TABLE
    interval: str = "minute15"
    window_lengths: tuple[int, ...] = (16, 48, 96, 288)
    stride: int = 4
    top_k: int = 10
    liquid_top_n: int = 50
    index_universe: str = "liquid-top"
    max_windows_per_ticker: int | None = None
    allow_btc_fallback: bool = True
    shape_weight: float = 0.50
    factor_weight: float = 0.30
    context_weight: float = 0.20


def parse_window_lengths(raw: str | Sequence[int]) -> tuple[int, ...]:
    if isinstance(raw, str):
        values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    else:
        values = [int(value) for value in raw]
    if not values or any(value < 4 for value in values):
        raise ValueError("window lengths must contain integers >= 4")
    return tuple(sorted(set(values)))


class HistoricalFlowMart:
    """Build and query ticker-aware historical flow statistics in DuckDB."""

    def __init__(self, config: HistoricalFlowConfig | None = None):
        self.config = config or HistoricalFlowConfig()
        if self.config.db_path:
            self.config = HistoricalFlowConfig(
                db_path=resolve_db_path(self.config.db_path),
                source_table=self.config.source_table,
                interval=self.config.interval,
                window_lengths=self.config.window_lengths,
                stride=self.config.stride,
                top_k=self.config.top_k,
                liquid_top_n=self.config.liquid_top_n,
                index_universe=self.config.index_universe,
                max_windows_per_ticker=self.config.max_windows_per_ticker,
                allow_btc_fallback=self.config.allow_btc_fallback,
                shape_weight=self.config.shape_weight,
                factor_weight=self.config.factor_weight,
                context_weight=self.config.context_weight,
            )

    def load_source_candles(self) -> pd.DataFrame:
        with duckdb.connect(self.config.db_path, read_only=True) as con:
            tables = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
            if self.config.source_table in tables:
                query = f"""
                    SELECT timestamp, ticker, open, high, low, close, volume, value
                    FROM {self.config.source_table}
                    WHERE ticker IS NOT NULL
                    ORDER BY ticker, timestamp
                """
                return con.execute(query).df()
            if self.config.allow_btc_fallback and BTC_FALLBACK_TABLE in tables:
                query = f"""
                    SELECT timestamp, 'KRW-BTC' AS ticker, open, high, low, close, volume, value
                    FROM {BTC_FALLBACK_TABLE}
                    ORDER BY timestamp
                """
                return con.execute(query).df()
        raise RuntimeError(
            f"Missing {self.config.source_table}. Build the full KRW candle mart first."
        )

    def build_from_candles(self, candles: pd.DataFrame) -> dict[str, int]:
        clean = self._attach_optional_context(self._prepare_candles(candles))
        liquid_tickers = self._select_liquid_tickers(clean)
        windows, feature_bundles, event_stats = self._build_windows_features_events(clean, liquid_tickers)
        regimes = self._build_regimes(feature_bundles)
        neighbors = self._build_neighbors(feature_bundles)
        self._persist(windows, feature_bundles, event_stats, regimes, neighbors, len(clean))
        return {
            "source_rows": len(clean),
            "window_rows": len(windows),
            "shape_rows": len(feature_bundles),
            "factor_rows": len(feature_bundles),
            "context_rows": len(feature_bundles),
            "event_stat_rows": len(event_stats),
            "regime_rows": len(regimes),
            "neighbor_rows": len(neighbors),
        }

    def query_similar_flows(
        self,
        ticker: str,
        window_length: int = 96,
        top_k: int | None = None,
        require_liquid_index: bool = True,
    ) -> pd.DataFrame:
        """Find historical analogs for the latest available ticker window."""
        top_k = top_k or self.config.top_k
        candles = self._attach_optional_context(self._prepare_candles(self.load_source_candles()))
        ticker_df = candles[candles["ticker"].eq(ticker)].sort_values("timestamp")
        if len(ticker_df) < window_length:
            raise ValueError(f"Not enough rows for {ticker} window_length={window_length}")

        query_window = ticker_df.tail(window_length)
        query_vector = self._window_feature_row(
            query_window,
            ticker=ticker,
            window_length=window_length,
            window_id="query",
            index_universe="query",
        )
        query_path = np.asarray(json.loads(query_vector["shape"]["return_path_json"]), dtype=float)
        query_factor = np.asarray(json.loads(query_vector["factor"]["factor_vector_json"]), dtype=float)
        query_context = np.asarray(json.loads(query_vector["context"]["context_vector_json"]), dtype=float)

        with duckdb.connect(self.config.db_path, read_only=True) as con:
            exists = self._relation_exists(con, FEATURE_TABLE)
            if not exists:
                return pd.DataFrame()
            where = "window_length = ? AND ticker <> ?"
            params: list[object] = [window_length, ticker]
            if require_liquid_index:
                where += " AND index_universe = 'liquid-top'"
            candidates = con.execute(
                f"""
                SELECT *
                FROM {FEATURE_TABLE}
                WHERE {where}
                ORDER BY window_end DESC
                """,
                params,
            ).df()

        if candidates.empty:
            return candidates

        scored_rows = []
        for row in candidates.to_dict("records"):
            candidate_path = np.asarray(json.loads(row["return_path_json"]), dtype=float)
            candidate_factor = np.asarray(json.loads(row["factor_vector_json"]), dtype=float)
            candidate_context = np.asarray(json.loads(row["context_vector_json"]), dtype=float)
            shape_distance, _ = fastdtw(query_path, candidate_path, dist=_point_distance)
            factor_distance = _safe_norm(query_factor - candidate_factor)
            context_distance = _safe_norm(query_context - candidate_context)
            composite_distance = self._composite_distance(shape_distance, factor_distance, context_distance)
            scored_rows.append(
                {
                    **row,
                    "query_dtw_distance": float(shape_distance),
                    "query_factor_distance": float(factor_distance),
                    "query_context_distance": float(context_distance),
                    "query_composite_distance": float(composite_distance),
                }
            )

        result = pd.DataFrame(scored_rows).sort_values("query_composite_distance").head(top_k)
        return result.reset_index(drop=True)

    def _prepare_candles(self, candles: pd.DataFrame) -> pd.DataFrame:
        required = {"timestamp", "ticker", "open", "high", "low", "close", "volume", "value"}
        missing = required.difference(candles.columns)
        if missing:
            raise ValueError(f"missing candle columns: {sorted(missing)}")
        df = candles.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp", "ticker", "close"])
        df = df[df["close"].gt(0)]
        df["value"] = df["value"].fillna(df["close"] * df["volume"].fillna(0.0))
        df = df.sort_values(["ticker", "timestamp"])
        df = df.drop_duplicates(["ticker", "timestamp"], keep="last")
        return df.reset_index(drop=True)

    def _attach_optional_context(self, candles: pd.DataFrame) -> pd.DataFrame:
        """Attach already-built text/context factors when available.

        The mart can run without context factors, but true historical analogs
        should use the same independent-variable state when the data exists.
        """
        df = candles.copy()
        try:
            with duckdb.connect(self.config.db_path, read_only=True) as con:
                if not self._relation_exists(con, "text_features_15m"):
                    for column in CONTEXT_COLUMNS:
                        df[column] = 0.0
                    return df
                context = con.execute(
                    """
                    SELECT timestamp,
                           text_event_count,
                           text_sentiment_mean,
                           text_shock_z,
                           text_sentiment_momentum_1h,
                           text_risk_count,
                           text_macro_count,
                           text_crypto_count,
                           text_regulation_count,
                           text_liquidity_count
                    FROM text_features_15m
                    """
                ).df()
        except duckdb.IOException:
            for column in CONTEXT_COLUMNS:
                df[column] = 0.0
            return df
        context["timestamp"] = pd.to_datetime(context["timestamp"], errors="coerce").dt.floor("15min")
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce").dt.floor("15min")
        df = df.merge(context, on="timestamp", how="left")
        for column in CONTEXT_COLUMNS:
            df[column] = df[column].fillna(0.0)
        return df

    def _select_liquid_tickers(self, candles: pd.DataFrame) -> set[str]:
        ranking = (
            candles.groupby("ticker", as_index=False)["value"]
            .sum()
            .sort_values("value", ascending=False)
        )
        return set(ranking.head(self.config.liquid_top_n)["ticker"].astype(str))

    def _build_windows_features_events(
        self, candles: pd.DataFrame, liquid_tickers: set[str]
    ) -> tuple[pd.DataFrame, list[dict[str, dict[str, object]]], pd.DataFrame]:
        window_rows: list[dict[str, object]] = []
        feature_bundles: list[dict[str, dict[str, object]]] = []
        event_rows: list[dict[str, object]] = []
        built_at = datetime.utcnow()

        for ticker, ticker_df in candles.groupby("ticker", sort=True):
            ticker_df = ticker_df.sort_values("timestamp").reset_index(drop=True)
            for window_length in self.config.window_lengths:
                if len(ticker_df) < window_length + 4:
                    continue
                starts = range(0, len(ticker_df) - window_length + 1, self.config.stride)
                if self.config.max_windows_per_ticker is not None:
                    starts = list(starts)[-self.config.max_windows_per_ticker :]
                for start_idx in starts:
                    end_idx = start_idx + window_length
                    window = ticker_df.iloc[start_idx:end_idx]
                    if len(window) != window_length:
                        continue
                    window_id = self._window_id(ticker, window_length, window.iloc[0]["timestamp"])
                    index_universe = "liquid-top" if ticker in liquid_tickers else "all-only"
                    feature = self._window_feature_row(
                        window,
                        ticker=ticker,
                        window_length=window_length,
                        window_id=window_id,
                        index_universe=index_universe,
                    )
                    window_rows.append(
                        {
                            "window_id": window_id,
                            "ticker": ticker,
                            "window_length": window_length,
                            "window_start": window.iloc[0]["timestamp"],
                            "window_end": window.iloc[-1]["timestamp"],
                            "start_close": float(window.iloc[0]["close"]),
                            "end_close": float(window.iloc[-1]["close"]),
                            "index_universe": index_universe,
                            "built_at": built_at,
                        }
                    )
                    feature_bundles.append(feature)
                    event_rows.append(
                        self._event_stats_row(ticker_df, ticker, window_id, end_idx - 1, window_length)
                    )

        return pd.DataFrame(window_rows), feature_bundles, pd.DataFrame(event_rows)

    def _window_feature_row(
        self,
        window: pd.DataFrame,
        ticker: str,
        window_length: int,
        window_id: str,
        index_universe: str,
    ) -> dict[str, dict[str, object]]:
        close = window["close"].astype(float).to_numpy()
        volume = window["volume"].fillna(0.0).astype(float).to_numpy()
        value = window["value"].fillna(0.0).astype(float).to_numpy()
        returns = np.diff(np.log(close), prepend=np.log(close[0]))
        return_path = np.cumsum(returns)
        return_path = return_path - return_path[0]
        volume_z = _zscore_path(volume)
        value_z = _zscore_path(value)
        running_peak = np.maximum.accumulate(close)
        drawdown = close / running_peak - 1.0
        elapsed_hours = max(window_length * 0.25, 0.25)
        trend_slope = (close[-1] / close[0] - 1.0) / elapsed_hours
        rsi_14 = _rsi(close, period=14)
        roc_16 = close[-1] / close[max(0, len(close) - 17)] - 1.0 if len(close) > 1 else 0.0
        bb_width_20 = _bb_width(close, period=20)
        amihud = np.abs(returns) / np.maximum(value, 1.0)
        context_summary = self._context_summary(window)
        shape_row = {
            "window_id": window_id,
            "ticker": ticker,
            "window_length": window_length,
            "window_start": window.iloc[0]["timestamp"],
            "window_end": window.iloc[-1]["timestamp"],
            "index_universe": index_universe,
            "return_total": float(close[-1] / close[0] - 1.0),
            "realized_vol": float(np.std(returns) * np.sqrt(max(window_length, 1))),
            "mdd_in_window": float(np.min(drawdown)),
            "trend_slope_per_hour": float(trend_slope),
            "return_path_json": json.dumps(return_path.round(8).tolist(), separators=(",", ":")),
            "volume_z_path_json": json.dumps(volume_z.round(8).tolist(), separators=(",", ":")),
            "value_z_path_json": json.dumps(value_z.round(8).tolist(), separators=(",", ":")),
            "updated_at": datetime.utcnow(),
        }
        factor_values = {
            "return_total": float(close[-1] / close[0] - 1.0),
            "realized_vol": float(np.std(returns) * np.sqrt(max(window_length, 1))),
            "mdd_in_window": float(np.min(drawdown)),
            "trend_slope_per_hour": float(trend_slope),
            "value_sum": float(np.sum(value)),
            "value_z_last": float(value_z[-1]) if len(value_z) else 0.0,
            "volume_z_last": float(volume_z[-1]) if len(volume_z) else 0.0,
            "rsi_14_last": float(rsi_14),
            "roc_16_last": float(roc_16),
            "bb_width_20_last": float(bb_width_20),
            "amihud_illiq_mean": float(np.mean(amihud)),
        }
        factor_vector = _standard_vector([factor_values[col] for col in FACTOR_VECTOR_COLUMNS if col in factor_values])
        factor_row = {
            "window_id": window_id,
            "ticker": ticker,
            "window_length": window_length,
            "window_start": window.iloc[0]["timestamp"],
            "window_end": window.iloc[-1]["timestamp"],
            "index_universe": index_universe,
            **factor_values,
            "factor_vector_json": json.dumps(factor_vector.round(8).tolist(), separators=(",", ":")),
            "updated_at": datetime.utcnow(),
            "event_type": _classify_event(
                return_total=float(close[-1] / close[0] - 1.0),
                mdd=float(np.min(drawdown)),
                vol=float(np.std(returns) * np.sqrt(max(window_length, 1))),
                shock_z=float(context_summary["context_shock_z_max"]),
                risk_count=float(context_summary["context_risk_count_sum"]),
                sentiment=float(context_summary["context_sentiment_mean"]),
            ),
        }
        context_row = {
            "window_id": window_id,
            "ticker": ticker,
            "window_length": window_length,
            "window_start": window.iloc[0]["timestamp"],
            "window_end": window.iloc[-1]["timestamp"],
            "index_universe": index_universe,
            **context_summary,
            "context_vector_json": json.dumps(
                _standard_vector([context_summary[col] for col in CONTEXT_VECTOR_COLUMNS]).round(8).tolist(),
                separators=(",", ":"),
            ),
            "updated_at": datetime.utcnow(),
        }
        return {
            "shape": shape_row,
            "factor": factor_row,
            "context": context_row,
            "combined": {**shape_row, **factor_row, **context_row},
        }

    def _context_summary(self, window: pd.DataFrame) -> dict[str, float]:
        for column in CONTEXT_COLUMNS:
            if column not in window.columns:
                window[column] = 0.0
        return {
            "context_event_count_mean": float(window["text_event_count"].mean()),
            "context_sentiment_mean": float(window["text_sentiment_mean"].mean()),
            "context_shock_z_max": float(window["text_shock_z"].max()),
            "context_sentiment_momentum_1h": float(window["text_sentiment_momentum_1h"].iloc[-1]),
            "context_risk_count_sum": float(window["text_risk_count"].sum()),
            "context_macro_count_sum": float(window["text_macro_count"].sum()),
            "context_crypto_count_sum": float(window["text_crypto_count"].sum()),
            "context_regulation_count_sum": float(window["text_regulation_count"].sum()),
            "context_liquidity_count_sum": float(window["text_liquidity_count"].sum()),
        }

    def _event_stats_row(
        self,
        ticker_df: pd.DataFrame,
        ticker: str,
        window_id: str,
        end_idx: int,
        window_length: int,
    ) -> dict[str, object]:
        close = ticker_df["close"].astype(float).to_numpy()
        end_close = close[end_idx]

        def fwd_return(steps: int) -> float | None:
            target_idx = end_idx + steps
            if target_idx >= len(close):
                return None
            return float(close[target_idx] / end_close - 1.0)

        future = close[end_idx + 1 : min(end_idx + 1 + 288, len(close))]
        future_mdd = None
        max_rebound = None
        if len(future):
            future_mdd = float(np.min(future / end_close - 1.0))
            max_rebound = float(np.max(future / end_close - 1.0))

        return {
            "window_id": window_id,
            "ticker": ticker,
            "window_length": window_length,
            "forward_return_1h": fwd_return(4),
            "forward_return_4h": fwd_return(16),
            "forward_return_24h": fwd_return(96),
            "forward_return_3d": fwd_return(288),
            "future_mdd_3d": future_mdd,
            "max_rebound_3d": max_rebound,
            "rebound_positive_24h": None if fwd_return(96) is None else int(fwd_return(96) > 0),
            "updated_at": datetime.utcnow(),
        }

    def _build_regimes(self, feature_bundles: list[dict[str, dict[str, object]]]) -> pd.DataFrame:
        if not feature_bundles:
            return pd.DataFrame()
        df = pd.DataFrame([bundle["combined"] for bundle in feature_bundles])
        vol_q = df["realized_vol"].quantile([0.33, 0.66]).to_dict()
        liquidity_q = df["value_sum"].quantile([0.33, 0.66]).to_dict()
        rows = []
        for row in df.to_dict("records"):
            trend = "bull" if row["return_total"] > 0.01 else "bear" if row["return_total"] < -0.01 else "range"
            vol = "low_vol" if row["realized_vol"] <= vol_q[0.33] else "high_vol" if row["realized_vol"] >= vol_q[0.66] else "mid_vol"
            liq = "low_liq" if row["value_sum"] <= liquidity_q[0.33] else "high_liq" if row["value_sum"] >= liquidity_q[0.66] else "mid_liq"
            rows.append(
                {
                    "window_id": row["window_id"],
                    "ticker": row["ticker"],
                    "window_length": row["window_length"],
                    "window_start": row["window_start"],
                    "window_end": row["window_end"],
                    "regime_label": f"{trend}_{vol}_{liq}",
                    "trend_bucket": trend,
                    "volatility_bucket": vol,
                    "liquidity_bucket": liq,
                    "updated_at": datetime.utcnow(),
                }
            )
        return pd.DataFrame(rows)

    def _build_neighbors(self, feature_bundles: list[dict[str, dict[str, object]]]) -> pd.DataFrame:
        if not feature_bundles:
            return pd.DataFrame()
        rows: list[dict[str, object]] = []
        indexed = [
            bundle["combined"]
            for bundle in feature_bundles
            if bundle["combined"]["index_universe"] == "liquid-top"
        ]
        if not indexed:
            return pd.DataFrame()
        indexed_by_length: dict[int, list[dict[str, object]]] = {}
        for row in indexed:
            indexed_by_length.setdefault(int(row["window_length"]), []).append(row)
        for window_length, group in indexed_by_length.items():
            group = sorted(group, key=lambda row: row["window_end"])
            path_vectors = np.vstack([json.loads(value) for value in [row["return_path_json"] for row in group]])
            factor_vectors = np.vstack([json.loads(value) for value in [row["factor_vector_json"] for row in group]])
            context_vectors = np.vstack([json.loads(value) for value in [row["context_vector_json"] for row in group]])
            metadata = group
            for idx, query in enumerate(metadata):
                path_distances = np.linalg.norm(path_vectors - path_vectors[idx], axis=1)
                factor_distances = np.linalg.norm(factor_vectors - factor_vectors[idx], axis=1)
                context_distances = np.linalg.norm(context_vectors - context_vectors[idx], axis=1)
                composite_distances = np.array(
                    [
                        self._composite_distance(path_distances[i], factor_distances[i], context_distances[i])
                        for i in range(len(path_distances))
                    ]
                )
                order = np.argsort(composite_distances)
                rank = 0
                for candidate_idx in order:
                    if candidate_idx == idx:
                        continue
                    candidate = metadata[candidate_idx]
                    if candidate["window_end"] >= query["window_start"]:
                        continue
                    rank += 1
                    rows.append(
                        {
                            "query_window_id": query["window_id"],
                            "neighbor_window_id": candidate["window_id"],
                            "query_ticker": query["ticker"],
                            "neighbor_ticker": candidate["ticker"],
                            "window_length": int(window_length),
                            "rank": rank,
                            "shape_distance": float(path_distances[candidate_idx]),
                            "factor_distance": float(factor_distances[candidate_idx]),
                            "context_distance": float(context_distances[candidate_idx]),
                            "composite_distance": float(composite_distances[candidate_idx]),
                            "method": "shape_factor_context_composite",
                            "index_universe": "liquid-top",
                            "updated_at": datetime.utcnow(),
                        }
                    )
                    if rank >= self.config.top_k:
                        break
        return pd.DataFrame(rows)

    def _composite_distance(self, shape_distance: float, factor_distance: float, context_distance: float) -> float:
        return (
            self.config.shape_weight * float(shape_distance)
            + self.config.factor_weight * float(factor_distance)
            + self.config.context_weight * float(context_distance)
        )

    def _persist(
        self,
        windows: pd.DataFrame,
        feature_bundles: list[dict[str, dict[str, object]]],
        event_stats: pd.DataFrame,
        regimes: pd.DataFrame,
        neighbors: pd.DataFrame,
        source_rows: int,
    ) -> None:
        shape_features = pd.DataFrame([bundle["shape"] for bundle in feature_bundles])
        factor_features = pd.DataFrame([bundle["factor"] for bundle in feature_bundles])
        context_features = pd.DataFrame([bundle["context"] for bundle in feature_bundles])
        db_path = Path(self.config.db_path)
        try:
            con_ctx = duckdb.connect(self.config.db_path)
        except duckdb.IOException as exc:
            if "not a valid DuckDB database file" not in str(exc):
                raise
            if db_path.exists():
                db_path.unlink()
            con_ctx = duckdb.connect(self.config.db_path)

        with con_ctx as con:
            for relation_name in [FEATURE_TABLE, SHAPE_TABLE, FACTOR_TABLE, CONTEXT_TABLE]:
                con.execute(f"DROP VIEW IF EXISTS {relation_name}")
                con.execute(f"DROP TABLE IF EXISTS {relation_name}")

            for table_name, frame in [
                (WINDOW_TABLE, windows),
                (SHAPE_TABLE, shape_features),
                (FACTOR_TABLE, factor_features),
                (CONTEXT_TABLE, context_features),
                (EVENT_STATS_TABLE, event_stats),
                (REGIME_TABLE, regimes),
                (NEIGHBOR_TABLE, neighbors),
            ]:
                con.execute(f"DROP TABLE IF EXISTS {table_name}")
                con.register("frame_df", frame)
                con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM frame_df")
                con.unregister("frame_df")

            con.execute(
                f"""
                CREATE VIEW {FEATURE_TABLE} AS
                SELECT
                    s.window_id,
                    s.ticker,
                    s.window_length,
                    s.window_start,
                    s.window_end,
                    s.index_universe,
                    s.return_total,
                    s.realized_vol,
                    s.mdd_in_window,
                    s.trend_slope_per_hour,
                    s.return_path_json,
                    s.volume_z_path_json,
                    s.value_z_path_json,
                    f.value_sum,
                    f.value_z_last,
                    f.volume_z_last,
                    f.rsi_14_last,
                    f.roc_16_last,
                    f.bb_width_20_last,
                    f.amihud_illiq_mean,
                    f.factor_vector_json,
                    f.event_type,
                    c.context_event_count_mean,
                    c.context_sentiment_mean,
                    c.context_shock_z_max,
                    c.context_sentiment_momentum_1h,
                    c.context_risk_count_sum,
                    c.context_macro_count_sum,
                    c.context_crypto_count_sum,
                    c.context_regulation_count_sum,
                    c.context_liquidity_count_sum,
                    c.context_vector_json,
                    GREATEST(s.updated_at, f.updated_at, c.updated_at) AS updated_at
                FROM {SHAPE_TABLE} s
                JOIN {FACTOR_TABLE} f USING (window_id, ticker, window_length, window_start, window_end, index_universe)
                JOIN {CONTEXT_TABLE} c USING (window_id, ticker, window_length, window_start, window_end, index_universe)
                """
            )

            run_row = pd.DataFrame(
                [
                    {
                        "run_id": datetime.utcnow().strftime("%Y%m%d_%H%M%S"),
                        "source_table": self.config.source_table,
                        "source_rows": source_rows,
                        "window_lengths": ",".join(map(str, self.config.window_lengths)),
                        "stride": self.config.stride,
                        "top_k": self.config.top_k,
                        "liquid_top_n": self.config.liquid_top_n,
                        "index_universe": self.config.index_universe,
                        "created_at": datetime.utcnow(),
                    }
                ]
            )
            con.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {RUN_LOG_TABLE} (
                    run_id VARCHAR,
                    source_table VARCHAR,
                    source_rows INTEGER,
                    window_lengths VARCHAR,
                    stride INTEGER,
                    top_k INTEGER,
                    liquid_top_n INTEGER,
                    index_universe VARCHAR,
                    created_at TIMESTAMP
                )
                """
            )
            con.register("run_df", run_row)
            con.execute(f"INSERT INTO {RUN_LOG_TABLE} SELECT * FROM run_df")
            con.unregister("run_df")

    def _relation_exists(self, con: duckdb.DuckDBPyConnection, relation_name: str) -> bool:
        return bool(
            con.execute(
                """
                SELECT COUNT(*)
                FROM information_schema.tables
                WHERE table_name = ?
                  AND table_type IN ('BASE TABLE', 'VIEW')
                """,
                [relation_name],
            ).fetchone()[0]
        )

    def _window_id(self, ticker: str, window_length: int, window_start: object) -> str:
        timestamp = pd.Timestamp(window_start).strftime("%Y%m%d%H%M%S")
        return f"{ticker}|{window_length}|{timestamp}"


def _zscore_path(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return values
    std = float(np.std(values))
    if std == 0.0 or np.isnan(std):
        return np.zeros_like(values, dtype=float)
    return (values - float(np.mean(values))) / std


def _safe_norm(values: np.ndarray) -> float:
    values = np.nan_to_num(values.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    return float(np.linalg.norm(values))


def _point_distance(left: object, right: object) -> float:
    return float(abs(float(np.asarray(left).squeeze()) - float(np.asarray(right).squeeze())))


def _standard_vector(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _rsi(close: np.ndarray, period: int = 14) -> float:
    if len(close) < 2:
        return 50.0
    deltas = np.diff(close)
    recent = deltas[-period:] if len(deltas) >= period else deltas
    gains = recent[recent > 0].sum() / max(len(recent), 1)
    losses = -recent[recent < 0].sum() / max(len(recent), 1)
    if losses == 0:
        return 100.0 if gains > 0 else 50.0
    rs = gains / losses
    return float(100.0 - (100.0 / (1.0 + rs)))


def _bb_width(close: np.ndarray, period: int = 20) -> float:
    recent = close[-period:] if len(close) >= period else close
    mean = float(np.mean(recent))
    if mean == 0.0:
        return 0.0
    return float((4.0 * np.std(recent)) / mean)


def _classify_event(
    return_total: float,
    mdd: float,
    vol: float,
    shock_z: float,
    risk_count: float,
    sentiment: float,
) -> str:
    if mdd <= -0.08 and (shock_z >= 2.0 or risk_count > 0 or sentiment < -0.2):
        return "context_confirmed_drawdown"
    if mdd <= -0.08:
        return "price_only_drawdown"
    if return_total >= 0.08 and sentiment >= 0.1:
        return "context_confirmed_breakout"
    if return_total >= 0.08:
        return "price_only_breakout"
    if vol >= 0.08 or shock_z >= 2.0:
        return "volatility_shock"
    return "ordinary_flow"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build KRW-wide historical flow DuckDB mart.")
    parser.add_argument("--db-path", default="data/upbit_data.db")
    parser.add_argument("--window-lengths", default="16,48,96,288")
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--liquid-top-n", type=int, default=50)
    parser.add_argument("--max-windows-per-ticker", type=int, default=None)
    parser.add_argument("--no-btc-fallback", action="store_true")
    parser.add_argument("--shape-weight", type=float, default=0.50)
    parser.add_argument("--factor-weight", type=float, default=0.30)
    parser.add_argument("--context-weight", type=float, default=0.20)
    return parser


def config_from_args(args: argparse.Namespace) -> HistoricalFlowConfig:
    return HistoricalFlowConfig(
        db_path=args.db_path,
        window_lengths=parse_window_lengths(args.window_lengths),
        stride=args.stride,
        top_k=args.top_k,
        liquid_top_n=args.liquid_top_n,
        max_windows_per_ticker=args.max_windows_per_ticker,
        allow_btc_fallback=not args.no_btc_fallback,
        shape_weight=args.shape_weight,
        factor_weight=args.factor_weight,
        context_weight=args.context_weight,
    )


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    mart = HistoricalFlowMart(config_from_args(args))
    candles = mart.load_source_candles()
    summary = mart.build_from_candles(candles)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
