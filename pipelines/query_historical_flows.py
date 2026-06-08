"""Query precomputed historical flow analogs from DuckDB."""

from __future__ import annotations

import argparse

from marts.historical_flow import HistoricalFlowConfig, HistoricalFlowMart


def main() -> None:
    parser = argparse.ArgumentParser(description="Query similar historical KRW-market flows.")
    parser.add_argument("--db-path", default="upbit_data.db")
    parser.add_argument("--ticker", required=True, help="Example: KRW-BTC, KRW-ETH, KRW-SOL")
    parser.add_argument("--window-length", type=int, default=96)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--include-all-index", action="store_true")
    args = parser.parse_args()

    mart = HistoricalFlowMart(HistoricalFlowConfig(db_path=args.db_path))
    result = mart.query_similar_flows(
        ticker=args.ticker,
        window_length=args.window_length,
        top_k=args.top_k,
        require_liquid_index=not args.include_all_index,
    )
    if result.empty:
        print("No historical flow mart rows found. Build the mart first.")
        return
    cols = [
        "ticker",
        "window_start",
        "window_end",
        "event_type",
        "return_total",
        "realized_vol",
        "mdd_in_window",
        "query_dtw_distance",
        "query_factor_distance",
        "query_context_distance",
        "query_composite_distance",
    ]
    print(result[cols].to_string(index=False))


if __name__ == "__main__":
    main()
