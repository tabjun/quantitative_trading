"""
CLI entrypoint for building realtime text independent variables.

Usage:
    uv run pipelines/ingest_text_context.py

Optional environment variables:
    TEXT_RSS_URLS      Pipe-separated RSS URLs.
    TEXT_LOCAL_CSVS    Pipe-separated CSV files with published_at,title,body,url.
    NAVER_CLIENT_ID    Optional Naver Search API client id.
    NAVER_CLIENT_SECRET Optional Naver Search API client secret.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import duckdb

from contexts.text_context import TextDataCollector, TextFeatureBuilder


def load_recent_price_index(db_path: str, table_name: str = "btc_15m_advance", limit: int = 500):
    with duckdb.connect(db_path) as con:
        return con.execute(
            f"""
            SELECT timestamp
            FROM {table_name}
            ORDER BY timestamp DESC
            LIMIT {limit}
            """
        ).df()


def write_feature_report(inserted_count: int, feature_count: int, db_path: str, output_dir: str = "test/results") -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    report_path = output_path / f"text_context_feature_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    report = f"""# Text Context Feature Build Report

## Summary
- Built at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- DuckDB path: `{db_path}`
- Raw text records inserted/refreshed: {inserted_count}
- 15-minute feature rows generated: {feature_count}

## Feature Tables
- `text_events_raw`: normalized news/report/SNS text with lexicon sentiment and topic flags.
- `text_features_15m`: candle-aligned independent variables for multivariate analysis.

## Core Independent Variables
- `text_event_count`: number of text events in the 15-minute bucket.
- `text_sentiment_mean`: average lexicon sentiment, from -1.0 to +1.0.
- `text_shock_z`: rolling 96-bucket z-score for abnormal text volume.
- `text_sentiment_momentum_1h`: 1-hour rolling sentiment mean.
- `text_macro_count`, `text_risk_count`, `text_crypto_count`, `text_regulation_count`, `text_liquidity_count`: topic exposure flags aggregated by bucket.
"""
    report_path.write_text(report, encoding="utf-8")
    return report_path


def main() -> None:
    db_path = "upbit_data.db"
    collector = TextDataCollector()
    records = collector.collect_all(max_items_per_source=30)
    builder = TextFeatureBuilder(db_path=db_path)
    inserted_count = builder.persist_raw_records(records)

    try:
        price_index = load_recent_price_index(db_path)
    except Exception as exc:
        print(f"[WARN] Could not load price index; building text-only feature buckets. ({exc})")
        price_index = None

    features = builder.build_and_persist_15m_features(price_index=price_index)
    report_path = write_feature_report(inserted_count, len(features), db_path)
    print(f"[SUCCESS] Inserted/refreshed {inserted_count} text records.")
    print(f"[SUCCESS] Generated {len(features)} 15-minute text feature rows.")
    print(f"[SUCCESS] Report saved to {report_path}")


if __name__ == "__main__":
    main()
