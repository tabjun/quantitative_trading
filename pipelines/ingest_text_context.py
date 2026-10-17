"""
CLI entrypoint for building realtime text independent variables.

Usage:
    python -m pipelines.ingest_text_context [--lookback-days 7] [--no-tickers]

연구 종목군(서비스 DB `regime_boundaries`의 종목, 없으면 시세 DB의 종목) 가운데 지금 업비트에 상장된 종목마다
한국어 이름 검색어로 구글 뉴스 RSS를 받아 종목 태그를 단다. 일반 검색어는 MARKET 태그다. 요청은 순차로 보낸다.
15분 특징은 실시간 격자(마지막으로 닫힌 KST 봉까지 lookback_days)에 맞춘다. 시각 규약은
`contexts/text_context.py` 모듈 설명을 따른다.

Optional environment variables:
    TEXT_RSS_URLS      Pipe-separated RSS URLs.
    TEXT_LOCAL_CSVS    Pipe-separated CSV files with published_at,title,body,url(,available_at,ticker_hint).
    NAVER_CLIENT_ID    Optional Naver Search API client id.
    NAVER_CLIENT_SECRET Optional Naver Search API client secret.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import duckdb

from contexts.text_context import (
    DEFAULT_LOOKBACK_DAYS,
    TextDataCollector,
    TextFeatureBuilder,
    fetch_upbit_korean_names,
)
from database.paths import resolve_db_path, resolve_service_db_path


def study_universe(service_db_path: str, market_db: str = "data/upbit_data_20261012.db") -> list[str]:
    """국면 경계를 만든 연구 종목군. 경계 테이블이 없으면 시세 스냅숏의 종목."""
    if Path(service_db_path).exists():
        with duckdb.connect(service_db_path, read_only=True) as con:
            tables = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
            if "regime_boundaries" in tables:
                return [t for (t,) in con.execute("SELECT DISTINCT ticker FROM regime_boundaries ORDER BY 1").fetchall()]
    with duckdb.connect(resolve_db_path(market_db), read_only=True) as con:
        return [t for (t,) in con.execute("SELECT DISTINCT ticker FROM upbit_krw_candle ORDER BY 1").fetchall()]


def listed_names(universe: list[str]) -> tuple[dict[str, str], list[str]]:
    """(종목 → 한국어 이름, 상장 목록에 없어 뺀 종목)."""
    names = fetch_upbit_korean_names()
    return {tk: names[tk] for tk in universe if tk in names}, [tk for tk in universe if tk not in names]


def write_feature_report(
    inserted_count: int, feature_count: int, db_path: str, per_ticker: dict[str, int], dropped: list[str],
    off_topic: dict[str, int], failed_sources: dict[str, str], output_dir: str = "logs/text_context",
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    report_path = output_path / f"text_context_feature_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    lines = "\n".join(f"- {tk}: {n} (제목에 종목이 없어 버림 {off_topic.get(tk, 0)})" for tk, n in sorted(per_ticker.items()))
    report = f"""# Text Context Feature Build Report

## Summary
- Built at (KST): {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- DuckDB path: `{db_path}`
- Raw text records inserted/refreshed: {inserted_count}
- 15-minute feature rows generated: {feature_count}
- 상장 목록에 없어 뺀 종목: {", ".join(dropped) or "해당 없음(전부 상장 중)"}
- 수집 실패 출처: {"; ".join(f"{k}: {v}" for k, v in failed_sources.items()) or "해당 없음(모든 출처 성공)"}

## 이번 수집 기사 수(태그별)
{lines or "- 해당 없음(수집 0건)"}

## Feature Tables
- `text_events_raw`: 기사 원문·어휘 감성·주제 표시. 시각은 KST(published_at 발행, available_at 처음 수집).
- `text_features_15m`: (ticker, KST 15분 라벨)별 특징. 라벨 T = max(발행, 수집)이 속한 봉, T+15분에 알려진 정보.
"""
    report_path.write_text(report, encoding="utf-8")
    return report_path


def main() -> None:
    ap = argparse.ArgumentParser(description="종목별·시장 뉴스를 수집해 서비스 DB에 저장하고 15분 특징을 만든다.")
    ap.add_argument("--lookback-days", type=int, default=DEFAULT_LOOKBACK_DAYS)
    ap.add_argument("--max-items", type=int, default=30, help="검색어 하나당 최대 기사 수")
    ap.add_argument("--no-tickers", action="store_true", help="종목별 검색 없이 시장 검색어만")
    args = ap.parse_args()

    r = ingest(resolve_service_db_path(), args.lookback_days, args.max_items, args.no_tickers)
    print(f"[SUCCESS] Inserted/refreshed {r['inserted']} text records ({r['ticker_queries']} ticker queries, dropped {r['dropped']}).")
    print(f"[SUCCESS] Generated {r['feature_rows']} 15-minute text feature rows.")
    print(f"[SUCCESS] Report saved to {r['report']}")


def ingest(service_db_path: str, lookback_days: int = DEFAULT_LOOKBACK_DAYS, max_items: int = 30, no_tickers: bool = False) -> dict:
    """종목별·시장 뉴스 수집 → 저장 → 15분 특징. 건수와 보고서 경로를 돌려준다."""
    names, dropped = ({}, []) if no_tickers else listed_names(study_universe(service_db_path))
    collector = TextDataCollector(ticker_names=names, lookback_days=lookback_days)
    records = collector.collect_all(max_items_per_source=max_items)
    per_ticker: dict[str, int] = {tk: 0 for tk in [*names, "MARKET"]}  # 0건 종목도 보고서에 남긴다
    for record in records:
        per_ticker[record.ticker_hint] = per_ticker.get(record.ticker_hint, 0) + 1

    builder = TextFeatureBuilder(db_path=service_db_path)
    inserted_count = builder.persist_raw_records(records)
    features = builder.build_and_persist_15m_features(lookback_days=lookback_days)
    report_path = write_feature_report(
        inserted_count, len(features), service_db_path, per_ticker, dropped, collector.dropped_off_topic,
        collector.failed_sources,
    )
    return {"inserted": inserted_count, "ticker_queries": len(names), "dropped": dropped,
            "feature_rows": len(features), "report": str(report_path), "per_ticker": per_ticker,
            "failed": collector.failed_sources}


if __name__ == "__main__":
    main()
