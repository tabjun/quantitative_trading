"""Upbit 가격 마트 재구축 파이프라인.

서버 홈 초기화 등으로 `data/upbit_data.db`가 사라졌을 때, Upbit 공개 API에서
과거 캔들을 다시 수집해 DuckDB 마트를 재구축한다. 2번 실험의 최초 수집 로직
(`fetch_and_save_3_years_data`)과 같은 스키마(timestamp, open, high, low, close,
volume, value)를 유지해 engine/data.py 로더와 호환된다.

재현 실행 예시:
    uv run pipelines/rebuild_price_mart.py                       # KRW-BTC 3년 15분봉 -> btc_15m_advance
    uv run pipelines/rebuild_price_mart.py --years 1 --table btc_15m_advance
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import pandas as pd
import pyupbit


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upbit 캔들 수집으로 DuckDB 가격 마트 재구축")
    parser.add_argument("--db", default=None, help="DuckDB 경로 (기본: repo/data/upbit_data.db)")
    parser.add_argument("--ticker", default="KRW-BTC")
    parser.add_argument("--interval", default="minute15")
    parser.add_argument("--table", default="btc_15m_advance")
    parser.add_argument("--years", type=float, default=3.0)
    parser.add_argument("--sleep", type=float, default=0.11, help="API 호출 간격(초), Upbit rate limit 보호")
    return parser.parse_known_args(argv)[0]


def fetch_history(ticker: str, interval: str, years: float, sleep: float) -> pd.DataFrame:
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * years)
    frames: list[pd.DataFrame] = []
    curr = end_date
    request_count = 0
    while curr > start_date:
        df = pyupbit.get_ohlcv(ticker, interval=interval, to=curr.strftime("%Y-%m-%d %H:%M:%S"), count=200)
        if df is None or df.empty:
            print(f"[rebuild-mart] 응답 없음/빈 응답으로 중단: to={curr}")
            break
        frames.append(df)
        curr = df.index[0]
        request_count += 1
        if request_count % 50 == 0:
            print(f"[rebuild-mart] {request_count} requests, 현재 커서 {curr}")
        time.sleep(sleep)
    if not frames:
        raise RuntimeError("수집된 캔들이 없다. 네트워크/티커를 확인하라.")
    full_df = pd.concat(frames).sort_index()
    full_df = full_df[~full_df.index.duplicated(keep="first")]
    full_df = full_df.reset_index().rename(columns={"index": "timestamp"})
    return full_df


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    db_path = Path(args.db) if args.db else repo_root / "data" / "upbit_data.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[rebuild-mart] {args.ticker} {args.interval} {args.years}년 수집 시작 -> {db_path}:{args.table}")
    full_df = fetch_history(args.ticker, args.interval, args.years, args.sleep)
    print(
        f"[rebuild-mart] 수집 완료: rows={len(full_df)}, "
        f"기간 {full_df['timestamp'].min()} ~ {full_df['timestamp'].max()}"
    )
    with duckdb.connect(str(db_path)) as con:
        con.register("candles_df", full_df)
        con.execute(f"CREATE OR REPLACE TABLE {args.table} AS SELECT * FROM candles_df")
        count = con.execute(f"SELECT COUNT(*) FROM {args.table}").fetchone()[0]
    print(f"[rebuild-mart] DuckDB 적재 완료: {args.table} rows={count}")


if __name__ == "__main__":
    main()
