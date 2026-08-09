"""Upbit 가격 마트 재구축 파이프라인.

서버 홈 초기화 등으로 `data/upbit_data.db`가 사라졌을 때, Upbit 공개 API에서
과거 캔들을 다시 수집해 DuckDB 마트를 재구축한다.

두 가지 모드:
1. 단일 종목 모드(기본): 2번 실험의 최초 수집 스키마(timestamp, open, high, low,
   close, volume, value)로 단일 테이블을 만든다. engine/data.py 로더와 호환.
2. 전체 KRW 마켓 모드(--tickers all 또는 콤마 목록): 연구 지침(2026-06-08,
   "BTC 단일 종목 금지, 업비트 KRW 마켓 전체 ticker 축")에 따라 ticker 컬럼을 포함한
   `upbit_krw_candle` 테이블을 만든다. 이 테이블명은 `marts/historical_flow.py`
   (과거 유사국면 데이터마트)가 기대하는 SOURCE_TABLE과 동일해, 수집 즉시
   `pipelines/build_historical_flow_mart.py`의 입력이 된다. 종목 단위 upsert
   (DELETE 후 INSERT)라 중단 후 --resume으로 이어서 수집할 수 있다.

재현 실행 예시:
    uv run pipelines/rebuild_price_mart.py                       # KRW-BTC 3년 15분봉 -> btc_15m_advance
    uv run pipelines/rebuild_price_mart.py --years 1 --table btc_15m_advance
    uv run pipelines/rebuild_price_mart.py --tickers all --years 3          # KRW 전 종목 -> upbit_krw_candle
    uv run pipelines/rebuild_price_mart.py --tickers all --resume           # 중단분 이어서
    uv run pipelines/rebuild_price_mart.py --tickers KRW-ETH,KRW-XRP --years 3
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import pandas as pd
import pyupbit

KRW_TABLE = "upbit_krw_candle"  # marts/historical_flow.py SOURCE_TABLE과 동일해야 한다.


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upbit 캔들 수집으로 DuckDB 가격 마트 재구축")
    parser.add_argument("--db", default=None, help="DuckDB 경로 (기본: repo/data/upbit_data.db)")
    parser.add_argument("--ticker", default="KRW-BTC", help="단일 종목 모드 대상")
    parser.add_argument("--tickers", default=None,
                        help="전체 KRW 마켓 모드: 'all'이면 KRW 전 종목, 아니면 콤마 목록. "
                             f"지정 시 ticker 컬럼 포함 {KRW_TABLE} 테이블에 적재한다.")
    parser.add_argument("--interval", default="minute15")
    parser.add_argument("--table", default=None,
                        help=f"대상 테이블 (기본: 단일 모드 btc_15m_advance, 멀티 모드 {KRW_TABLE})")
    parser.add_argument("--years", type=float, default=3.0)
    parser.add_argument("--sleep", type=float, default=0.11, help="API 호출 간격(초), Upbit rate limit 보호")
    parser.add_argument("--resume", action="store_true",
                        help="멀티 모드에서 이미 적재된 종목(행>0)은 건너뛴다")
    parser.add_argument("--min-rows", type=int, default=200,
                        help="이보다 적게 수집된 종목은 상장 직후로 보고 그대로 적재하되 경고를 남긴다")
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
        next_curr = df.index[0]
        # 진행 가드: 상장 시점보다 이전을 요청하면 API가 같은 구간을 반복 반환할 수 있다.
        # 커서가 전진(과거로 이동)하지 않으면 수집 완료로 보고 중단한다(무한 루프 방지).
        if next_curr >= curr:
            print(f"[rebuild-mart] 커서 정체({curr} -> {next_curr}) — 상장 이전 구간으로 판단, 수집 종료")
            break
        curr = next_curr
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


def ensure_krw_table(con: duckdb.DuckDBPyConnection, table: str) -> None:
    con.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {table} (
            ticker VARCHAR,
            timestamp TIMESTAMP,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume DOUBLE,
            value DOUBLE
        )
        """
    )


def collected_tickers(con: duckdb.DuckDBPyConnection, table: str) -> dict[str, int]:
    tables = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
    if table not in tables:
        return {}
    rows = con.execute(f"SELECT ticker, COUNT(*) FROM {table} GROUP BY ticker").fetchall()
    return {ticker: int(count) for ticker, count in rows}


def run_multi_ticker(args: argparse.Namespace, db_path: Path, table: str) -> None:
    if args.tickers.strip().lower() == "all":
        tickers = pyupbit.get_tickers(fiat="KRW")
    else:
        tickers = [item.strip() for item in args.tickers.split(",") if item.strip()]
    if not tickers:
        raise RuntimeError("수집할 KRW 종목 목록을 얻지 못했다.")
    print(f"[rebuild-mart] KRW 마켓 모드: {len(tickers)}종목 {args.interval} {args.years}년 -> {db_path}:{table}")

    with duckdb.connect(str(db_path)) as con:
        ensure_krw_table(con, table)
        already = collected_tickers(con, table) if args.resume else {}
        done, skipped, failed = 0, 0, []
        for index, ticker in enumerate(tickers, start=1):
            if args.resume and already.get(ticker, 0) > 0:
                skipped += 1
                continue
            started = time.time()
            try:
                frame = fetch_history(ticker, args.interval, args.years, args.sleep)
            except Exception as exc:  # noqa: BLE001 - 종목 단위 격리(신규 상장/거래정지 등)
                failed.append({"ticker": ticker, "error": str(exc)})
                print(f"[rebuild-mart] {ticker} 실패: {exc}")
                continue
            frame.insert(0, "ticker", ticker)
            if "value" not in frame.columns:
                frame["value"] = frame["close"] * frame["volume"]
            frame = frame[["ticker", "timestamp", "open", "high", "low", "close", "volume", "value"]]
            if len(frame) < args.min_rows:
                print(f"[rebuild-mart] 경고: {ticker} rows={len(frame)} (< {args.min_rows}) — 신규 상장/저이력 종목")
            con.register("candles_df", frame)
            con.execute(f"DELETE FROM {table} WHERE ticker = ?", [ticker])
            con.execute(f"INSERT INTO {table} SELECT * FROM candles_df")
            done += 1
            print(
                f"[rebuild-mart] {index}/{len(tickers)} {ticker}: rows={len(frame)}, "
                f"{frame['timestamp'].min()} ~ {frame['timestamp'].max()}, {time.time() - started:.0f}s"
            )
        total = con.execute(f"SELECT COUNT(*), COUNT(DISTINCT ticker) FROM {table}").fetchone()
    print(f"[rebuild-mart] 완료: 적재 {done}, 재개 스킵 {skipped}, 실패 {len(failed)}")
    if failed:
        print(f"[rebuild-mart] 실패 목록: {[f['ticker'] for f in failed]}")
    print(f"[rebuild-mart] {table}: 총 {total[0]}행 / {total[1]}종목")
    print("[all-collected]")


def run_single_ticker(args: argparse.Namespace, db_path: Path, table: str) -> None:
    print(f"[rebuild-mart] {args.ticker} {args.interval} {args.years}년 수집 시작 -> {db_path}:{table}")
    full_df = fetch_history(args.ticker, args.interval, args.years, args.sleep)
    print(
        f"[rebuild-mart] 수집 완료: rows={len(full_df)}, "
        f"기간 {full_df['timestamp'].min()} ~ {full_df['timestamp'].max()}"
    )
    with duckdb.connect(str(db_path)) as con:
        con.register("candles_df", full_df)
        con.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM candles_df")
        count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    print(f"[rebuild-mart] DuckDB 적재 완료: {table} rows={count}")
    print("[all-collected]")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    db_path = Path(args.db) if args.db else repo_root / "data" / "upbit_data.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    if args.tickers:
        run_multi_ticker(args, db_path, args.table or KRW_TABLE)
    else:
        run_single_ticker(args, db_path, args.table or "btc_15m_advance")


if __name__ == "__main__":
    main()
