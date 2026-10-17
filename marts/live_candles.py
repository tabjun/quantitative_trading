"""실시간 15분봉(서비스 2단계 입력).

시세 스냅숏 DB(`data/upbit_data_YYYYMMDD.db`, 주 1회 생성, 읽기 전용)는 마지막 생성 시각에서 끝나므로, 판단 시점까지의
봉은 업비트 공개 API에서 받아 서비스 DB `candles_live_15m`에 쌓는다. 시각은 스냅숏과 같은 KST 봉 시작 시각이다
(pyupbit `candle_date_time_kst`). 아직 닫히지 않은 봉(시작 + 15분 > 수집 시각)은 버린다. 요청은 종목마다 순차로 보낸다.

    python -m marts.live_candles --count 800      # 연구 종목군 중 상장 종목의 최근 800봉(약 8.3일)
"""

from __future__ import annotations

import logging
import argparse
import time
from datetime import datetime

import duckdb
import pandas as pd

from common.db import WriteCount, write_changed
from common.errors import DataSourceError, each
from database.paths import resolve_service_db_path

TABLE = "candles_live_15m"
BAR = pd.Timedelta("15min")
log = logging.getLogger(__name__)



def fetch_ticker(ticker: str, count: int) -> pd.DataFrame:
    import pyupbit

    fetched_at = datetime.now()  # 서버 시간대 = KST
    try:
        df = pyupbit.get_ohlcv(ticker, interval="minute15", count=count)
    except (OSError, ValueError) as exc:
        raise DataSourceError(f"{ticker} 시세 요청 실패: {exc}") from exc
    if df is None or df.empty:  # pyupbit는 요청 실패를 None으로 돌려준다
        raise DataSourceError(f"{ticker} 시세 응답이 비었다")
    df = df.reset_index().rename(columns={"index": "timestamp"})
    df = df[pd.to_datetime(df["timestamp"]) + BAR <= fetched_at]  # 닫힌 봉만
    df.insert(0, "ticker", ticker)
    df["fetched_at"] = fetched_at
    return df[["ticker", "timestamp", "open", "high", "low", "close", "volume", "value", "fetched_at"]]


def save(frames: list[pd.DataFrame], service_db: str) -> WriteCount:
    """이미 있는 봉은 값이 같으면 건너뛰고, 거래소가 정정한 봉만 바꾸고, 새 봉만 넣는다."""
    return write_changed(service_db, TABLE, pd.concat(frames, ignore_index=True), keys=["ticker", "timestamp"],
                         compare=["open", "high", "low", "close", "volume", "value"])


def load_live(service_db: str, ticker: str, start: pd.Timestamp | None = None) -> pd.DataFrame:
    with duckdb.connect(service_db, read_only=True) as con:
        if TABLE not in {r[0] for r in con.execute("SHOW TABLES").fetchall()}:
            return pd.DataFrame()
        q = f"SELECT timestamp, close FROM {TABLE} WHERE ticker = ?" + (" AND timestamp >= ?" if start is not None else "")
        args = [ticker] + ([start.to_pydatetime()] if start is not None else [])
        return con.execute(q + " ORDER BY timestamp", args).df()


def fetch_all(service_db: str, count: int = 800) -> dict:
    """연구 종목군 중 상장 종목의 최근 count봉을 순차로 받아 저장한다."""
    from pipelines.ingest_text_context import listed_names, study_universe

    names, dropped = listed_names(study_universe(service_db))

    def fetch(tk: str) -> pd.DataFrame:
        time.sleep(0.2)  # 순차 요청 간격(공개 API)
        return fetch_ticker(tk, count)

    log.info("시세 수집 시작: %d종목 × 최근 %d봉", len(names), count)
    got = each(names, fetch)
    if not got.ok:
        raise DataSourceError(f"모든 종목 시세 수집 실패: {got.failed}")
    span = pd.concat(got.ok.values())
    w = save(list(got.ok.values()), service_db)
    return {"tickers": len(got.ok), "dropped": dropped, "failed": got.failed, "bars": len(span),
            "inserted": w.inserted, "updated": w.updated, "unchanged": w.unchanged,
            "first": str(span["timestamp"].min()), "last": str(span["timestamp"].max())}


def main() -> None:
    ap = argparse.ArgumentParser(description="연구 종목군의 최근 15분봉을 서비스 DB에 쌓는다.")
    ap.add_argument("--count", type=int, default=800)
    ap.add_argument("--service-db", default=None)
    args = ap.parse_args()
    r = fetch_all(resolve_service_db_path(args.service_db), args.count)
    print(f"{r['tickers']}종목(제외 {r['dropped']}) {r['bars']}봉: 새로 {r['inserted']}, 바뀜 {r['updated']}, 같음 {r['unchanged']}, 범위 {r['first']} ~ {r['last']}")


if __name__ == "__main__":
    main()
