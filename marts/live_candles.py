"""실시간 15분봉(서비스 2단계 입력).

시세 스냅숏 DB(`data/upbit_data_YYYYMMDD.db`, 주 1회 생성, 읽기 전용)는 마지막 생성 시각에서 끝나므로, 판단 시점까지의
봉은 업비트 공개 API에서 받아 서비스 DB `candles_live_15m`에 쌓는다. 시각은 스냅숏과 같은 KST 봉 시작 시각이다
(pyupbit `candle_date_time_kst`). 아직 닫히지 않은 봉(시작 + 15분 > 수집 시각)은 버린다. 요청은 종목마다 순차로 보낸다.

    python -m marts.live_candles --count 800      # 연구 종목군 중 상장 종목의 최근 800봉(약 8.3일)
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime

import duckdb
import pandas as pd

from database.paths import resolve_service_db_path

TABLE = "candles_live_15m"
BAR = pd.Timedelta("15min")


def fetch_ticker(ticker: str, count: int) -> pd.DataFrame:
    import pyupbit

    fetched_at = datetime.now()  # 서버 시간대 = KST
    df = pyupbit.get_ohlcv(ticker, interval="minute15", count=count)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.reset_index().rename(columns={"index": "timestamp"})
    df = df[pd.to_datetime(df["timestamp"]) + BAR <= fetched_at]  # 닫힌 봉만
    df.insert(0, "ticker", ticker)
    df["fetched_at"] = fetched_at
    return df[["ticker", "timestamp", "open", "high", "low", "close", "volume", "value", "fetched_at"]]


def save(frames: list[pd.DataFrame], service_db: str) -> int:
    new = pd.concat([f for f in frames if len(f)], ignore_index=True)
    if new.empty:
        return 0
    with duckdb.connect(service_db) as con:
        con.register("n", new)
        con.execute(f"CREATE TABLE IF NOT EXISTS {TABLE} AS SELECT * FROM n WHERE FALSE")
        con.execute(f"DELETE FROM {TABLE} t USING n WHERE t.ticker = n.ticker AND t.timestamp = n.timestamp")
        con.execute(f"INSERT INTO {TABLE} SELECT * FROM n")
        con.unregister("n")
    return len(new)


def load_live(service_db: str, ticker: str, start: pd.Timestamp | None = None) -> pd.DataFrame:
    with duckdb.connect(service_db, read_only=True) as con:
        if TABLE not in {r[0] for r in con.execute("SHOW TABLES").fetchall()}:
            return pd.DataFrame()
        q = f"SELECT timestamp, close FROM {TABLE} WHERE ticker = ?" + (" AND timestamp >= ?" if start is not None else "")
        args = [ticker] + ([start.to_pydatetime()] if start is not None else [])
        return con.execute(q + " ORDER BY timestamp", args).df()


def main() -> None:
    from pipelines.ingest_text_context import listed_names, study_universe

    ap = argparse.ArgumentParser(description="연구 종목군의 최근 15분봉을 서비스 DB에 쌓는다.")
    ap.add_argument("--count", type=int, default=800)
    ap.add_argument("--service-db", default=None)
    args = ap.parse_args()
    service_db = resolve_service_db_path(args.service_db)
    names, dropped = listed_names(study_universe(service_db))
    frames = []
    for tk in names:
        frames.append(fetch_ticker(tk, args.count))
        time.sleep(0.2)
    n = save(frames, service_db)
    span = pd.concat([f for f in frames if len(f)])
    print(f"{len(names)}종목(제외 {dropped}) {n}봉 저장, 범위 {span['timestamp'].min()} ~ {span['timestamp'].max()}")


if __name__ == "__main__":
    main()
