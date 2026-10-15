"""사전 변동성 국면 경계 테이블(서비스 2단계).

국면 정의는 연구 27d·26c와 같다: 예측 시점 직전 H분의 실현 변동성
RV = sqrt(Σ r²)(15분 로그수익률, 거래소 점검 봉과 점검 직후 첫 체결 봉까지 제외)를
그 종목의 학습 구간(2023-10-06 ~ 2025-11-11 분할 전) 값의 20·40·60·80% 분위수로 나눠
Q1(잔잔) ~ Q5(요동)로 둔다. 경계는 한 번 계산해 `regime_boundaries`에 버전과 함께 저장하고,
실시간에는 조회만 한다(docs/service_design.md §4). 연구 쪽 사전 등록이 경계를 바꾸면 새 버전을 쓴다.

계산식은 `test/models/26c_recent_twopart_test.py`의 build_from_close·horizon_data·halt_labels를
옮긴 것이고, 같은 값이 나오는지 20종목 × 5구간에서 대조했다(2026-10-15, 최대 차이는 보고서 참조).

    python -m marts.regime_boundaries            # 기본: 연구 창·분할로 계산해 data/llm_agent.db에 저장
"""

from __future__ import annotations

import argparse
from datetime import datetime

import duckdb
import numpy as np
import pandas as pd

from database.paths import resolve_db_path, resolve_service_db_path

BAR = pd.Timedelta("15min")
HORIZONS_MIN = (15, 30, 60, 240, 720)
HALT_REF = ("KRW-BTC", "KRW-ETH", "KRW-XRP")  # 셋이 동시에 빈 15분 라벨 = 거래소 전체 점검
WARMUP = pd.Timedelta(days=7)
# 연구 26c·27·28과 같은 창과 분할. 2026-10-06 이후(연구 확증 구간)는 경계 계산에 쓰지 않는다.
SOURCE_DB = "data/upbit_data_20261005.db"
DATA_START = pd.Timestamp("2023-10-06 00:00:00")
DATA_END = pd.Timestamp("2026-10-05 20:45:00")
SPLIT = pd.Timestamp("2025-11-11")
VERSION = "v1_26c_split20251111"
TABLE = "regime_boundaries"


def load_close(con: duckdb.DuckDBPyConnection, ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    d = con.execute(
        "SELECT timestamp, close FROM upbit_krw_candle WHERE ticker=? AND timestamp>=? AND timestamp<? ORDER BY timestamp",
        [ticker, start.to_pydatetime(), end.to_pydatetime()],
    ).df()
    if not len(d):
        raise LookupError(f"{ticker}: 창 안에 데이터가 없다")
    return pd.Series(d["close"].astype(float).clip(lower=1e-9).to_numpy(), index=pd.DatetimeIndex(d["timestamp"]))


def halt_labels(con: duckdb.DuckDBPyConnection, start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    missing = []
    for tk in HALT_REF:
        c = load_close(con, tk, start, end)
        g = pd.date_range(c.index.min(), c.index.max(), freq=BAR)
        missing.append(set(g.difference(c.index)))
    return pd.DatetimeIndex(sorted(set.intersection(*missing)))


def log_returns(close: pd.Series, halts: pd.DatetimeIndex) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """15분 격자 로그수익률. 점검 봉과 점검 뒤 첫 체결 봉까지는 NaN(26c build_from_close와 같음)."""
    close = close.sort_index()
    grid = pd.date_range(close.index.min(), close.index.max(), freq=BAR)
    c = close.reindex(grid)
    traded = c.notna().to_numpy()
    is_halt = grid.isin(halts) & ~traded
    cf = c.ffill().to_numpy(dtype=float)
    r = np.full(len(grid), np.nan)
    r[1:] = np.log(cf[1:] / cf[:-1])
    bad = is_halt.copy()
    bad[0] = True
    pending = False
    for i in range(len(grid)):
        if is_halt[i]:
            pending = True
        elif pending:
            bad[i] = True
            if traded[i]:
                pending = False
    r[bad] = np.nan
    return grid, r


def prior_rv(grid: pd.DatetimeIndex, r: np.ndarray, horizon_min: int) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """예측 시점 T마다 직전 H분 RV. 26c horizon_data처럼 [T-H, T+H) 안에 제외 봉이 없는 시점만 남긴다."""
    m = horizon_min // 15
    bad = ~np.isfinite(r)
    cs_r2 = np.concatenate([[0.0], np.cumsum(np.nan_to_num(r ** 2, nan=0.0))])
    cs_bad = np.concatenate([[0.0], np.cumsum(bad.astype(float))])
    j = np.arange(m, len(grid) - m + 1)
    j = j[(cs_bad[j + m] - cs_bad[j - m]) == 0]
    j = j[grid[j] >= grid[0] + WARMUP]
    return grid[j], np.sqrt(cs_r2[j] - cs_r2[j - m])


def compute_boundaries(market_db: str, split: pd.Timestamp = SPLIT) -> pd.DataFrame:
    rows = []
    with duckdb.connect(market_db, read_only=True) as con:
        tickers = [t for (t,) in con.execute("SELECT DISTINCT ticker FROM upbit_krw_candle ORDER BY ticker").fetchall()]
        halts = halt_labels(con, DATA_START, DATA_END)
        for tk in tickers:
            grid, r = log_returns(load_close(con, tk, DATA_START, DATA_END), halts)
            for h in HORIZONS_MIN:
                T, nai = prior_rv(grid, r, h)
                train = np.asarray(T + pd.Timedelta(minutes=h) <= split)
                q = np.quantile(nai[train], [0.2, 0.4, 0.6, 0.8])
                rows.append({
                    "version": VERSION, "ticker": tk, "horizon_min": h,
                    "q20": q[0], "q40": q[1], "q60": q[2], "q80": q[3],
                    "zero_q20": bool(q[0] == 0.0),  # 가격 정지가 많아 Q1 경계가 0인 종목(27d 1-1)
                    "n_train": int(train.sum()), "train_start": T[0], "split": split,
                    "source_db": market_db, "computed_at": datetime.now(),
                })
    return pd.DataFrame(rows)


def assign_regime(prior_rv_value: float, row: pd.Series) -> int:
    """직전 RV 하나를 국면 번호 1~5(Q1~Q5)로. 27d의 np.digitize(nai, 경계)+1과 같다."""
    return int(np.digitize([prior_rv_value], [row["q20"], row["q40"], row["q60"], row["q80"]])[0]) + 1


def save(df: pd.DataFrame, service_db: str) -> None:
    with duckdb.connect(service_db) as con:
        con.register("b", df)
        con.execute(f"CREATE TABLE IF NOT EXISTS {TABLE} AS SELECT * FROM b WHERE FALSE")
        con.execute(f"DELETE FROM {TABLE} WHERE version = ?", [VERSION])
        con.execute(f"INSERT INTO {TABLE} SELECT * FROM b")
        con.unregister("b")


def main() -> None:
    ap = argparse.ArgumentParser(description="사전 변동성 국면 경계를 계산해 서비스 DB에 저장한다.")
    ap.add_argument("--market-db", default=SOURCE_DB)
    ap.add_argument("--service-db", default=None)
    args = ap.parse_args()
    df = compute_boundaries(resolve_db_path(args.market_db))
    save(df, resolve_service_db_path(args.service_db))
    print(f"{VERSION}: {df['ticker'].nunique()}종목 × {len(HORIZONS_MIN)}구간 = {len(df)}행 저장, "
          f"Q1 경계 0인 칸 {int(df['zero_q20'].sum())}개")


if __name__ == "__main__":
    main()
