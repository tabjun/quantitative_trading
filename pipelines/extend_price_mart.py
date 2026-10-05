"""Upbit 가격 마트 연장 파이프라인(기존 DB는 읽기 전용, 결과는 새 DB 파일).

연구 종목군의 기존 캔들(`data/upbit_data.db`의 `upbit_krw_candle`)에, 기존 마지막 시각 이후부터
지금까지의 15분봉을 Upbit 공개 API로 이어 받아 새 DuckDB 파일에 합쳐 적재한다. 기존 결과의
재현성을 지키려고 원본 DB는 열기만 하고 쓰지 않는다.

- 이어 받는 구간은 기존 마지막 시각보다 `--overlap-days`일 앞에서 시작한다. 겹치는 구간의 종가가
  기존 값과 같은지 대조해, 시각 규약(KST, 캔들 시작 시각)이나 캔들 정정이 어긋나지 않았는지 확인한다.
- 수집 시점에 아직 끝나지 않은 봉(시작 + 15분 > 수집 시각)은 버린다.
- 겹치는 구간은 새로 받은 완성 봉으로 바꾼다. 2026-10-05 연장에서 불일치는 종목마다 기존 마지막 봉
  1개뿐이었다(기존 수집 시점에 진행 중이던 봉). 대조 결과는 `extend_log` 표로 남긴다.
- 기존 수집 이후 상장폐지된 종목은 API가 빈 응답을 주므로 기존 데이터만 그대로 옮긴다.

재현 실행 예시:
    .venv/bin/python pipelines/extend_price_mart.py --out data/upbit_data_20261005.db
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import pandas as pd
import pyupbit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "test" / "scripts"))
TABLE = "upbit_krw_candle"
BAR = timedelta(minutes=15)


def fetch_since(ticker: str, since: datetime, sleep: float) -> pd.DataFrame:
    """since 이후의 15분봉을 최신부터 거꾸로 200개씩 받는다."""
    frames, curr = [], datetime.now()
    while curr > since:
        df = pyupbit.get_ohlcv(ticker, interval="minute15", to=curr.strftime("%Y-%m-%d %H:%M:%S"), count=200)
        if df is None or df.empty:
            break
        frames.append(df)
        nxt = df.index[0].to_pydatetime()
        if nxt >= curr:
            break
        curr = nxt
        time.sleep(sleep)
    if not frames:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "value"])
    out = pd.concat(frames).sort_index()
    out = out[~out.index.duplicated(keep="first")]
    out = out[out.index >= since]
    out = out.reset_index().rename(columns={"index": "timestamp"})
    if "value" not in out.columns:
        out["value"] = out["close"] * out["volume"]
    return out[["timestamp", "open", "high", "low", "close", "volume", "value"]]


def main(argv=None) -> None:
    from report_header import study_universe
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "data" / "upbit_data.db"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--overlap-days", type=float, default=7.0)
    ap.add_argument("--sleep", type=float, default=0.12)
    a = ap.parse_args(argv)
    out_path = Path(a.out)
    if out_path.exists():
        raise FileExistsError(f"{out_path}가 이미 있다. 덮어쓰지 않는다")
    tickers, _ = study_universe()
    fetched_at = datetime.now()
    src = duckdb.connect(a.src, read_only=True)
    rows, checks = [], []
    for i, tk in enumerate(tickers, 1):
        old = src.execute(f"SELECT * FROM {TABLE} WHERE ticker=? ORDER BY timestamp", [tk]).df()
        last = old["timestamp"].max().to_pydatetime()
        new = fetch_since(tk, last - timedelta(days=a.overlap_days), a.sleep)
        new = new[pd.to_datetime(new["timestamp"]) + BAR <= fetched_at]
        ov = old.merge(new, on="timestamp", suffixes=("_old", "_new"))
        mism = int((ov["close_old"] != ov["close_new"]).sum())
        only_old = int((old["timestamp"] >= new["timestamp"].min()).sum() - len(ov)) if len(new) else 0
        bad_ts = ov.loc[ov["close_old"] != ov["close_new"], "timestamp"]
        add = new[new["timestamp"] > last].copy()
        add.insert(0, "ticker", tk)
        keep_old = old[~old["timestamp"].isin(new["timestamp"])] if len(new) else old
        ovr = new[new["timestamp"] <= last].copy()
        ovr.insert(0, "ticker", tk)
        rows.append(pd.concat([keep_old, ovr[old.columns], add[old.columns]], ignore_index=True)
                    .sort_values("timestamp"))
        checks.append({"종목": tk, "기존_끝": last, "새_끝": new["timestamp"].max() if len(new) else pd.NaT,
                       "추가봉": len(add), "겹침봉": len(ov), "겹침_종가불일치": mism,
                       "겹침_기존에만있는봉": only_old,
                       "불일치_시각": ",".join(str(t) for t in bad_ts), "API_빈응답": len(new) == 0})
        print(f"[extend] {i}/{len(tickers)} {tk}: +{len(add)}봉, 겹침 {len(ov)}봉(불일치 {mism}, 기존에만 {only_old})",
              flush=True)
    src.close()
    allrows = pd.concat(rows, ignore_index=True)
    with duckdb.connect(str(out_path)) as con:
        con.register("df", allrows)
        con.execute(f"CREATE TABLE {TABLE} AS SELECT * FROM df ORDER BY ticker, timestamp")
        ck =pd.DataFrame(checks)
        ck["수집시각"] = fetched_at
        con.register("ck", ck)
        con.execute("CREATE TABLE extend_log AS SELECT * FROM ck")
    print(pd.DataFrame(checks).to_string(index=False))
    print(f"[extend] 완료: {out_path} · 수집 시각 {fetched_at:%Y-%m-%d %H:%M:%S}")


if __name__ == "__main__":
    main()
