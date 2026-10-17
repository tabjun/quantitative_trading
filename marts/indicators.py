"""2단계 보조 지표 스냅숏: 종목별 사전 국면, GARCH-t 예측 RV, 가격 정지 비율(CPU만).

입력 시세 = 스냅숏 DB(읽기 전용, 연구 창 시작부터) + 서비스 DB `candles_live_15m`(스냅숏 이후 판단 시점까지).
기준 시각(as_of) = 마지막으로 닫힌 15분 봉의 종가 시각. 예측 대상은 그 다음 H분(as_of ~ as_of + H).
값마다 출처 모델·적합 구간 끝·데이터 끝·예측 시각을 붙여 `indicator_snapshots`에 JSON으로 저장한다.

GARCH-t는 연구 26c(`garch_t_fit`·`garch_multistep`·`var_to_rv`)와 같은 절차다: 주기 스케일(요일×시각 168칸 RMS 비율,
학습 구간으로 추정)로 나눈 15분 수익률 × 100에 GARCH(1,1)-t(평균 0)를 학습 구간(분할 2025-11-11 전)으로 한 번
적합하고, 그 뒤는 모수를 고정해 필터만 한다. 다단계 분산을 주기 스케일로 되돌려 창 RV로 바꾼다.
**연구와 다른 점** (1) 내부 검증 보정 계수 c와 가격 정지 두 부분 모형(π)을 쓰지 않는다. 그래서 수준이 연구 최종
예측과 다를 수 있고, 출력에 `calibration: "없음"`으로 표시한다. (2) 수익률 척도를 ×100이 아니라 ×1000으로 적합한다.
×100에서는 입력 표준편차가 약 0.25라 arch 최적화가 초깃값(α=0.1, β=0.88) 근처에서 5회 만에 멈추는 종목이 있다
(2026-10-18 진단: 연구 26c 20종목 중 AQT·BTC·MOC·STRAX·XLM·XRP 6종목, 같은 척도 로그우도가 BTC 132.6·MOC 625.3 낮음).
척도만 바꾼 것이라 모형은 같다(ω만 척도² 배). 적합 뒤 모수가 초깃값과 같으면 `fit_suspect`로 표시한다.
이 값의 정확도는 계산하지 않는다(누설 규칙).

    python -m marts.indicators            # 상장 중인 연구 종목 전부, 순차
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime

import duckdb
import numpy as np
import pandas as pd

from database.paths import resolve_db_path, resolve_service_db_path
from marts import live_candles
from marts import regime_boundaries as rb

SNAPSHOT_DB = "data/upbit_data_20261012.db"
TABLE = "indicator_snapshots"
FORECAST_MIN = (60, 240, 720)  # 보조 지표 구간 1h·4h·12h
REGIME_MIN = (60, 240, 720)
GARCH_SCALE = 1000.0  # 연구 26c는 100. 위 모듈 설명의 (2) 참조


def load_close(ticker: str, snapshot_db: str, service_db: str) -> pd.Series:
    """스냅숏(연구 창 시작부터) 뒤에 실시간 봉을 이어 붙인다. 겹치는 시각은 스냅숏 값을 쓴다."""
    with duckdb.connect(snapshot_db, read_only=True) as con:
        snap = rb.load_close(con, ticker, rb.DATA_START, pd.Timestamp("2100-01-01"))
    live = live_candles.load_live(service_db, ticker, start=snap.index.max() + rb.BAR)
    if len(live):
        snap = pd.concat([snap, pd.Series(live["close"].astype(float).clip(lower=1e-9).to_numpy(),
                                          index=pd.DatetimeIndex(live["timestamp"]))])
    return snap[~snap.index.duplicated(keep="first")].sort_index()


def halt_labels(snapshot_db: str, service_db: str) -> pd.DatetimeIndex:
    missing = []
    for tk in rb.HALT_REF:
        c = load_close(tk, snapshot_db, service_db)
        g = pd.date_range(c.index.min(), c.index.max(), freq=rb.BAR)
        missing.append(set(g.difference(c.index)))
    return pd.DatetimeIndex(sorted(set.intersection(*missing)))


def periodic_scale(grid: pd.DatetimeIndex, r: np.ndarray, split: pd.Timestamp, slots: pd.Index | None = None) -> np.ndarray:
    """26c build_from_close의 주기 스케일: 요일×시각 슬롯별 학습 구간 RMS / 전체 RMS의 제곱근."""
    slot = (grid.dayofweek * 24 + grid.hour).to_numpy()
    tr = (grid < split) & np.isfinite(r)
    r2 = r ** 2
    slot_ms = pd.Series(r2[tr]).groupby(slot[tr]).mean()
    overall = float(np.mean(r2[tr]))
    table = np.sqrt((slot_ms.reindex(np.arange(168)).fillna(overall) / overall).to_numpy())
    idx = slot if slots is None else np.asarray(slots)
    return table[idx]


def garch_t_forecast(grid: pd.DatetimeIndex, r: np.ndarray, split: pd.Timestamp) -> dict:
    from arch import arch_model

    cfac = periodic_scale(grid, r, split)
    d = r / cfac
    ok = np.isfinite(d)
    gc, dc = grid[ok], d[ok]
    split_c = int(np.searchsorted(gc, split))
    x = dc * GARCH_SCALE
    res = arch_model(x[:split_c], mean="Zero", vol="GARCH", p=1, q=1, dist="t").fit(disp="off", show_warning=False)
    p = res.params
    om, al, be = float(p["omega"]), float(p["alpha[1]"]), float(p["beta[1]"])
    suspect = res.convergence_flag != 0 or (abs(al - 0.1) < 1e-6 and abs(be - 0.88) < 1e-6)
    cv = np.asarray(res.conditional_volatility) ** 2
    h = cv[split_c - 1]
    for xi in x[split_c - 1:]:  # 학습 끝 이후는 모수 고정 필터. 마지막 반복이 다음 봉(as_of 이후 첫 봉)의 분산
        h = om + al * xi ** 2 + be * h
    phi = al + be
    out = {}
    last = gc[-1]
    for H in FORECAST_MIN:
        m = H // 15
        k = np.arange(m)
        geo = k.astype(float) if abs(phi - 1) < 1e-12 else (1 - phi ** k) / (1 - phi)
        hp = om * geo + phi ** k * h  # 봉별 주기 제거 분산(척도² 단위)
        fut = pd.date_range(last + rb.BAR, periods=m, freq=rb.BAR)
        c2 = periodic_scale(grid, r, split, slots=pd.Index(fut.dayofweek * 24 + fut.hour)) ** 2
        out[f"{H}m"] = float(np.sqrt(np.sum(hp * c2)) / GARCH_SCALE)
    return {"rv_forecast": out, "params": {"omega": om, "alpha": al, "beta": be, "nu": float(p.get("nu", np.nan))},
            "n_fit": split_c, "fit_suspect": bool(suspect), "iterations": int(res.optimization_result.nit),
            "loglik": float(res.loglikelihood)}


def regime_now(grid: pd.DatetimeIndex, r: np.ndarray, bounds: pd.DataFrame) -> dict:
    """마지막 닫힌 봉까지의 직전 H분 RV와 국면(Q1~Q5). 창 안에 점검 등 제외 봉이 있으면 국면을 정하지 않는다."""
    out = {}
    for H in REGIME_MIN:
        m = H // 15
        w = r[-m:]
        row = bounds[bounds["horizon_min"] == H]
        if row.empty:
            out[f"{H}m"] = {"prior_rv": None, "regime": None, "note": "경계 없음"}
            continue
        if not np.all(np.isfinite(w)):
            out[f"{H}m"] = {"prior_rv": None, "regime": None, "note": "창 안에 제외 봉(점검 등)"}
            continue
        rv = float(np.sqrt(np.sum(w ** 2)))
        out[f"{H}m"] = {"prior_rv": rv, "regime": f"Q{rb.assign_regime(rv, row.iloc[0])}",
                        "zero_q20": bool(row.iloc[0]["zero_q20"])}
    return out


def stall_ratio(close: pd.Series, grid: pd.DatetimeIndex, r: np.ndarray) -> dict:
    """가격 정지: 직전 1일·7일 봉 중 수익률이 정확히 0인 비율과 체결 없는 봉(캔들 없음) 비율."""
    traded = grid.isin(close.index)
    out = {}
    for name, n in (("1d", 96), ("7d", 672)):
        rr, tt = r[-n:], traded[-n:]
        fin = np.isfinite(rr)
        out[name] = {"zero_return_share": float(np.mean(rr[fin] == 0)) if fin.any() else None,
                     "no_trade_share": float(1 - np.mean(tt))}
    return out


def build_snapshot(
    ticker: str, snapshot_db: str, service_db: str, halts: pd.DatetimeIndex, bounds: pd.DataFrame, end_bar: pd.Timestamp
) -> dict:
    """end_bar = 거래소 전체의 마지막 닫힌 봉(BTC 기준). 저유동 종목도 이 봉까지 격자를 늘려 기준 시각을 맞춘다."""
    close = load_close(ticker, snapshot_db, service_db)
    grid, r = rb.log_returns(close, halts, end=end_bar)
    as_of = grid[-1] + rb.BAR  # 마지막 닫힌 봉의 종가 시각
    g = garch_t_forecast(grid, r, rb.SPLIT)
    return {
        "ticker": ticker,
        "as_of": str(as_of),
        "data_end_bar": str(grid[-1]),
        "last_trade_bar": str(close.index.max()),
        "regime": {"version": rb.VERSION, "values": regime_now(grid, r, bounds[bounds["ticker"] == ticker])},
        "garch_t": {
            "source": "GARCH(1,1)-t, 주기 스케일 보정, 연구 26c 절차(척도 ×1000)",
            "fit_end": str(rb.SPLIT), "fit_start": str(grid[0]), "n_fit": g["n_fit"],
            "calibration": "없음(연구의 보정 계수 c·두 부분 모형 π 미적용)",
            "params": g["params"], "rv_forecast": g["rv_forecast"], "forecast_from": str(as_of),
            "fit_suspect": g["fit_suspect"], "iterations": g["iterations"],
        },
        "price_stall": stall_ratio(close, grid, r),
        "timesfm": {"status": "보류", "note": "이번 세션 GPU는 llm_runtime 경유 단발 호출만 허용"},
        "created_at": str(datetime.now()),
    }


def save(snaps: list[dict], service_db: str) -> None:
    df = pd.DataFrame({"as_of": [pd.Timestamp(s["as_of"]) for s in snaps], "ticker": [s["ticker"] for s in snaps],
                       "payload": [json.dumps(s, ensure_ascii=False) for s in snaps],
                       "created_at": [pd.Timestamp(s["created_at"]) for s in snaps]})
    with duckdb.connect(service_db) as con:
        con.register("s", df)
        con.execute(f"CREATE TABLE IF NOT EXISTS {TABLE} AS SELECT * FROM s WHERE FALSE")
        con.execute(f"DELETE FROM {TABLE} t USING s WHERE t.ticker = s.ticker AND t.as_of = s.as_of")
        con.execute(f"INSERT INTO {TABLE} SELECT * FROM s")
        con.unregister("s")


def main() -> None:
    from pipelines.ingest_text_context import listed_names, study_universe

    ap = argparse.ArgumentParser(description="종목별 보조 지표 스냅숏을 만들어 서비스 DB에 저장한다(CPU, 순차).")
    ap.add_argument("--snapshot-db", default=SNAPSHOT_DB)
    ap.add_argument("--service-db", default=None)
    ap.add_argument("--tickers", default=None, help="쉼표 구분. 없으면 상장 중인 연구 종목 전부")
    args = ap.parse_args()
    service_db = resolve_service_db_path(args.service_db)
    snapshot_db = resolve_db_path(args.snapshot_db)
    tickers = args.tickers.split(",") if args.tickers else list(listed_names(study_universe(service_db))[0])
    with duckdb.connect(service_db, read_only=True) as con:
        bounds = con.execute("SELECT * FROM regime_boundaries WHERE version = ?", [rb.VERSION]).df()
    halts = halt_labels(snapshot_db, service_db)
    end_bar = load_close("KRW-BTC", snapshot_db, service_db).index.max()
    snaps = []
    for tk in tickers:
        s = build_snapshot(tk, snapshot_db, service_db, halts, bounds, end_bar)
        snaps.append(s)
        rg = {k: v["regime"] for k, v in s["regime"]["values"].items()}
        print(tk, s["as_of"], "마지막 체결", s["last_trade_bar"][11:16], rg, {k: round(v, 4) for k, v in s["garch_t"]["rv_forecast"].items()},
              "정지1d", round(s["price_stall"]["1d"]["zero_return_share"], 2), "의심" if s["garch_t"]["fit_suspect"] else "", flush=True)
    save(snaps, service_db)
    print(f"{len(snaps)}종목 스냅숏 저장")


if __name__ == "__main__":
    main()
