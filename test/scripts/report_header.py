"""보고서 표준 헤더 생성기 (재사용 도구, AGENTS.md 2.9g).

모든 분석·모델링 보고서가 자기완결적이도록 6항목 헤더를 자동 생성한다.
데이터 조건을 첫 보고서에만 쓰고 생략하면, 나중 보고서만 봐서는 어떤 데이터를 어떻게
다뤄서 그 결과가 나왔는지 파악할 수 없다(2026-09-07 사용자 지적).

사용:
    from test.scripts.report_header import render_standard_header, top_tickers
    md = render_standard_header(tickers=top_tickers(20), train_frac=0.7,
                                 transforms=[("주기제거", "시간대별 변동성 주기가 지속성을 왜곡")])
    print(md)

CLI 확인:
    uv run test/scripts/report_header.py --top 20
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import stats as st

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "upbit_data.db"
SOURCE_TABLE = "upbit_krw_candle"
BARS_PER_DAY = 96
MIN_ROWS_FOR_TOP = 90_000  # 약 2.5년 이상 이력(신규 상장·생존편의 배제)


def _con():
    if not DB_PATH.exists():
        raise SystemExit(f"DB가 없다: {DB_PATH}")
    return duckdb.connect(str(DB_PATH), read_only=True)


def num(v, d: int = 4) -> str:
    """과학적표기 없이 포맷(AGENTS.md 2.9b)."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return "n/a" if not np.isfinite(f) else f"{f:,.{d}f}"


def top_tickers(n: int = 20, min_rows: int = MIN_ROWS_FOR_TOP) -> list[str]:
    """변동성 × 거래대금 상위 n종목(긴 이력 한정). 20번·21번과 동일 기준."""
    con = _con()
    df = con.execute(f"""
        with r as (
            select ticker,
                   ln(close / lag(close) over (partition by ticker order by timestamp)) lr,
                   value
            from {SOURCE_TABLE}
        )
        select ticker, count(lr) n, stddev_samp(lr) vol, sum(value) sv
        from r where lr is not null and isfinite(lr)
        group by ticker having count(lr) >= {int(min_rows)}
    """).df()
    con.close()
    df["score"] = df["vol"] * np.log(df["sv"].clip(lower=1))
    return df.sort_values("score", ascending=False).head(n)["ticker"].tolist()


def ticker_profile(tickers: list[str]) -> pd.DataFrame:
    """종목별 기간·봉수·기초통계량."""
    con = _con()
    rows = []
    for t in tickers:
        d = con.execute(
            f"select timestamp, close from {SOURCE_TABLE} where ticker=? order by timestamp", [t]
        ).df()
        if len(d) < 100:
            continue
        c = d["close"].astype(float).clip(lower=1e-9).to_numpy()
        r = np.diff(np.log(c))
        rows.append({
            "종목": t,
            "시작": str(d["timestamp"].iloc[0])[:16],
            "종료": str(d["timestamp"].iloc[-1])[:16],
            "봉수": len(d),
            "평균": r.mean(),
            "표준편차": r.std(),
            "왜도": float(st.skew(r)),
            "초과첨도": float(st.kurtosis(r)),
        })
    con.close()
    return pd.DataFrame(rows)


def data_quality(ticker: str) -> dict:
    """연속성·결측 점검(15분 간격 위반 건수)."""
    con = _con()
    d = con.execute(
        f"select timestamp, open, high, low, close, volume from {SOURCE_TABLE} "
        f"where ticker=? order by timestamp", [ticker]
    ).df()
    con.close()
    dt = d["timestamp"].diff().dt.total_seconds().div(60).dropna()
    return {
        "n": len(d),
        "start": str(d["timestamp"].iloc[0])[:16],
        "end": str(d["timestamp"].iloc[-1])[:16],
        "gaps": int((dt != 15).sum()),
        "gap_pct": float((dt != 15).mean() * 100),
        "missing": int(d[["open", "high", "low", "close", "volume"]].isna().sum().sum()),
    }


def split_ranges(ticker: str, train_frac: float) -> dict:
    """시간순 분할의 실제 날짜 구간."""
    con = _con()
    d = con.execute(
        f"select timestamp from {SOURCE_TABLE} where ticker=? order by timestamp", [ticker]
    ).df()
    con.close()
    n = len(d)
    s = int(n * train_frac)
    return {
        "n": n, "n_train": s, "n_val": n - s,
        "train": f"{str(d['timestamp'].iloc[0])[:16]} ~ {str(d['timestamp'].iloc[s-1])[:16]}",
        "val": f"{str(d['timestamp'].iloc[s])[:16]} ~ {str(d['timestamp'].iloc[-1])[:16]}",
        "train_days": s / BARS_PER_DAY, "val_days": (n - s) / BARS_PER_DAY,
    }


def series_stats(x: np.ndarray) -> dict:
    """변환 전후 비교용 기초통계량 + 핵심 검정."""
    from statsmodels.stats.diagnostic import het_arch
    from statsmodels.tsa.stattools import adfuller, kpss
    x = np.asarray(x)[np.isfinite(x)]
    sub = x[-40000:] if len(x) > 40000 else x
    out = {
        "n": len(x), "평균": x.mean(), "표준편차": x.std(),
        "왜도": float(st.skew(x)), "초과첨도": float(st.kurtosis(x)),
        "최소": x.min(), "최대": x.max(),
    }
    try:
        out["ADF p"] = float(adfuller(sub, autolag="AIC")[1])
    except Exception:
        out["ADF p"] = np.nan
    try:
        out["KPSS p"] = float(kpss(sub, regression="c", nlags="auto")[1])
    except Exception:
        out["KPSS p"] = np.nan
    try:
        out["ARCH p"] = float(het_arch(sub - sub.mean(), nlags=16)[1])
    except Exception:
        out["ARCH p"] = np.nan
    return out


def render_standard_header(
    tickers: list[str],
    train_frac: float = 0.70,
    rep_ticker: str = "KRW-BTC",
    transforms: list[tuple[str, str]] | None = None,
    transformed_series: dict[str, np.ndarray] | None = None,
    rows_used: int | None = None,
) -> str:
    """AGENTS.md 2.9g 표준 헤더 6항목을 마크다운으로 생성.

    transforms          : [(처리명, 적용 이유)] — 이번 회차에 새로 적용한 처리
    transformed_series  : {"원계열": r, "처리후": x} — 변환 전후 통계 재확인용
    rows_used           : 비용 제한으로 일부만 썼다면 그 행 수
    """
    L: list[str] = []
    prof = ticker_profile(tickers)
    q = data_quality(rep_ticker)
    sp = split_ranges(rep_ticker, train_frac)

    L.append("## 분석 조건 (표준 헤더)")
    L.append("")
    L.append("> 이 보고서만 읽어도 데이터 조건을 파악할 수 있도록, 매 회차 동일 항목을 싣는다"
             "(AGENTS.md 2.9g).")
    L.append("")

    # 1. 데이터 출처·형식
    L.append("### 1. 데이터 출처·형식")
    L.append("")
    L.append(f"- **DB / 테이블**: `data/upbit_data.db` (DuckDB) / `{SOURCE_TABLE}`")
    L.append(f"- **봉 간격**: 15분봉 (하루 {BARS_PER_DAY}봉)")
    L.append(f"- **원본 컬럼**: timestamp, open, high, low, close, volume, value")
    L.append(f"- **분석 종목 {len(tickers)}개** (변동성 × 거래대금 상위, 이력 {MIN_ROWS_FOR_TOP:,}봉 이상):")
    if rep_ticker not in tickers:
        L.append(f"- **대표 종목은 `{rep_ticker}`** — 저변동 대형 종목이라 변동성 상위 목록에는 들지 "
                 f"않지만, 기존 연구와의 연결·심층 진단을 위해 별도로 사용한다(총 {len(tickers)+1}종목).")
    L.append("")
    L.append("| # | 종목 | 봉 수 | 기간 | 표준편차 | 왜도 | 초과첨도 |")
    L.append("| ---: | :--- | ---: | :--- | ---: | ---: | ---: |")
    for i, r in prof.iterrows():
        L.append(f"| {i+1} | {r['종목']} | {int(r['봉수']):,} | {r['시작'][:10]}~{r['종료'][:10]} | "
                 f"{num(r['표준편차'],6)} | {num(r['왜도'])} | {num(r['초과첨도'],2)} |")
    L.append("")

    # 2. 수집 기간·규모
    L.append("### 2. 수집 기간·규모·품질 (대표 종목 " + rep_ticker + ")")
    L.append("")
    L.append(f"- **기간**: {q['start']} ~ {q['end']}")
    L.append(f"- **총 봉 수**: {q['n']:,}개 (약 {q['n']/BARS_PER_DAY/365:.2f}년)")
    L.append(f"- **연속성**: 15분이 아닌 간격 {q['gaps']}개 ({num(q['gap_pct'],4)}%) — 코인은 24시간 "
             f"거래라 장 마감·주말 갭이 없다")
    L.append(f"- **결측**: {q['missing']}개")
    if rows_used:
        L.append(f"- **이번 분석 사용량**: 최근 {rows_used:,}봉 (적합 비용 제한)")
    L.append("")

    # 3. 학습/검증 분할
    L.append("### 3. 학습/검증 분할")
    L.append("")
    L.append(f"- **방식**: 시간순 분할(셔플 없음) — 미래 정보 누설 방지")
    L.append(f"- **비율**: train {train_frac*100:.0f}% / validation {(1-train_frac)*100:.0f}%")
    L.append(f"- **train**: {sp['train']} ({sp['n_train']:,}봉, 약 {sp['train_days']:.0f}일)")
    L.append(f"- **validation**: {sp['val']} ({sp['n_val']:,}봉, 약 {sp['val_days']:.0f}일)")
    L.append(f"- 스케일러·모델 파라미터는 **train 구간에서만** 적합한다.")
    L.append("")

    # 4. 기본 전처리·가정
    L.append("### 4. 기본 전처리 파이프라인과 가정")
    L.append("")
    L.append("```")
    L.append("원본 close (KRW)")
    L.append("  → 로그 변환 log(close)        [스케일 안정화. 여전히 비정상: 단위근]")
    L.append("  → 1차 차분 r = Δlog(close)    [★ 여기서 정상성 확보 — 분석의 출발점]")
    L.append("  → (회차별 추가 처리)")
    L.append("  → 모델 적합")
    L.append("```")
    L.append("")
    L.append("- **왜 차분하는가**: 로그가격은 단위근이 있어 비정상(ADF p=0.19)이고, 1차 차분하면 "
             "정상이 된다(ADF p=0.000). 정상성은 로그가 아니라 **차분**에서 온다.")
    L.append("- **예측 대상**: 수익률의 부호(방향)가 아니라 **크기(변동성)**. 방향은 자기상관이 "
             "0에 가까워 예측이 어렵고, 크기는 장기기억이 있어 예측 가능하다.")
    L.append("")

    # 5. 기초통계량 · EDA 특성
    L.append("### 5. 기초통계량과 데이터 특성 (이전 EDA 확인 사항)")
    L.append("")
    if transformed_series and "원계열" in transformed_series:
        s0 = series_stats(transformed_series["원계열"])
        L.append("| 항목 | 값 |")
        L.append("| :--- | ---: |")
        for k in ("n", "평균", "표준편차", "왜도", "초과첨도", "최소", "최대", "ADF p", "KPSS p", "ARCH p"):
            d = 0 if k == "n" else (8 if k == "평균" else 4)
            L.append(f"| {k} | {num(s0[k], d)} |")
        L.append("")
    L.append("- **조건부 이분산**: ARCH-LM p=0.000 → 변동성이 시간에 따라 변한다(GARCH 계열 사용 근거)")
    L.append("- **장기기억**: |수익률| 자기상관이 하루(96봉) 뒤에도 0.12로 완만히 감소")
    L.append("- **두꺼운 꼬리**: 초과첨도 107 (정규분포는 0). 좌우 대칭(왜도 0.004)이나 양쪽 꼬리가 "
             "모두 두껍다. 4σ 초과가 정규 대비 약 100배, 6σ는 약 87만배 빈발")
    L.append("- **선형성 기각**: RESET·BDS 검정 p=0.000 → 선형 모델을 쓸 근거가 없다(비선형·커널 계열 필요)")
    L.append("- **방향 무예측성**: 수익률 자기상관 ≈ 0")
    L.append("")

    # 6. 이번 회차 변환
    L.append("### 6. 이번 회차에 적용한 변환")
    L.append("")
    if transforms:
        L.append("| 처리 | 적용 이유 |")
        L.append("| :--- | :--- |")
        for name, why in transforms:
            L.append(f"| {name} | {why} |")
        L.append("")
    else:
        L.append("- 추가 변환 없음(기본 파이프라인만 적용).")
        L.append("")
    if transformed_series and len(transformed_series) > 1:
        L.append("**변환 전후 비교**")
        L.append("")
        keys = list(transformed_series.keys())
        stats_list = [series_stats(v) for v in transformed_series.values()]
        L.append("| 항목 | " + " | ".join(keys) + " |")
        L.append("| :--- | " + " | ".join(["---:"] * len(keys)) + " |")
        for k in ("표준편차", "왜도", "초과첨도", "ADF p", "KPSS p", "ARCH p"):
            L.append(f"| {k} | " + " | ".join(num(s[k], 4) for s in stats_list) + " |")
        L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--train-frac", type=float, default=0.70)
    a = ap.parse_args()
    print(render_standard_header(top_tickers(a.top), a.train_frac))


if __name__ == "__main__":
    main()
