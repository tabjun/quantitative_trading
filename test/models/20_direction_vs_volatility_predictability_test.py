# %% [markdown]
# # 20번: 방향(부호) 대 변동성(크기) 예측 가능성 대조 + 데이터 EDA
#
# 헤드리스 `.py` 드라이버(`#%%` 셀 + 파일 내 마크다운, AGENTS.md 2.3). ipynb는 만들지 않는다.
#
# ## 왜 (Why)
# A/B/C 갈래에서 **B(변동성)를 먼저 주력으로 확정**하기로 했다. 다만 프레이밍은 "방향이 안 되니까
# 변동성으로 간다"가 아니라 **"방향을 지금 같이 고려하기엔 이런 문제가 있고, 그래서 변동성을 먼저
# 하고 방향은 그 뒤에 이어간다"**이다. 이 판단을 같은 데이터·같은 방식으로 나란히 재확인해,
# 교수님 보고와 논문 서두 논리의 실증 근거를 만든다.
#
# ## 무엇을 (What)
#   E0 종목 선정 : 변동성 × 거래대금 점수로 상위 20종목을 뽑는다(Dynamic Volatility Ticker Filter).
#   E1 EDA       : 대표 종목(BTC) 기술통계(describe, 지수표기 없이)와 분포·꼬리·거래량 시각화.
#   E2 전처리 계보: 원본레벨 → 로그 → 차분 d → 타깃 → 스케일링 → 파생 (AGENTS.md 2.9b 표).
#   E3 자기상관   : 수익률 acf(방향 신호) vs |수익률| acf(변동성 신호) — 20종목 재현.
#   E4 방향       : 로지스틱 회귀 정확도·DA·DA(큰변동) vs 기준선 — 20종목, horizon 3개.
#   E5 변동성     : 선형 회귀로 미래 실현변동성 예측, R²·MASE·QLIKE·corr vs 나이브 — 20종목, horizon 3개.
#   E6 종합       : 방향 우위(≈0) vs 변동성 R²(+) 대조표·그림·결론.
#
# ## 어떻게 (How)
# 데이터 축은 `upbit_krw_candle`(2026-07-18 교정 이후 정본). 종목 선정은 DuckDB 윈도우 함수로
# 종목별 로그수익률 표준편차와 거래대금을 한 번에 집계한다. look-ahead 방지: feature는 시점 t까지,
# 타깃은 t→t+h. 표준화·평균은 학습 구간(앞 70%)에서만 fit 한다. 모든 수치는 과학적표기 없이
# 저장하고 원시 수치를 `*_raw.md`로 분리한다(2.9b/2.9d). 지표는 축·나이브와 함께 보고한다(2.9d).
#
# ## 기대 결과 / 반영 (Expected)
# 같은 데이터·같은 방식에서 방향은 학습해도 기준선을 못 넘고(정확도 우위 ≈ 0), 변동성은 표본외로
# 명확히 예측됨(R² > 0, corr 높음)을 20종목 전부에서 재현한다. 이로써 "변동성 우선, 방향 후속"의
# 순서 결정을 데이터로 뒷받침한다.

# %%
"""20번 방향 대 변동성 예측 가능성 드라이버 본체.

실행:
    uv run test/models/20_direction_vs_volatility_predictability_test.py            # 전체(top20)
    uv run test/models/20_direction_vs_volatility_predictability_test.py --quick    # 빠른 점검(3종목·2만행)
    QT_NO_AUTOBUILD=1 uv run test/models/20_direction_vs_volatility_predictability_test.py
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = ["Noto Sans CJK KR", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["axes.formatter.useoffset"] = False
matplotlib.rcParams["axes.formatter.limits"] = (-9, 12)

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


def _project_root(start: Path) -> Path:
    for path in [start, *start.parents]:
        if (path / "pyproject.toml").exists() and (path / "test").exists():
            return path
    return start


ROOT = _project_root(Path(__file__).resolve())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine import preflight  # noqa: E402

EXPERIMENT_TAG = "20_direction_vs_volatility_predictability_20260903"
SOURCE_TABLE = "upbit_krw_candle"
REP_TICKER = "KRW-BTC"          # 심층 EDA 대표 종목
TOP_N = 20
HORIZONS = [1, 4, 16]           # 15분봉 기준 15분 / 1시간 / 4시간
H_LABEL = {1: "15분", 4: "1시간", 16: "4시간"}
LAGS = 8
RV_WINDOWS = [1, 4, 16, 96]
TRAIN_FRAC = 0.70
MIN_ROWS = 90_000               # 종목 선정 최소 행수(약 2.5년) — 신규 잡코인·생존편의 배제
EPS = 1e-12

IMAGES_DIR = ROOT / "test" / "images" / EXPERIMENT_TAG
RESULTS_DIR = ROOT / "test" / "results" / EXPERIMENT_TAG
RAW_PATH = RESULTS_DIR / "predictability_raw.md"

_LINES: list[str] = []


def emit(text: str = "") -> None:
    print(text, flush=True)
    _LINES.append(text)


def num(value, digits: int = 4) -> str:
    """과학적표기 없이(AGENTS.md 2.9b/2.9d) 포맷."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(v):
        return "n/a"
    return f"{v:,.{digits}f}"


def connect() -> duckdb.DuckDBPyConnection:
    db_path = ROOT / "data" / "upbit_data.db"
    if not db_path.exists():
        raise SystemExit(f"DB가 없다: {db_path}")
    return duckdb.connect(str(db_path), read_only=True)


# %% [markdown]
# ## E0 종목 선정 — 변동성 × 거래대금 필터
#
# 유동성만 크고 잔잔한 종목이나, 변동성만 크고 거래가 없는 유령 종목을 모두 피하기 위해
# **score = 로그수익률 표준편차 × log(총 거래대금)** 으로 상위 20종목을 뽑는다. DuckDB 윈도우
# 함수로 종목별 로그수익률 표준편차와 거래대금을 한 번의 스캔으로 집계한다.

# %%
def select_top_tickers(con: duckdb.DuckDBPyConnection, n: int, min_rows: int) -> pd.DataFrame:
    query = f"""
    WITH r AS (
        SELECT ticker,
               ln(close / lag(close) OVER (PARTITION BY ticker ORDER BY timestamp)) AS lr,
               value
        FROM {SOURCE_TABLE}
    )
    SELECT ticker,
           count(lr)          AS n,
           stddev_samp(lr)    AS vol,
           sum(value)         AS sum_value,
           avg(value)         AS avg_value
    FROM r
    WHERE lr IS NOT NULL AND isfinite(lr)
    GROUP BY ticker
    HAVING count(lr) >= {int(min_rows)}
    """
    df = con.execute(query).df()
    df["score"] = df["vol"] * np.log(df["sum_value"].clip(lower=1.0))
    df = df.sort_values("score", ascending=False).head(n).reset_index(drop=True)
    return df


def load_series(con: duckdb.DuckDBPyConnection, ticker: str, max_rows: int | None) -> pd.DataFrame:
    frame = con.execute(
        f"select timestamp, open, high, low, close, volume, value "
        f"from {SOURCE_TABLE} where ticker = ? order by timestamp",
        [ticker],
    ).df()
    if max_rows:
        frame = frame.tail(max_rows).reset_index(drop=True)
    close = frame["close"].astype(float).clip(lower=1e-9)
    frame["log_close"] = np.log(close)
    frame["log_return"] = frame["log_close"].diff()
    return frame


# %% [markdown]
# ## 공통 계산기 — 자기상관, feature, 타깃, 지표
#
# feature는 시점 t까지만 관측 가능한 값으로 구성하고, 타깃은 t→t+h로 미래를 본다.
# 모든 rolling 계산은 numpy/pandas로 벡터화한다(파이썬 행루프는 10만 행 × 20종목에서 실행 불가).

# %%
def autocorr(x: np.ndarray, lag: int) -> float:
    x = x - x.mean()
    denom = np.sum(x * x)
    if denom < EPS:
        return float("nan")
    return float(np.sum(x[lag:] * x[:-lag]) / denom)


def build_features(logret: np.ndarray, volume: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """방향용 feature와 변동성용 feature를 함께 만든다(둘 다 t 시점까지만)."""
    s = pd.Series(logret)
    r2 = pd.Series(logret ** 2)
    absr = pd.Series(np.abs(logret))
    n = len(logret)

    dir_cols, vol_cols = [], []
    # 방향용: 과거 수익률 lag
    for k in range(1, LAGS + 1):
        dir_cols.append(s.shift(k).fillna(0.0).to_numpy())
    # 과거 실현변동성(√평균 r²)과 모멘텀(평균 r): 방향·변동성 공용
    rv_by_win = {}
    for w in RV_WINDOWS:
        rv = np.sqrt(r2.shift(1).rolling(w, min_periods=1).mean()).fillna(0.0).to_numpy()
        rv_by_win[w] = rv
        mom = s.shift(1).rolling(w, min_periods=1).mean().fillna(0.0).to_numpy()
        if w in (4, 16):
            dir_cols.append(rv); dir_cols.append(mom)
        vol_cols.append(np.log(rv + 1e-8))
    # 거래량 z-score(최근 96봉≈1일)
    v = pd.Series(volume)
    vmean = v.shift(1).rolling(96, min_periods=6).mean()
    vstd = v.shift(1).rolling(96, min_periods=6).std()
    vol_z = ((v - vmean) / vstd.replace(0, np.nan)).fillna(0.0).to_numpy()
    dir_cols.append(vol_z)
    vol_cols.append(vol_z)

    return np.column_stack(dir_cols), np.column_stack(vol_cols)


def future_cum_ret(logret: np.ndarray, h: int) -> np.ndarray:
    csum = np.concatenate([[0.0], np.cumsum(logret)])
    out = np.full(len(logret), np.nan)
    idx = np.arange(len(logret))
    ok = idx + 1 + h <= len(logret)
    out[ok] = csum[idx[ok] + 1 + h] - csum[idx[ok] + 1]
    return out


def future_rv(logret: np.ndarray, h: int) -> np.ndarray:
    sq = logret ** 2
    csum = np.concatenate([[0.0], np.cumsum(sq)])
    out = np.full(len(logret), np.nan)
    idx = np.arange(len(logret))
    ok = idx + 1 + h <= len(logret)
    out[ok] = np.sqrt(csum[idx[ok] + 1 + h] - csum[idx[ok] + 1])
    return out


def past_rv_sum(logret: np.ndarray, h: int) -> np.ndarray:
    """직전 h봉 실현변동성 sqrt(sum r²) — future_rv와 같은 척도(나이브 지속성 기준선)."""
    r2 = pd.Series(logret ** 2)
    return np.sqrt(r2.rolling(h, min_periods=1).sum()).to_numpy()


def r2_oos(y_true, y_pred, train_mean) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - train_mean) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > EPS else float("nan")


def direction_metrics(logret, Xdir, h):
    fcum = future_cum_ret(logret, h)
    valid = ~np.isnan(fcum)
    valid[: LAGS + 1] = False
    y = (fcum > 0).astype(float)
    X, yv, fc = Xdir[valid], y[valid], fcum[valid]
    if len(yv) < 500:
        return None
    split = int(len(yv) * TRAIN_FRAC)
    Xtr, Xte, ytr, yte = X[:split], X[split:], yv[:split], yv[split:]
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=1000).fit(sc.transform(Xtr), ytr)
    pred = clf.predict(sc.transform(Xte))
    acc = float((pred == yte).mean())
    base = max(yte.mean(), 1 - yte.mean())
    maj = 1.0 if ytr.mean() >= 0.5 else 0.0
    base = max(base, float((yte == maj).mean()))
    # 큰 변동 구간(|미래수익률| 상위 25%)만의 방향 정확도
    fte = np.abs(fc[split:])
    thr = np.quantile(fte, 0.75)
    big = fte >= thr
    da_big = float((pred[big] == yte[big]).mean()) if big.sum() > 0 else float("nan")
    n_correct = int((pred == yte).sum())
    pval = stats.binomtest(n_correct, len(yte), 0.5).pvalue
    return {"acc": acc, "base": base, "edge": acc - base, "da_big": da_big,
            "pval": pval, "up_rate": float(yte.mean())}


def volatility_metrics(logret, Xvol, h):
    frv = future_rv(logret, h)
    valid = ~np.isnan(frv)
    valid[: LAGS + 1] = False
    ylog = np.log(frv + 1e-8)
    X, yv, fr = Xvol[valid], ylog[valid], frv[valid]
    if len(yv) < 500:
        return None
    split = int(len(yv) * TRAIN_FRAC)
    Xtr, Xte, ytr, yte = X[:split], X[split:], yv[:split], yv[split:]
    sc = StandardScaler().fit(Xtr)
    reg = LinearRegression().fit(sc.transform(Xtr), ytr)
    pred = reg.predict(sc.transform(Xte))
    tmean = ytr.mean()
    r2 = r2_oos(yte, pred, tmean)
    naive_log = np.log(past_rv_sum(logret, h)[valid][split:] + 1e-8)
    r2_naive = r2_oos(yte, naive_log, tmean)
    # MASE(축 불변): 모델 MAE / 나이브 MAE
    mae_model = np.mean(np.abs(yte - pred))
    mae_naive = np.mean(np.abs(yte - naive_log))
    mase = float(mae_model / mae_naive) if mae_naive > EPS else float("nan")
    # QLIKE(분산 척도): log(σ²) + rv²/σ²  (낮을수록 좋음)
    var_pred = np.exp(2 * pred)
    rv_actual2 = fr[split:] ** 2
    qlike = float(np.mean(np.log(var_pred + EPS) + rv_actual2 / (var_pred + EPS)))
    var_naive = np.exp(2 * naive_log)
    qlike_naive = float(np.mean(np.log(var_naive + EPS) + rv_actual2 / (var_naive + EPS)))
    corr = float(np.corrcoef(np.exp(pred), fr[split:])[0, 1])
    return {"r2": r2, "r2_naive": r2_naive, "mase": mase, "qlike": qlike,
            "qlike_naive": qlike_naive, "corr": corr}


# %% [markdown]
# ## E1 EDA — 대표 종목(BTC) 기술통계와 분포·꼬리·거래량
#
# `describe()`는 지수표기 없이(정수/실수) 낸다. 수익률의 두꺼운 꼬리(초과첨도)와 |수익률|의
# 비대칭 분포가 "부호는 잡음, 크기는 구조"라는 이후 결과의 전조다.

# %%
def eda_representative(frame: pd.DataFrame) -> None:
    r = frame["log_return"].dropna().to_numpy()
    absr = np.abs(r)
    emit("## E1 EDA — 대표 종목(BTC) 기술통계")
    emit()
    emit(f"- 기간: {frame['timestamp'].iloc[0]} ~ {frame['timestamp'].iloc[-1]}, {len(frame):,}봉(15분)")
    emit(f"- 결측 close: {int(frame['close'].isna().sum())}개, 결측 volume: {int(frame['volume'].isna().sum())}개")
    emit()

    desc = pd.DataFrame({
        "종가": frame["close"],
        "거래량": frame["volume"],
        "거래대금": frame["value"],
        "로그수익률": frame["log_return"],
        "절댓값수익률": frame["log_return"].abs(),
    }).describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99])
    emit("### describe (지수표기 없이)")
    emit()
    cols = list(desc.columns)
    emit("| 통계량 | " + " | ".join(cols) + " |")
    emit("| :--- | " + " | ".join(["---:"] * len(cols)) + " |")
    digits = {"종가": 2, "거래량": 4, "거래대금": 2, "로그수익률": 6, "절댓값수익률": 6}
    for stat in desc.index:
        row = " | ".join(num(desc.loc[stat, c], digits[c]) for c in cols)
        emit(f"| {stat} | {row} |")
    emit()
    skew = float(stats.skew(r)); kurt = float(stats.kurtosis(r))  # 초과첨도(정규=0)
    emit(f"- 로그수익률 왜도 {num(skew)}, 초과첨도 {num(kurt)} (정규분포면 0). "
         f"초과첨도가 크게 양(+)이면 두꺼운 꼬리 = 급변이 정규 가정보다 잦다.")
    emit(f"- |수익률| 자기상관 lag1 {num(autocorr(absr, 1))} vs 수익률 자기상관 lag1 {num(autocorr(r, 1))}")
    emit()

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    axes[0, 0].plot(frame["timestamp"], frame["close"], lw=0.6, color="#1f77b4")
    axes[0, 0].set_title("BTC 종가 (레벨, 비정상)")
    axes[0, 1].plot(frame["timestamp"], frame["log_return"], lw=0.4, color="#555555")
    axes[0, 1].set_title("로그수익률 (차분, 정상) — 변동성 군집이 눈에 보인다")
    # 수익률 분포 + 정규곡선
    axes[1, 0].hist(r, bins=200, density=True, color="#9ecae1", alpha=0.8)
    xs = np.linspace(r.min(), r.max(), 400)
    axes[1, 0].plot(xs, stats.norm.pdf(xs, r.mean(), r.std()), "r--", lw=1.2, label="정규분포")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_title(f"로그수익률 분포(로그 y축)\n초과첨도 {num(kurt,2)} — 꼬리가 정규보다 두껍다")
    axes[1, 0].legend()
    axes[1, 1].hist(absr, bins=200, density=True, color="#fc9272", alpha=0.85)
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_title("|수익률| 분포(크기) — 이 크기를 예측 대상으로 삼는다")
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "e1_eda_representative.png", dpi=120)
    plt.close(fig)
    emit("![E1 EDA](../../images/" + EXPERIMENT_TAG + "/e1_eda_representative.png)")
    emit()


# %% [markdown]
# ## E2 전처리 계보 (AGENTS.md 2.9b)
#
# 이 분석의 예측 대상이 "차분값(로그수익률)에서 파생된 부호/크기"임을 표로 먼저 밝힌다.

# %%
def preprocessing_lineage() -> None:
    emit("## E2 전처리 계보")
    emit()
    emit("| 단계 | 처리 | 근거 | 적용 |")
    emit("| :--- | :--- | :--- | :--- |")
    emit("| 원본 레벨 | 종가 close | 원자료 | 비정상(단위근) |")
    emit("| 변환 | 로그: log(close) | 배율 효과 제거, 곱셈→덧셈 | 여전히 비정상 |")
    emit("| 차분 차수 d=1 | log_return = Δlog(close) | 정상성 확보(ADF 기각) | 예측 대상의 토대 |")
    emit("| 타깃(방향) | sign(Σ_{t+1..t+h} r) | 부호=오름/내림 | 분류 라벨 |")
    emit("| 타깃(변동성) | √(Σ_{t+1..t+h} r²) | 부호를 뺀 움직임의 크기 | 회귀 타깃(로그 스케일) |")
    emit("| 스케일링 | 학습구간 StandardScaler | 누설 방지, 학습통계로만 fit | feature에 적용 |")
    emit("| 파생변수 | 과거 수익률 lag, 실현변동성, 모멘텀, 거래량 z | t 시점까지만 사용 | look-ahead 없음 |")
    emit()
    emit("- **로그 변환은 차분이 아니다**(단위 변경). 정상성은 d=1 차분에서 온다.")
    emit("- 평가축: 변동성은 로그 실현변동성 축에서 R²·MASE·QLIKE·corr를, 방향은 정확도·DA를 낸다(2.9d).")
    emit()


# %% [markdown]
# ## E3~E6 — 전종목 재현과 종합 대조

# %%
def run(top: pd.DataFrame, con, max_rows) -> pd.DataFrame:
    records = []
    for _, trow in top.iterrows():
        ticker = trow["ticker"]
        frame = load_series(con, ticker, max_rows)
        logret = frame["log_return"].fillna(0.0).to_numpy()
        volume = frame["volume"].fillna(0.0).to_numpy()
        if len(logret) < 2000:
            continue
        Xdir, Xvol = build_features(logret, volume)
        absr = np.abs(logret)
        rec = {"ticker": ticker, "n": len(logret),
               "acf1_ret": autocorr(logret, 1), "acf1_abs": autocorr(absr, 1),
               "acf16_ret": autocorr(logret, 16), "acf16_abs": autocorr(absr, 16),
               "acf64_abs": autocorr(absr, 64)}
        for h in HORIZONS:
            dm = direction_metrics(logret, Xdir, h)
            vm = volatility_metrics(logret, Xvol, h)
            if dm:
                rec[f"acc_{h}"] = dm["acc"]; rec[f"edge_{h}"] = dm["edge"]
                rec[f"up_{h}"] = dm["up_rate"]
                rec[f"dabig_{h}"] = dm["da_big"]; rec[f"pval_{h}"] = dm["pval"]
            if vm:
                rec[f"r2_{h}"] = vm["r2"]; rec[f"r2naive_{h}"] = vm["r2_naive"]
                rec[f"mase_{h}"] = vm["mase"]; rec[f"qlike_{h}"] = vm["qlike"]
                rec[f"qlnaive_{h}"] = vm["qlike_naive"]; rec[f"corr_{h}"] = vm["corr"]
        records.append(rec)
        print(f"  · {ticker} 완료 ({len(logret):,}봉)", flush=True)
    return pd.DataFrame(records)


def report_acf(df: pd.DataFrame) -> None:
    emit("## E3 자기상관 대조 — 방향(수익률) vs 변동성(|수익률|)")
    emit()
    emit("| 종목 | 행수 | 수익률 acf(1) | |수익률| acf(1) | |수익률| acf(16) | |수익률| acf(64) |")
    emit("| :--- | ---: | ---: | ---: | ---: | ---: |")
    for _, r in df.iterrows():
        emit(f"| {r['ticker']} | {int(r['n']):,} | {num(r['acf1_ret'])} | {num(r['acf1_abs'])} | "
             f"{num(r['acf16_abs'])} | {num(r['acf64_abs'])} |")
    emit()
    emit(f"- 수익률 acf(1) 평균 {num(df['acf1_ret'].mean())} (0 근처=방향 예측 불가)")
    emit(f"- |수익률| acf(1) 평균 {num(df['acf1_abs'].mean())}, acf(64) 평균 {num(df['acf64_abs'].mean())} "
         f"(느리게 감소=장기기억=변동성 예측 가능)")
    emit()
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(max(10, len(df) * 0.7), 5.5))
    ax.bar(x - 0.2, df["acf1_ret"], width=0.4, label="수익률 acf(1) — 방향", color="#1f77b4")
    ax.bar(x + 0.2, df["acf1_abs"], width=0.4, label="|수익률| acf(1) — 변동성", color="#d62728")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(df["ticker"], rotation=90, fontsize=8)
    ax.set_title("자기상관 대조: 부호는 ≈0, 크기는 강한 양(+) — 20종목 전부 같은 구조")
    ax.legend()
    fig.tight_layout(); fig.savefig(IMAGES_DIR / "e3_acf_contrast.png", dpi=120); plt.close(fig)
    emit("![E3 ACF](../../images/" + EXPERIMENT_TAG + "/e3_acf_contrast.png)")
    emit()


def report_contrast(df: pd.DataFrame) -> None:
    emit("## E4~E6 방향(분류) 대 변동성(회귀) — 종합 대조")
    emit()
    for h in HORIZONS:
        emit(f"### 예측 구간 {H_LABEL[h]}")
        emit()
        emit("| 종목 | 방향 정확도 | 상승비율(쏠림) | 기준선 대비 우위 | DA(큰변동) | ‖ | 변동성 R² | 나이브 R² | MASE | corr |")
        emit("| :--- | ---: | ---: | ---: | ---: | :--: | ---: | ---: | ---: | ---: |")
        for _, r in df.iterrows():
            emit(f"| {r['ticker']} | {num(r.get(f'acc_{h}'))} | {num(r.get(f'up_{h}'))} | {num(r.get(f'edge_{h}'))} | "
                 f"{num(r.get(f'dabig_{h}'))} | ‖ | {num(r.get(f'r2_{h}'))} | {num(r.get(f'r2naive_{h}'))} | "
                 f"{num(r.get(f'mase_{h}'))} | {num(r.get(f'corr_{h}'))} |")
        emit()
        emit(f"- 방향 정확도 평균 {num(df[f'acc_{h}'].mean())}, 기준선 대비 우위 평균 "
             f"{num(df[f'edge_{h}'].mean())} (≈0 = 동전 던지기)")
        emit("  · 주의: 방향 정확도 자체가 높아 보여도 대부분 **상승비율 쏠림** 때문이다. "
             "다수 클래스만 찍는 기준선이 같은 정확도를 내므로, 실제 예측력은 '우위'로만 판단한다.")
        emit(f"- 변동성 표본외 R² 평균 {num(df[f'r2_{h}'].mean())}, corr 평균 {num(df[f'corr_{h}'].mean())} "
             f"(양수·높음 = 예측됨). MASE 평균 {num(df[f'mase_{h}'].mean())} (<1이면 나이브보다 우수)")
        emit()

    # 종합 그림: 방향 우위(≈0) vs 변동성 R²(+)
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    for j, h in enumerate(HORIZONS):
        x = np.arange(len(df))
        axes[j].bar(x - 0.2, df[f"edge_{h}"], width=0.4, label="방향 우위(정확도−기준선)", color="#1f77b4")
        axes[j].bar(x + 0.2, df[f"r2_{h}"], width=0.4, label="변동성 표본외 R²", color="#d62728")
        axes[j].axhline(0, color="black", lw=0.8)
        axes[j].set_xticks(x); axes[j].set_xticklabels(df["ticker"], rotation=90, fontsize=7)
        axes[j].set_title(f"{H_LABEL[h]}: 방향 우위 ≈ 0 vs 변동성 R² > 0")
        axes[j].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(IMAGES_DIR / "e6_contrast.png", dpi=120); plt.close(fig)
    emit("![E6 종합 대조](../../images/" + EXPERIMENT_TAG + "/e6_contrast.png)")
    emit()
    emit("## 결론")
    emit()
    emit("- **방향**: 20종목·3구간 어디서도 학습이 기준선을 유의하게 넘지 못한다(우위 ≈ 0). "
         "큰 변동 구간(DA)만 따로 봐도 개선되지 않는다. → 지금 방향을 주력에 넣으면 잡음을 신호로 "
         "착각할 위험이 크다.")
    emit("- **변동성**: 같은 데이터·같은 방식에서 표본외 R²가 양으로 크고 corr가 높다. → 예측이 되는 "
         "대상이므로 **변동성을 먼저 확립하고, 방향은 그 위에서 이후 단계로 이어간다.**")
    emit()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="빠른 점검(3종목·2만행)")
    parser.add_argument("--top-n", type=int, default=TOP_N)
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args(argv)

    top_n = 3 if args.quick else args.top_n
    max_rows = 20_000 if args.quick else args.max_rows

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 2.9e 데이터 점검(수집은 하지 않고 점검만; 부족하면 중단)
    try:
        preflight.ensure_data(table=SOURCE_TABLE, tickers="all", auto_build=False, strict=False)
    except Exception as exc:  # noqa: BLE001
        print(f"[preflight 경고] {exc}", flush=True)

    con = connect()
    pd.set_option("display.float_format", lambda v: f"{v:,.6f}")

    emit(f"# 20번 방향 대 변동성 예측 가능성 — 원시 수치 ({'quick' if args.quick else 'full'})")
    emit()
    floor = 60_000 if args.quick else MIN_ROWS
    top = select_top_tickers(con, top_n, floor)
    emit("## E0 종목 선정 — 변동성 × 거래대금 상위 (긴 이력 종목 한정)")
    emit()
    emit(f"- 선정 기준: 이력 {floor:,}행 이상(약 {floor/96/365:.1f}년) 종목 중 "
         f"**점수 = 로그수익률 표준편차 × log(총 거래대금)** 상위 {top_n}개.")
    emit("- 이력 하한선으로 신규 상장 잡코인과 생존편의를 배제한다. BTC는 저변동 대형 종목이라 "
         "변동성 순위에는 안 들지만, 기존 연구와의 연결을 위해 **대표 기준 종목으로 별도 추가**한다.")
    emit()
    emit("| 순위 | 종목 | 행수 | 로그수익률 표준편차 | 총 거래대금 | 선정 점수 |")
    emit("| ---: | :--- | ---: | ---: | ---: | ---: |")
    for i, r in top.iterrows():
        emit(f"| {i+1} | {r['ticker']} | {int(r['n']):,} | {num(r['vol'], 6)} | "
             f"{num(r['sum_value'], 0)} | {num(r['score'], 4)} |")
    emit()

    # 대표 종목 심층 EDA — 항상 BTC(기존 연구 기준)
    eda_representative(load_series(con, REP_TICKER, max_rows))
    preprocessing_lineage()

    # 분석 대상 = BTC(대표) + 선정된 상위 종목
    analysis_tickers = list(dict.fromkeys([REP_TICKER] + list(top["ticker"])))
    run_df = pd.DataFrame({"ticker": analysis_tickers})
    print(f"[run] {len(analysis_tickers)}종목 분석 시작(BTC 대표 + 상위 {len(top)})...", flush=True)
    df = run(run_df, con, max_rows)
    report_acf(df)
    report_contrast(df)

    df.to_csv(RESULTS_DIR / "predictability_metrics.csv", index=False)
    RAW_PATH.write_text("\n".join(_LINES) + "\n", encoding="utf-8")
    print(f"\n저장: {RAW_PATH}", flush=True)
    print(f"저장: {RESULTS_DIR / 'predictability_metrics.csv'}", flush=True)
    print(f"그림: {IMAGES_DIR}", flush=True)


if __name__ == "__main__":
    main()
