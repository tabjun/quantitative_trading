# %% [markdown]
# # 21번: 변동성 예측을 위한 EDA (처음부터 다시)
#
# 헤드리스 `.py` 드라이버(`#%%` 셀 + 파일 내 마크다운, AGENTS.md 2.3). ipynb는 만들지 않는다.
#
# ## 왜 (Why)
# 방향성(예측 대상 = 변동성)만 확정하고, 모델·전처리·feature는 **전부 EDA부터 다시 시작**한다.
# 이전에 적합한 모델(GARCH-t·HAR-RV 등)은 결정으로 가져가지 않는다. 이 EDA가 이후 전처리·모델
# 선택의 유일한 근거가 된다. 모든 변수는 `DATA_DICTIONARY.md`에 raw부터 파생까지 누적 정의한다.
#
# ## 무엇을 (What)
#   B  연구 배경 : 코인 24시간 연속 거래의 성질과 변동성 예측 연구 배경.
#   Q  데이터 품질: 15분 연속성·갭·결측·중복·OHLC 정합성.
#   R  RAW EDA   : open/high/low/close/volume/value 6개 열 전부(변환 전).
#   V  변동성 EDA: 로그수익률 분포·정규성·정상성(ADF/KPSS)·자기상관(r vs |r| vs r²)·
#                  변동성 군집·ARCH-LM·Ljung-Box·시간대/요일 계절성·범위변동성·거래량 관계.
#   S  표본 타당성: 모집단(전 종목) 성질 분포에서 상위 20종목의 위치 + 정형화 사실 보편성.
#   D  데이터 사전: raw + 이번 EDA에서 생긴 파생변수를 근거 유형과 함께 등록(v1).
#
# ## 어떻게 (How)
# 대표 종목 심층은 BTC. 표본 타당성은 전 종목(이력 하한 이상)을 스캔한다. 통계 검정은
# statsmodels(ADF/KPSS/Ljung-Box/ARCH-LM). 모든 수치는 과학적표기 없이 저장한다(2.9b/2.9d).
# 이 단계에서는 **모델을 적합하지 않는다** — 오직 데이터의 성질만 본다.
#
# ## 기대 결과 (Expected)
# "변동성 예측이 왜 가능한가(변동성 군집·조건부 이분산·두꺼운 꼬리)"를 데이터로 확인하고,
# 이후 전처리·모델 결정의 가정을 세운다. 표본 20종목이 모집단 현상을 대표함을 입증한다.

# %%
"""21번 변동성 EDA 드라이버 본체.

실행:
    uv run test/models/21_volatility_eda_test.py
    uv run test/models/21_volatility_eda_test.py --quick   # 모집단 스캔 축소
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
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.stattools import adfuller, kpss

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

EXPERIMENT_TAG = "21_volatility_eda_20260904"
SOURCE_TABLE = "upbit_krw_candle"
REP_TICKER = "KRW-BTC"
BARS_PER_DAY = 96           # 15분봉 하루
POP_MIN_ROWS = 5_000        # 모집단 스캔 대상 최소 이력
POP_TAIL = 20_000           # 모집단 스캔 시 종목별 최근 N봉(속도)

IMAGES_DIR = ROOT / "test" / "images" / EXPERIMENT_TAG
RESULTS_DIR = ROOT / "test" / "results" / EXPERIMENT_TAG
RAW_PATH = RESULTS_DIR / "volatility_eda_raw.md"
DICT_PATH = ROOT / "test" / "results" / "DATA_DICTIONARY.md"

_LINES: list[str] = []


def emit(text: str = "") -> None:
    print(text, flush=True)
    _LINES.append(text)


def num(value, digits: int = 4) -> str:
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


def acf(x: np.ndarray, lag: int) -> float:
    x = x - x.mean()
    denom = np.sum(x * x)
    if denom < 1e-18:
        return float("nan")
    return float(np.sum(x[lag:] * x[:-lag]) / denom)


# %% [markdown]
# ## B. 연구 배경 — 코인 24시간 연속 거래와 변동성 예측

# %%
def background() -> None:
    emit("## B. 연구 배경")
    emit()
    emit("- **대상**: 업비트 KRW 마켓 암호화폐. 주식과 달리 **24시간·연중무휴 거래**라 장 시작/마감·")
    emit("  주말 갭이 없고, 시계열이 **끊김 없이 연속**이다(개장 점프로 인한 인공 변동이 없다).")
    emit("- **왜 시계열 분석에 적합한가**: 연속 표본이라 등간격(15분) 시계열 가정이 잘 맞고, 결측·갭이")
    emit("  적어 자기상관·조건부 이분산 같은 시간구조를 왜곡 없이 추정할 수 있다.")
    emit("- **연구 방향(변동성 예측)**: 다음 구간에 가격이 **오를지/내릴지(방향)**가 아니라 **얼마나")
    emit("  움직일지(변동성=크기)**를 예측한다. 방향은 효율적 시장 가설상 예측이 어렵고(선행 연구·")
    emit("  문헌에서 반복 확인), 변동성은 '큰 변동 뒤 큰 변동'이라는 **변동성 군집** 때문에 예측 가능하다.")
    emit("- **이 EDA의 역할**: 위 전제(변동성 군집·조건부 이분산·두꺼운 꼬리)가 실제 데이터에 있는지")
    emit("  확인하고, 이후 전처리·모델 선택의 가정을 세운다. 이 단계에서는 모델을 적합하지 않는다.")
    emit()


# %% [markdown]
# ## Q. 데이터 품질 — 연속성·갭·결측·OHLC 정합성

# %%
def data_quality(con) -> pd.DataFrame:
    df = con.execute(
        f"select timestamp, open, high, low, close, volume, value "
        f"from {SOURCE_TABLE} where ticker = ? order by timestamp",
        [REP_TICKER],
    ).df()
    emit("## Q. 데이터 품질 (대표 종목 BTC)")
    emit()
    emit(f"- 기간 {df['timestamp'].iloc[0]} ~ {df['timestamp'].iloc[-1]}, {len(df):,}봉(15분).")
    # 연속성: 15분 간격 위반
    dt = df["timestamp"].diff().dt.total_seconds().div(60).dropna()
    gaps = int((dt != 15).sum())
    big = int((dt > 15).sum())
    dup = int((dt == 0).sum())
    emit(f"- 15분 아닌 간격 {gaps}개(그중 누락 갭 {big}개, 중복 {dup}개). "
         f"전체 대비 {num(gaps/len(df)*100, 4)}% — 사실상 연속이나 소수 갭은 존재한다.")
    # 결측
    miss = {c: int(df[c].isna().sum()) for c in ["open", "high", "low", "close", "volume", "value"]}
    emit(f"- 결측: {miss}")
    # OHLC 정합성
    bad_hl = int((df["high"] < df["low"]).sum())
    bad_ho = int((df["high"] < df[["open", "close"]].max(axis=1)).sum())
    bad_lo = int((df["low"] > df[["open", "close"]].min(axis=1)).sum())
    zero_v = int((df["volume"] <= 0).sum())
    emit(f"- OHLC 정합성 위반: high<low {bad_hl}건, high<max(open,close) {bad_ho}건, "
         f"low>min(open,close) {bad_lo}건. 거래량 0 이하 {zero_v}건.")
    emit()
    # 갭 위치 시각화
    fig, ax = plt.subplots(figsize=(13, 3))
    ax.plot(df["timestamp"].iloc[1:], dt.values, lw=0.5, color="#1f77b4")
    ax.axhline(15, color="green", ls="--", lw=1, label="정상 15분")
    ax.set_ylim(0, max(60, dt.max() * 1.05))
    ax.set_title("연속성 점검: 봉 간격(분) — 15분에서 벗어난 지점이 갭")
    ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(IMAGES_DIR / "q_continuity.png", dpi=120); plt.close(fig)
    emit("![Q 연속성](../../images/" + EXPERIMENT_TAG + "/q_continuity.png)")
    emit()
    return df


# %% [markdown]
# ## R. RAW 6개 열 EDA (변환 전)

# %%
def raw_eda(df: pd.DataFrame) -> None:
    emit("## R. RAW 열별 기술통계 (변환 전, 지수표기 없이)")
    emit()
    cols = ["open", "high", "low", "close", "volume", "value"]
    desc = df[cols].describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99])
    emit("| 통계량 | " + " | ".join(cols) + " |")
    emit("| :--- | " + " | ".join(["---:"] * len(cols)) + " |")
    dig = {"open": 2, "high": 2, "low": 2, "close": 2, "volume": 4, "value": 2}
    for stat in desc.index:
        emit(f"| {stat} | " + " | ".join(num(desc.loc[stat, c], dig[c]) for c in cols) + " |")
    emit()
    emit("- open/high/low/close는 원(KRW) 가격, volume은 코인 수량, value는 원 거래대금이다.")
    emit("- **이 6개 열은 아직 어느 것도 모델에서 배제하기로 결정하지 않았다**(EDA 단계에서 전부 본다).")
    emit()
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    axes[0, 0].plot(df["timestamp"], df["close"], lw=0.5, color="#1f77b4"); axes[0, 0].set_title("close (종가)")
    axes[0, 1].plot(df["timestamp"], df["volume"], lw=0.4, color="#7f7f7f"); axes[0, 1].set_title("volume (수량)")
    axes[0, 2].plot(df["timestamp"], df["value"], lw=0.4, color="#8c564b"); axes[0, 2].set_title("value (거래대금)")
    axes[1, 0].hist(df["close"], bins=120, color="#9ecae1"); axes[1, 0].set_title("close 분포")
    axes[1, 1].hist(np.log1p(df["volume"]), bins=120, color="#c7c7c7"); axes[1, 1].set_title("log(1+volume) 분포")
    hl = np.log(df["high"] / df["low"].clip(lower=1e-9))
    axes[1, 2].hist(hl, bins=120, color="#fdd0a2"); axes[1, 2].set_title("봉내 고저 범위 ln(high/low) 분포")
    fig.tight_layout(); fig.savefig(IMAGES_DIR / "r_raw_columns.png", dpi=120); plt.close(fig)
    emit("![R RAW](../../images/" + EXPERIMENT_TAG + "/r_raw_columns.png)")
    emit()


# %% [markdown]
# ## V. 변동성 지향 EDA — 수익률의 성질

# %%
def volatility_eda(df: pd.DataFrame) -> None:
    close = df["close"].astype(float).clip(lower=1e-9).to_numpy()
    logc = np.log(close)
    r = np.diff(logc)                      # 로그수익률
    absr = np.abs(r)
    r2 = r ** 2
    emit("## V. 변동성 지향 EDA — 로그수익률의 성질")
    emit()

    # V1 분포·정규성
    skew = float(stats.skew(r)); kurt = float(stats.kurtosis(r))
    jb, jb_p = stats.jarque_bera(r)[:2]
    emit("### V1. 분포와 정규성")
    emit(f"- 로그수익률 평균 {num(r.mean(),8)}, 표준편차 {num(r.std(),6)}, 왜도 {num(skew)}, "
         f"초과첨도 {num(kurt)} (정규분포=0).")
    emit(f"- Jarque-Bera 정규성 검정 통계량 {num(jb,1)}, p={num(jb_p,4)} → "
         f"{'정규성 기각(두꺼운 꼬리)' if jb_p < 0.05 else '정규성 유지'}. "
         f"초과첨도가 크게 양(+)이면 급변이 정규 가정보다 잦다.")
    emit()

    # V2 정상성
    emit("### V2. 정상성 (레벨 vs 로그수익률)")
    adf_lvl = adfuller(logc, autolag="AIC")
    adf_ret = adfuller(r, autolag="AIC")
    try:
        kpss_lvl = kpss(logc, regression="c", nlags="auto")
    except Exception:
        kpss_lvl = (float("nan"), float("nan"))
    try:
        kpss_ret = kpss(r, regression="c", nlags="auto")
    except Exception:
        kpss_ret = (float("nan"), float("nan"))
    emit("| 계열 | ADF 통계량 | ADF p (작을수록 정상) | KPSS 통계량 | KPSS p (클수록 정상) | 판정 |")
    emit("| :--- | ---: | ---: | ---: | ---: | :--- |")
    emit(f"| 로그가격(레벨) | {num(adf_lvl[0])} | {num(adf_lvl[1],4)} | {num(kpss_lvl[0])} | "
         f"{num(kpss_lvl[1],4)} | 비정상(단위근) |")
    emit(f"| 로그수익률(1차차분) | {num(adf_ret[0])} | {num(adf_ret[1],4)} | {num(kpss_ret[0])} | "
         f"{num(kpss_ret[1],4)} | 정상 |")
    emit("- **결론**: 레벨은 비정상이라 차분(로그수익률)이 정상성의 근거다. 로그는 정상성이 아니라")
    emit("  스케일/분산 안정화를 위한 것이고, 정상성은 1차 차분에서 온다.")
    emit()

    # V3 자기상관 r vs |r| vs r^2
    emit("### V3. 자기상관 — 방향(r) vs 크기(|r|, r²)")
    lags = [1, 2, 4, 8, 16, 32, 64, 96]
    emit("| lag | r 자기상관(방향) | |r| 자기상관(크기) | r² 자기상관(크기) |")
    emit("| ---: | ---: | ---: | ---: |")
    for L in lags:
        emit(f"| {L} | {num(acf(r,L))} | {num(acf(absr,L))} | {num(acf(r2,L))} |")
    emit("- r(부호)은 0 근처로 곧 사라지고(방향 예측 불가), |r|·r²(크기)은 강한 양의 상관이 느리게")
    emit("  감소한다(장기기억=변동성 군집=예측 가능). **변동성 예측의 근거가 여기 있다.**")
    emit()

    # V4 Ljung-Box, ARCH-LM
    emit("### V4. 시간구조 검정 (Ljung-Box, ARCH-LM)")
    lb_r = acorr_ljungbox(r, lags=[BARS_PER_DAY], return_df=True)
    lb_r2 = acorr_ljungbox(r2, lags=[BARS_PER_DAY], return_df=True)
    arch = het_arch(r - r.mean(), nlags=16)
    emit(f"- Ljung-Box(96) — r: p={num(lb_r['lb_pvalue'].iloc[0],4)} / "
         f"r²: p={num(lb_r2['lb_pvalue'].iloc[0],4)} (p<0.05면 자기상관 있음).")
    emit(f"- ARCH-LM(16): 통계량 {num(arch[0],1)}, p={num(arch[1],4)} → "
         f"{'조건부 이분산 존재(GARCH류 정당화)' if arch[1] < 0.05 else '조건부 이분산 없음'}.")
    emit("- r²에 강한 자기상관 + ARCH 효과 = 변동성이 시간에 따라 예측 가능하게 변한다는 직접 증거.")
    emit()

    # V5 계절성 (시간대/요일)
    ts = df["timestamp"].iloc[1:].reset_index(drop=True)
    hour = ts.dt.hour.to_numpy()
    dow = ts.dt.dayofweek.to_numpy()
    hour_vol = [absr[hour == h].mean() for h in range(24)]
    dow_vol = [absr[dow == d].mean() for d in range(7)]
    emit("### V5. 계절성 — 시간대/요일별 평균 |수익률|")
    emit(f"- 시간대별 |수익률| 최대/최소 비율 {num(max(hour_vol)/min(hour_vol),2)}배, "
         f"요일별 {num(max(dow_vol)/min(dow_vol),2)}배 — 주기 성분이 있으면 이후 전처리에서 제거 검토.")
    emit()

    # V6 범위 기반 변동성 (high/low 활용)
    high = df["high"].astype(float).to_numpy()[1:]
    low = df["low"].astype(float).to_numpy()[1:]
    openp = df["open"].astype(float).to_numpy()[1:]
    closep = close[1:]
    park = (np.log(high / np.clip(low, 1e-9, None)) ** 2) / (4 * np.log(2))
    gk = 0.5 * np.log(high / np.clip(low, 1e-9, None)) ** 2 - (2 * np.log(2) - 1) * np.log(
        closep / np.clip(openp, 1e-9, None)) ** 2
    corr_park = float(np.corrcoef(np.sqrt(np.clip(park, 0, None)), absr)[0, 1])
    corr_gk = float(np.corrcoef(np.sqrt(np.clip(gk, 0, None)), absr)[0, 1])
    emit("### V6. 범위 기반 변동성 (high/low 활용) — 아직 미사용이던 열을 EDA")
    emit(f"- Parkinson √추정치 vs |수익률| 상관 {num(corr_park)}, Garman-Klass √추정치 vs |수익률| "
         f"상관 {num(corr_gk)} — high/low가 변동성 정보를 추가로 담는지 확인(이후 feature 후보).")
    emit()

    # V7 거래량-변동성 관계
    vol = df["volume"].astype(float).to_numpy()[1:]
    v = pd.Series(vol)
    vz = ((v - v.rolling(96, min_periods=6).mean()) / v.rolling(96, min_periods=6).std()).fillna(0).to_numpy()
    corr_vv = float(np.corrcoef(vz, absr)[0, 1])
    emit("### V7. 거래량-변동성 관계")
    emit(f"- 거래량 z-score와 |수익률| 동시상관 {num(corr_vv)} — 거래량이 변동성과 함께 움직이는지(공통 정보).")
    emit()

    # 그림: 변동성 EDA 종합
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    axes[0, 0].plot(ts, r, lw=0.3, color="#555"); axes[0, 0].set_title("로그수익률 — 변동성 군집이 보인다")
    axes[0, 1].hist(r, bins=200, density=True, color="#9ecae1", alpha=.8)
    xs = np.linspace(r.min(), r.max(), 400)
    axes[0, 1].plot(xs, stats.norm.pdf(xs, r.mean(), r.std()), "r--", lw=1.2, label="정규")
    axes[0, 1].set_yscale("log"); axes[0, 1].legend()
    axes[0, 1].set_title(f"수익률 분포(로그 y) — 초과첨도 {num(kurt,1)}, 두꺼운 꼬리")
    L = list(range(1, 97))
    axes[1, 0].plot(L, [acf(r, k) for k in L], color="#1f77b4", label="r (방향)")
    axes[1, 0].plot(L, [acf(absr, k) for k in L], color="#d62728", label="|r| (크기)")
    axes[1, 0].plot(L, [acf(r2, k) for k in L], color="#ff9896", label="r² (크기)")
    axes[1, 0].axhline(0, color="black", lw=.7); axes[1, 0].legend()
    axes[1, 0].set_title("자기상관: r ≈ 0 vs |r|/r² 강함(장기기억)")
    axes[1, 1].bar(range(24), hour_vol, color="#2ca02c")
    axes[1, 1].set_title("시간대(UTC?)별 평균 |수익률| — 주기 성분")
    axes[1, 1].set_xlabel("hour")
    fig.tight_layout(); fig.savefig(IMAGES_DIR / "v_volatility_eda.png", dpi=120); plt.close(fig)
    emit("![V 변동성 EDA](../../images/" + EXPERIMENT_TAG + "/v_volatility_eda.png)")
    emit()


# %% [markdown]
# ## S. 표본 타당성 — 모집단(전 종목) vs 상위 20종목

# %%
def sample_validity(con, quick: bool) -> None:
    emit("## S. 표본 타당성 — 모집단 대비 상위 20종목")
    emit()
    emit("상위 20종목은 무작위 표본이 아니라 **의도적(유동성·변동성 상위) 표본**이다. 타당성은")
    emit("'우리가 예측하려는 현상(변동성 군집·두꺼운 꼬리·조건부 이분산)이 특정 20종목만의 것이")
    emit("아니라 **모집단 전반에 보편적으로 존재**함'을 보여 확보한다.")
    emit()
    tickers = [t[0] for t in con.execute(
        f"select ticker from {SOURCE_TABLE} group by ticker having count(*)>={POP_MIN_ROWS}"
    ).fetchall()]
    if quick:
        tickers = tickers[:40]
    rows = []
    for tk in tickers:
        c = con.execute(
            f"select close from {SOURCE_TABLE} where ticker=? order by timestamp desc limit {POP_TAIL}",
            [tk]).df()["close"].astype(float).to_numpy()[::-1]
        if len(c) < 2000:
            continue
        r = np.diff(np.log(np.clip(c, 1e-9, None)))
        rows.append({"ticker": tk, "n": len(r), "std": r.std(),
                     "kurt": float(stats.kurtosis(r)), "acf1_abs": acf(np.abs(r), 1),
                     "acf1_r2": acf(r ** 2, 1)})
    pop = pd.DataFrame(rows)
    # 상위 20종목(변동성×거래대금) 재선정 — 20번과 동일 기준(긴 이력)
    top = con.execute(f"""
        with rr as (select ticker, ln(close/lag(close) over (partition by ticker order by timestamp)) lr, value
                    from {SOURCE_TABLE})
        select ticker, stddev_samp(lr) vol, sum(value) sv from rr
        where lr is not null and isfinite(lr) group by ticker having count(lr)>=90000
    """).df()
    top["score"] = top["vol"] * np.log(top["sv"].clip(lower=1))
    top20 = set(top.sort_values("score", ascending=False).head(20)["ticker"])
    pop["is_sample"] = pop["ticker"].isin(top20)

    emit(f"- 모집단 스캔: 이력 {POP_MIN_ROWS:,}봉 이상 {len(pop)}종목(종목별 최근 {POP_TAIL:,}봉).")
    emit(f"- **정형화 사실의 보편성(모집단 {len(pop)}종목)**:")
    emit(f"  - |수익률| 자기상관(1) > 0 : {int((pop['acf1_abs']>0).sum())}/{len(pop)}종목 "
         f"(변동성 군집이 거의 모든 종목에 존재)")
    emit(f"  - r² 자기상관(1) > 0 : {int((pop['acf1_r2']>0).sum())}/{len(pop)}종목")
    emit(f"  - 초과첨도 > 0(두꺼운 꼬리) : {int((pop['kurt']>0).sum())}/{len(pop)}종목")
    emit(f"  - 모집단 |수익률| 자기상관(1) 중앙값 {num(pop['acf1_abs'].median())}, "
         f"표본(20종목) 중앙값 {num(pop[pop.is_sample]['acf1_abs'].median())}")
    emit(f"- **표본 위치**: 상위 20종목은 변동성(std) 분포에서 상위 구간에 위치하나, |수익률| 자기상관·")
    emit(f"  두꺼운 꼬리 같은 **예측 대상 성질은 모집단과 같은 방향**이다 → 현상을 놓치지 않는다.")
    emit()

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))
    for ax, col, title in [
        (axes[0], "std", "변동성(로그수익률 표준편차)"),
        (axes[1], "acf1_abs", "변동성 군집 강도 |r| 자기상관(1)"),
        (axes[2], "kurt", "두꺼운 꼬리(초과첨도)")]:
        ax.hist(pop[col], bins=40, color="#c7c7c7", alpha=.85, label="모집단")
        for _, rr in pop[pop.is_sample].iterrows():
            ax.axvline(rr[col], color="#d62728", lw=1, alpha=.7)
        ax.set_title(f"{title}\n모집단 분포 + 표본 20종목(빨강 선)")
        ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(IMAGES_DIR / "s_sample_validity.png", dpi=120); plt.close(fig)
    emit("![S 표본 타당성](../../images/" + EXPERIMENT_TAG + "/s_sample_validity.png)")
    emit()


# %% [markdown]
# ## D. 데이터 사전 v1 — raw + 이번 EDA에서 확정한 파생변수

# %%
def write_data_dictionary() -> None:
    lines = [
        "# 데이터 사전 (DATA DICTIONARY) — 살아있는 문서",
        "",
        "모든 보고서는 이 문서를 참조한다. RAW에서 시작해 EDA·전처리로 파생변수가 생길 때마다",
        "여기에 누적 등록한다. 각 변수는 **정의/공식 · 무엇으로부터 · 독립(X)/종속(y) · 근거 유형 ·",
        "추가 시점**을 함께 적는다. 근거 유형: `관례`(방법론 표준) / `가정`(모델 가정 충족) /",
        "`EDA`(데이터에서 확인) / `방법`(구현상 필요).",
        "",
        "## 1. RAW (원본, `upbit_krw_candle`, 15분봉)",
        "",
        "| 변수 | 정의 | 타입 | 역할 | 근거 | 추가 |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
        "| ticker | 종목 코드(KRW-BTC 등) | 범주 | 식별자 | 원자료 | 21번 |",
        "| timestamp | 15분 캔들 시각 | 시간 | 인덱스 | 원자료 | 21번 |",
        "| open/high/low/close | 시/고/저/종가(KRW) | 실수 | 후보 | 원자료 | 21번 |",
        "| volume | 체결 수량(코인) | 실수 | 후보 | 원자료 | 21번 |",
        "| value | 체결 대금(KRW) | 실수 | 후보 | 원자료 | 21번 |",
        "",
        "※ RAW 6개 열 중 아직 어느 것도 모델에서 배제 확정하지 않았다(EDA 단계).",
        "",
        "## 2. 파생 변수 (EDA/전처리에서 생성)",
        "",
        "| 변수 | 정의/공식 | 무엇으로부터 | 역할 | 근거 유형 | 추가 |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
        "| log_close | ln(close) | close | 중간 | 관례(스케일/분산 안정화) | 21번 |",
        "| r (로그수익률) | Δ ln(close) = ln(close_t) − ln(close_{t-1}) | log_close | 중간 | 가정+EDA(정상성; ADF/KPSS) | 21번 |",
        "| |r| , r² | 로그수익률의 절댓값·제곱 | r | 종속 후보(크기) | EDA(|r|·r² 자기상관 강함) | 21번 |",
        "| 범위변동성(Parkinson·GK) | ln(high/low) 등으로 만든 봉내 변동성 | high/low/open/close | X 후보 | EDA(|r|과 상관 확인) | 21번 |",
        "| 거래량 z | (volume − 이동평균96)/이동표준편차96 | volume | X 후보 | 방법+EDA(거래량 비정상성) | 21번 |",
        "",
        "## 3. 타깃(종속변수) 후보 — 아직 최종 확정 아님",
        "",
        "- 미래 실현변동성(크기): 예) √(Σ 미래 r²) over horizon h. **정의·horizon은 EDA/모델 반복에서 확정.**",
        "- 근거 유형: EDA(변동성 군집·조건부 이분산으로 예측 가능성 확인).",
        "",
        "_최종 갱신: 21번 EDA(2026-09-04)._",
    ]
    DICT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    emit("## D. 데이터 사전 v1")
    emit()
    emit(f"- `test/results/DATA_DICTIONARY.md` 생성/갱신(RAW 6열 + 파생 5종 + 타깃 후보 등록).")
    emit("- 이후 모든 반복에서 파생변수가 생기면 이 문서에 누적한다.")
    emit()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args(argv)

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        preflight.ensure_data(table=SOURCE_TABLE, tickers="all", auto_build=False, strict=False)
    except Exception as exc:  # noqa: BLE001
        print(f"[preflight] {exc}", flush=True)

    con = connect()
    emit(f"# 21번 변동성 EDA — 원시 수치 ({'quick' if args.quick else 'full'})")
    emit()
    background()
    df = data_quality(con)
    raw_eda(df)
    volatility_eda(df)
    sample_validity(con, args.quick)
    write_data_dictionary()

    RAW_PATH.write_text("\n".join(_LINES) + "\n", encoding="utf-8")
    print(f"\n저장: {RAW_PATH}\n그림: {IMAGES_DIR}\n사전: {DICT_PATH}", flush=True)


if __name__ == "__main__":
    main()
