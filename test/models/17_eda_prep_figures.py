# %% [markdown]
# # 17번 준비작업 그림 드라이버 — 데이터 감사·시장요인·계절성·거래량 선행성
#
# 헤드리스 `.py` 드라이버(`#%%` 셀 + 파일 내 마크다운, AGENTS.md 2.3). ipynb는 만들지 않는다.
#
# ## 왜 (Why)
# s10~s12 세 장은 2026-07-23에 임시 실행으로 만들어져 **이미지만 커밋되고 생성 코드가 남지
# 않았다**(process.md 2026-07-23 행: "준비작업 시각화 3장 git 커밋 완료(자산 보존)").
# 보고서로 정식 서술하려면 그림 속 숫자를 인용해야 하는데, 재현 코드가 없으면 그 숫자는
# 검증 불가능한 인용이 된다 — AGENTS.md가 금지하는 "서술만 쌓고 실행에 연결 안 된" 상태다.
# 같은 이유로 professor_brief 4.6절의 "거래량 lag=1 선행 +0.20"도 코드가 남아 있지 않다.
#
# ## 무엇을 (What)
#   S10 데이터 길이 감사 : 269종목 행수 분포 — 불균형 패널/상장시점 편의의 크기를 잰다
#   S11 시장요인 PCA     : 상위 15종목 수익률 상관행렬 + 주성분 설명분산 — 공통요인의 지배력
#   S12 계절성           : 시간대(KST)·요일별 평균 |수익률| — 결정적 주기 성분
#   S13 거래량 선행성    : 거래량 z → 미래 |수익률| lead-lag(신규, process.md 미생성 항목)
#
# ## 어떻게 (How)
# 데이터 축은 `upbit_krw_candle`(KRW 전 종목, 2026-07-18 데이터 축 교정 이후의 정본)로
# 통일한다. S12·S13의 단일종목 심층 분석은 KRW-BTC를 대표로 쓰되, S13은 상위 20종목으로
# 로버스트니스를 함께 본다. 모든 숫자는 과학적표기 없이 저장하고, 그림과 별개로
# `prep_figures_raw.md`에 원시 수치를 남겨 보고서가 그 파일을 인용하게 한다.
#
# ## 기대 결과 / 반영 (Expected)
# s10~s12를 재현 가능한 상태로 되돌리고, B(변동성) 갈래의 핵심 입력인 거래량 선행신호를
# 그림·수치로 확정해 `17_eda_prep_figures_report`가 인용할 근거를 만든다.

# %%
"""17번 준비작업 그림 드라이버 본체.

실행:
    uv run test/models/17_eda_prep_figures.py
    uv run test/models/17_eda_prep_figures.py --skip s10 s11   # 일부만 재생성
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
import statsmodels.api as sm
from matplotlib.ticker import FuncFormatter
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")


def _project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "pyproject.toml").exists() and (candidate / "engine").is_dir():
            return candidate
    raise RuntimeError("engine을 담은 quantitative_trading 디렉터리를 찾지 못했다.")


try:
    _START = Path(__file__).resolve().parent
except NameError:
    _START = Path.cwd()
ROOT = _project_root(_START)

EXPERIMENT_TAG = "17_eda_direction_signal_20260722"
SOURCE_TABLE = "upbit_krw_candle"
BASE_TICKER = "KRW-BTC"
BARS_PER_DAY = 96                    # 15분봉 기준 하루
IMAGES_DIR = ROOT / "test" / "images" / EXPERIMENT_TAG
RESULTS_DIR = ROOT / "test" / "results" / EXPERIMENT_TAG
RAW_PATH = RESULTS_DIR / "prep_figures_raw.md"

# 그림에 쓸 상위 종목 수
PCA_TOP_N = 15
ROBUSTNESS_TOP_N = 20
# lead-lag 스윕 범위(±12봉 = ±3시간)
MAX_LAG = 12

_LINES: list[str] = []


def emit(text: str = "") -> None:
    """원시 수치 파일과 stdout에 동시에 남긴다."""
    print(text)
    _LINES.append(text)


def num(value: float, digits: int = 4) -> str:
    """과학적표기 없이(AGENTS.md 보고 규칙) 천단위 구분 포함 포맷."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    return f"{value:,.{digits}f}"


def connect() -> duckdb.DuckDBPyConnection:
    db_path = ROOT / "data" / "upbit_data.db"
    if not db_path.exists():
        raise SystemExit(f"DB가 없다: {db_path}")
    return duckdb.connect(str(db_path), read_only=True)


def load_ticker(con: duckdb.DuckDBPyConnection, ticker: str) -> pd.DataFrame:
    """단일 종목의 종가·거래량을 시간순으로 읽어 로그수익률을 붙인다."""
    frame = con.execute(
        f"select timestamp, close, volume from {SOURCE_TABLE} where ticker = ? order by timestamp",
        [ticker],
    ).df()
    frame["log_return"] = np.log(frame["close"]).diff()
    return frame.dropna(subset=["log_return"]).reset_index(drop=True)


# %% [markdown]
# ## S10 — 데이터 길이 감사
#
# 269종목이 같은 기간을 갖지 않는다. 상장 시점이 제각각이라 패널이 불균형(unbalanced)이고,
# 짧은 종목만 골라 빼면 생존편의(survivorship bias)가 들어온다. 얼마나 불균형한지를 먼저 잰다.

# %%
def figure_s10(con: duckdb.DuckDBPyConnection) -> None:
    lengths = con.execute(
        f"""
        select ticker, count(*) as rows, min(timestamp) as first_ts, max(timestamp) as last_ts
        from {SOURCE_TABLE}
        group by ticker
        order by rows desc
        """
    ).df()

    max_rows = int(lengths["rows"].max())
    threshold = 0.95 * max_rows
    full_history = lengths[lengths["rows"] >= threshold]
    total_rows = int(lengths["rows"].sum())

    emit("## S10 데이터 길이 감사")
    emit()
    emit(f"- 종목 수: {len(lengths):,}개, 총 행수: {total_rows:,}행")
    emit(f"- 최장 종목 행수: {max_rows:,}행 (풀히스토리 기준 95% = {threshold:,.0f}행)")
    emit(f"- 풀히스토리 종목: {len(full_history):,}개 ({100 * len(full_history) / len(lengths):.1f}%)")
    emit(
        f"- 풀히스토리 종목이 차지하는 행 비중: "
        f"{100 * int(full_history['rows'].sum()) / total_rows:.1f}%"
    )
    quartiles = lengths["rows"].quantile([0.25, 0.50, 0.75])
    emit(
        f"- 행수 사분위: 25% {quartiles.loc[0.25]:,.0f} / 중앙값 {quartiles.loc[0.50]:,.0f} / "
        f"75% {quartiles.loc[0.75]:,.0f}"
    )
    emit(f"- 최단 종목 행수: {int(lengths['rows'].min()):,}행")
    emit(
        f"- 관측 기간 전체: {lengths['first_ts'].min():%Y-%m-%d} ~ {lengths['last_ts'].max():%Y-%m-%d}"
    )
    short_share = float((lengths["rows"] < 0.5 * max_rows).mean())
    emit(f"- 절반 미만 길이 종목 비중: {100 * short_share:.1f}%")
    emit()

    fig, ax = plt.subplots(figsize=(12.8, 6.4))
    ax.hist(lengths["rows"], bins=30, color="#4c9ac9", edgecolor="white", linewidth=0.4)
    ax.axvline(threshold, color="red", linestyle="--", lw=2, label="풀히스토리 기준(95%)")
    ax.set_title(f"{len(lengths)}종목 데이터 길이 분포 (풀히스토리 {len(full_history)}개)")
    ax.set_xlabel("종목별 행 수")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "s10_data_length_audit.png", dpi=120)
    plt.close(fig)


# %% [markdown]
# ## S11 — 시장요인 PCA
#
# 종목별로 따로 모델을 세우는 게 의미가 있으려면 종목마다 다른 정보가 있어야 한다.
# 첫 주성분이 분산의 대부분을 먹는다면 "전 종목이 사실상 한 시장요인"이라는 뜻이고,
# 전종목 확장은 표본 수를 늘릴 뿐 독립 정보를 늘리지 못한다.

# %%
def figure_s11(con: duckdb.DuckDBPyConnection) -> None:
    top = con.execute(
        f"""
        select ticker, sum(value) as traded_value
        from {SOURCE_TABLE}
        group by ticker
        order by traded_value desc
        limit {PCA_TOP_N}
        """
    ).df()
    tickers = top["ticker"].tolist()

    placeholders = ", ".join("?" for _ in tickers)
    panel = con.execute(
        f"""
        select timestamp, ticker, close
        from {SOURCE_TABLE}
        where ticker in ({placeholders})
        order by timestamp
        """,
        tickers,
    ).df()

    wide = panel.pivot(index="timestamp", columns="ticker", values="close")[tickers]
    returns = np.log(wide).diff().dropna(how="any")

    corr = returns.corr()
    pca = PCA().fit(returns.values)
    explained = pca.explained_variance_ratio_

    off_diagonal = corr.values[~np.eye(len(tickers), dtype=bool)]
    stable = [t for t in tickers if t.endswith("USDT") or t.endswith("USDC")]
    risky = [t for t in tickers if t not in stable]
    risky_corr = corr.loc[risky, risky].values[~np.eye(len(risky), dtype=bool)]

    emit("## S11 시장요인 PCA")
    emit()
    emit(f"- 대상: 누적 거래대금 상위 {PCA_TOP_N}종목")
    emit(f"- 공통 관측 구간: {returns.index.min():%Y-%m-%d} ~ {returns.index.max():%Y-%m-%d} ({len(returns):,}봉)")
    emit(f"- PC1 설명분산: {100 * explained[0]:.1f}%, PC2 {100 * explained[1]:.1f}%, PC3 {100 * explained[2]:.1f}%")
    emit(f"- 상위 3개 주성분 누적: {100 * explained[:3].sum():.1f}%")
    emit(f"- 평균 쌍별 상관(전체 {PCA_TOP_N}종목): {num(float(off_diagonal.mean()), 3)}")
    emit(f"- 평균 쌍별 상관(스테이블코인 {stable} 제외): {num(float(risky_corr.mean()), 3)}")
    emit(f"- 쌍별 상관 최소/최대(스테이블 제외): {num(float(risky_corr.min()), 3)} / {num(float(risky_corr.max()), 3)}")
    for name in stable:
        others = [t for t in tickers if t != name]
        emit(f"- {name} 평균 상관(그 외 전 종목): {num(float(corr.loc[name, others].mean()), 3)}")
    emit()

    fig, axes = plt.subplots(1, 2, figsize=(19.2, 7.6), gridspec_kw={"width_ratios": [1.25, 1]})
    image = axes[0].imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    axes[0].set_xticks(range(len(tickers)))
    axes[0].set_xticklabels(tickers, rotation=90)
    axes[0].set_yticks(range(len(tickers)))
    axes[0].set_yticklabels(tickers)
    axes[0].set_title(f"상위 {PCA_TOP_N}종목 수익률 상관행렬")
    fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)

    axes[1].bar(range(1, len(explained) + 1), explained, color="#d62728")
    axes[1].set_title(f"PCA 설명분산비율 (PC1={100 * explained[0]:.1f}%)")
    axes[1].set_xlabel("주성분")
    axes[1].set_ylabel("설명 분산 비율")
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "s11_market_factor_pca.png", dpi=120)
    plt.close(fig)


# %% [markdown]
# ## S12 — 계절성 (시간대·요일)
#
# 변동성에 결정적(deterministic) 주기가 있으면, 그 주기는 모델이 "예측"한 것이 아니라
# 달력이 알려주는 것이다. 주기 성분의 크기를 먼저 재야 GARCH류 확률적 성분과 분리할 수 있다.

# %%
def figure_s12(con: duckdb.DuckDBPyConnection) -> None:
    frame = load_ticker(con, BASE_TICKER)
    frame["abs_return"] = frame["log_return"].abs()
    frame["hour"] = frame["timestamp"].dt.hour
    frame["dow"] = frame["timestamp"].dt.dayofweek

    by_hour = frame.groupby("hour")["abs_return"].mean()
    by_dow = frame.groupby("dow")["abs_return"].mean()
    dow_labels = ["월", "화", "수", "목", "금", "토", "일"]

    overall = float(frame["abs_return"].mean())

    emit("## S12 계절성 (시간대·요일)")
    emit()
    emit(f"- 대상: {BASE_TICKER}, {len(frame):,}봉 (타임스탬프는 KST — pyupbit 수집 기준)")
    emit(f"- 전체 평균 |수익률|: {num(overall, 6)}")
    emit(
        f"- 시간대 최대: {int(by_hour.idxmax())}시 {num(float(by_hour.max()), 6)} / "
        f"최소: {int(by_hour.idxmin())}시 {num(float(by_hour.min()), 6)} "
        f"→ 최대/최소 비율 {num(float(by_hour.max() / by_hour.min()), 3)}배"
    )
    emit(
        f"- 요일 최대: {dow_labels[int(by_dow.idxmax())]} {num(float(by_dow.max()), 6)} / "
        f"최소: {dow_labels[int(by_dow.idxmin())]} {num(float(by_dow.min()), 6)} "
        f"→ 최대/최소 비율 {num(float(by_dow.max() / by_dow.min()), 3)}배"
    )
    weekday = float(frame[frame["dow"] < 5]["abs_return"].mean())
    weekend = float(frame[frame["dow"] >= 5]["abs_return"].mean())
    emit(f"- 평일 평균 {num(weekday, 6)} vs 주말 평균 {num(weekend, 6)} (주말이 {100 * (1 - weekend / weekday):.1f}% 낮음)")
    emit(
        "- 시간대 프로파일(시:평균|수익률|): "
        + ", ".join(f"{h}시 {num(float(v), 6)}" for h, v in by_hour.items())
    )
    emit(
        "- 요일 프로파일: "
        + ", ".join(f"{dow_labels[d]} {num(float(v), 6)}" for d, v in by_dow.items())
    )
    # 주기 성분이 |수익률| 총분산에서 차지하는 비중(집단평균으로 설명되는 분산 = eta^2)
    hour_eta = float(frame.groupby("hour")["abs_return"].transform("mean").var() / frame["abs_return"].var())
    dow_eta = float(frame.groupby("dow")["abs_return"].transform("mean").var() / frame["abs_return"].var())
    emit(f"- |수익률| 분산 중 시간대 고정효과 설명분(eta^2): {100 * hour_eta:.2f}%")
    emit(f"- |수익률| 분산 중 요일 고정효과 설명분(eta^2): {100 * dow_eta:.2f}%")
    emit()

    fig, axes = plt.subplots(1, 2, figsize=(18, 6.4))
    axes[0].bar(by_hour.index, by_hour.values, color="#1f77b4")
    axes[0].set_title("시간대별(KST) 평균 |수익률|")
    axes[0].set_xlabel("시(hour)")
    axes[1].bar(range(7), by_dow.values, color="#ff7f0e")
    axes[1].set_xticks(range(7))
    axes[1].set_xticklabels(dow_labels)
    axes[1].set_title("요일별 평균 |수익률|")
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "s12_seasonality.png", dpi=120)
    plt.close(fig)


# %% [markdown]
# ## S13 — 거래량 선행지표 (신규)
#
# process.md 2026-07-23 행의 미생성 항목. professor_brief 4.6절이 "거래량 z-score(과거)와
# 미래 |수익률| 상관 lag=1에서 +0.20"이라고 인용하지만 그 코드가 남아 있지 않다. 정의를
# 세 가지(원시 전역 z / 로그 전역 z / 로그 롤링 z)로 나눠 재현하고, 어느 정의에서 나온
# 수치인지까지 남긴다.

# %%
def _volume_variants(frame: pd.DataFrame) -> dict[str, pd.Series]:
    volume = frame["volume"].astype(float)
    log_volume = np.log1p(volume)
    rolling_mean = log_volume.rolling(BARS_PER_DAY).mean()
    rolling_std = log_volume.rolling(BARS_PER_DAY).std()
    return {
        "원시 거래량 전역 z": (volume - volume.mean()) / volume.std(),
        "로그 거래량 전역 z": (log_volume - log_volume.mean()) / log_volume.std(),
        f"로그 거래량 롤링{BARS_PER_DAY} z": (log_volume - rolling_mean) / rolling_std,
    }


def _lead_lag_profile(signal: pd.Series, target: pd.Series, max_lag: int) -> pd.Series:
    """corr(signal_{t-k}, target_t). k>0이면 signal이 target을 선행한다."""
    out = {}
    for k in range(-max_lag, max_lag + 1):
        out[k] = float(signal.shift(k).corr(target))
    return pd.Series(out)


def figure_s13(con: duckdb.DuckDBPyConnection) -> None:
    frame = load_ticker(con, BASE_TICKER)
    abs_return = frame["log_return"].abs()
    variants = _volume_variants(frame)

    emit("## S13 거래량 선행성 (거래량 → 미래 |수익률|)")
    emit()
    emit(f"- 대상: {BASE_TICKER}, {len(frame):,}봉. k>0 = 거래량이 |수익률|을 k봉 선행")
    emit()
    emit("| 거래량 정의 | lag=1 | lag=2 | lag=4 | lag=8 | 동시(k=0) | 역방향(k=-1) |")
    emit("| :--- | ---: | ---: | ---: | ---: | ---: | ---: |")
    profiles = {}
    for name, series in variants.items():
        profile = _lead_lag_profile(series, abs_return, MAX_LAG)
        profiles[name] = profile
        emit(
            f"| {name} | {num(profile[1], 3)} | {num(profile[2], 3)} | {num(profile[4], 3)} | "
            f"{num(profile[8], 3)} | {num(profile[0], 3)} | {num(profile[-1], 3)} |"
        )
    emit()

    # 방향과의 상관 — 크기에만 신호가 있고 방향엔 없다는 대조군
    signed = frame["log_return"]
    for name, series in variants.items():
        direction_corr = float(series.shift(1).corr(np.sign(signed)))
        emit(f"- {name}: 부호(방향)와의 lag=1 상관 {num(direction_corr, 4)}")
    emit()

    primary_name = f"로그 거래량 롤링{BARS_PER_DAY} z"
    primary = variants[primary_name]
    primary_profile = profiles[primary_name]

    # 거래량 분위 → 다음 봉 |수익률| 계단
    valid = pd.DataFrame({"z": primary, "next_abs": abs_return.shift(-1)}).dropna()
    valid["decile"] = pd.qcut(valid["z"], 10, labels=False, duplicates="drop")
    staircase = valid.groupby("decile")["next_abs"].mean()
    baseline = float(valid["next_abs"].mean())
    emit(f"- 거래량 z 십분위별 다음 봉 평균 |수익률| (전체 평균 {num(baseline, 6)}):")
    for decile, value in staircase.items():
        emit(f"  - {int(decile) + 1}분위: {num(float(value), 6)} (전체 대비 {value / baseline:.2f}배)")
    emit(
        f"- 최상위 10% 대 최하위 10% 배율: "
        f"{num(float(staircase.iloc[-1] / staircase.iloc[0]), 2)}배"
    )
    emit()

    # 증분 설명력 — |수익률|의 자기상관(0.36)을 이미 넣은 뒤에도 거래량이 남는 정보를 주는가.
    # 이게 "B 모델 입력으로 쓸 수 있는가"의 실제 판정 기준이다. lead-lag 상관만으로는
    # 거래량·변동성의 동시성분(k=0에서 가장 큼)에 속을 수 있다.
    design = pd.DataFrame(
        {
            "target": abs_return,
            "abs_lag1": abs_return.shift(1),
            "abs_day": abs_return.shift(1).rolling(BARS_PER_DAY).mean(),
            "vol_z_lag1": primary.shift(1),
        }
    ).dropna()

    def _ols_r2(columns: list[str]) -> tuple[float, sm.regression.linear_model.RegressionResultsWrapper]:
        exog = sm.add_constant(design[columns])
        model = sm.OLS(design["target"], exog).fit(cov_type="HAC", cov_kwds={"maxlags": BARS_PER_DAY})
        return float(model.rsquared), model

    r2_persist, _ = _ols_r2(["abs_lag1", "abs_day"])
    r2_volume_only, _ = _ols_r2(["vol_z_lag1"])
    r2_both, model_both = _ols_r2(["abs_lag1", "abs_day", "vol_z_lag1"])

    emit("### 증분 설명력 (거래량이 |수익률| 자체의 과거를 넘어 정보를 더하는가)")
    emit()
    emit(f"- 표본: {len(design):,}행, 표준오차는 HAC(Newey-West, maxlags={BARS_PER_DAY})")
    emit(f"- 지속성만 (|r|(t-1) + 직전 1일 평균|r|): R^2 = {num(r2_persist, 4)}")
    emit(f"- 거래량만 (거래량 z(t-1)): R^2 = {num(r2_volume_only, 4)}")
    emit(f"- 둘 다: R^2 = {num(r2_both, 4)} → 거래량의 증분 R^2 = {num(r2_both - r2_persist, 4)}")
    coefficient = float(model_both.params["vol_z_lag1"])
    t_value = float(model_both.tvalues["vol_z_lag1"])
    p_value = float(model_both.pvalues["vol_z_lag1"])
    emit(
        f"- 거래량 계수: {num(coefficient, 6)} (HAC t = {num(t_value, 2)}, p = {num(p_value, 4)})"
    )
    emit(
        f"- 해석: 지속성 대비 설명력이 {100 * (r2_both - r2_persist) / max(r2_persist, 1e-12):.1f}% "
        f"늘어난다."
    )
    emit()

    # 전종목 로버스트니스
    top = con.execute(
        f"""
        select ticker, sum(value) as traded_value
        from {SOURCE_TABLE}
        group by ticker
        order by traded_value desc
        limit {ROBUSTNESS_TOP_N}
        """
    ).df()
    per_ticker = {}
    for ticker in top["ticker"]:
        sub = load_ticker(con, ticker)
        if len(sub) < 5 * BARS_PER_DAY:
            continue
        sub_variants = _volume_variants(sub)
        per_ticker[ticker] = float(
            sub_variants[primary_name].shift(1).corr(sub["log_return"].abs())
        )
    per_ticker_series = pd.Series(per_ticker).sort_values(ascending=False)
    emit(f"- 상위 {ROBUSTNESS_TOP_N}종목 lag=1 상관({primary_name}): "
         f"평균 {num(float(per_ticker_series.mean()), 3)}, "
         f"중앙값 {num(float(per_ticker_series.median()), 3)}, "
         f"최소 {num(float(per_ticker_series.min()), 3)}, "
         f"최대 {num(float(per_ticker_series.max()), 3)}")
    positive = int((per_ticker_series > 0).sum())
    emit(f"- 양(+)의 선행성을 보인 종목: {positive}/{len(per_ticker_series)}개")
    emit("- 종목별: " + ", ".join(f"{t} {num(v, 3)}" for t, v in per_ticker_series.items()))
    emit()

    fig, axes = plt.subplots(1, 3, figsize=(21, 6.2))

    for name, profile in profiles.items():
        axes[0].plot(profile.index, profile.values, marker="o", ms=3, lw=1.4, label=name)
    axes[0].axvline(0, color="black", lw=0.8)
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_title("거래량 → |수익률| lead-lag 상관\n(k>0: 거래량이 선행)")
    axes[0].set_xlabel("시차 k (봉, 15분)")
    axes[0].set_ylabel("상관계수")
    axes[0].legend(fontsize=9)

    axes[1].bar(range(1, len(staircase) + 1), staircase.values, color="#2ca02c")
    axes[1].axhline(baseline, color="red", ls="--", lw=1.5, label="전체 평균")
    axes[1].set_title("거래량 z 십분위 → 다음 봉 평균 |수익률|")
    axes[1].set_xlabel("거래량 z 십분위")
    axes[1].set_xticks(range(1, len(staircase) + 1))
    axes[1].legend()

    axes[2].bar(range(len(per_ticker_series)), per_ticker_series.values, color="#9467bd")
    axes[2].set_xticks(range(len(per_ticker_series)))
    axes[2].set_xticklabels(per_ticker_series.index, rotation=90, fontsize=8)
    axes[2].axhline(float(per_ticker_series.mean()), color="red", ls="--", lw=1.5, label="평균")
    axes[2].set_title(f"상위 {ROBUSTNESS_TOP_N}종목 lag=1 선행 상관")
    axes[2].set_ylabel("상관계수")
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "s13_volume_lead_volatility.png", dpi=120)
    plt.close(fig)


# %%
SECTIONS = {"s10": figure_s10, "s11": figure_s11, "s12": figure_s12, "s13": figure_s13}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="17번 준비작업 그림(s10~s13) 생성")
    parser.add_argument("--skip", nargs="*", default=[], choices=list(SECTIONS), help="건너뛸 섹션")
    args = parser.parse_args(argv)

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    emit(f"# 17번 준비작업 그림 원시 수치 (source={SOURCE_TABLE})")
    emit()
    emit("생성: `uv run test/models/17_eda_prep_figures.py`. 이 파일은 자동 생성물이며,")
    emit("보고서(`17_eda_prep_figures_report_*.md`)가 여기의 숫자를 인용한다.")
    emit()

    con = connect()
    try:
        for name, fn in SECTIONS.items():
            if name in args.skip:
                emit(f"## {name.upper()} — 건너뜀")
                emit()
                continue
            fn(con)
    finally:
        con.close()

    RAW_PATH.write_text("\n".join(_LINES) + "\n", encoding="utf-8")
    print(f"[prep-figures] 원시 수치 저장: {RAW_PATH}")
    print(f"[prep-figures] 그림 저장: {IMAGES_DIR}")


if __name__ == "__main__":
    main(sys.argv[1:])
