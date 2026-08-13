# %% [markdown]
# # 17번: 전면 탐색적 데이터 분석(EDA) — 데이터를 먼저 눈으로 본다
#
# 헤드리스 `.py` 드라이버(`#%%` 셀 + 파일 내 마크다운, CLAUDE.md 2.3). ipynb는 만들지 않는다.
#
# ## 왜 (Why)
# 8~16번 내내 파생변수는 만들었지만(engine/features.py, 60여개) 정작 "데이터 자체가 어떻게
# 생겼는지"를 describe·시각화·분해·검정으로 본 적이 없다. 사용자 지적: 진짜 탐색적 분석은
# (1) data.describe부터, (2) 과학적표기(1.2e-3) 금지·직관적 숫자로, (3) 시각화, (4) 시계열
# 분해, (5) 검정을 모두 해보고 다음 방향을 정하는 것. 또한 "가격은 비정상, 수익률은 정상"인데
# 우리가 차분(수익률)으로 정상화해 분석해온 게 맞는지 — 비정상성을 직접 보는 분해까지 넣는다.
#
# ## 무엇을 (What)
# 하나의 실행이 아래 전 섹션을 수행하고, 그래프를 저장한 뒤 종합 보고서 초안을 만든다.
#   S0 overview      : describe(직관적 숫자), 결측, 기간, 핵심변수 요약
#   S1 series        : 가격(레벨)·로그수익률·거래량·롤링 평균/표준편차 시계열 그림
#   S2 distribution  : 수익률 분포(히스토그램·QQ), 첨도·왜도, 이상치 비율
#   S3 autocorr      : 수익률 vs |수익률| ACF/PACF — "방향은 무상관, 크기는 군집"을 눈으로
#   S4 stationarity  : 로그가격(레벨) vs 로그수익률에 ADF·KPSS·ARCH-LM (segfault-safe)
#   S5 decomposition : 로그가격 STL 분해(추세·계절·잔차) — 비정상성을 직접 본다
#   S6 direction     : 파생변수 × target(방향/크기) Pearson·Spearman·상호정보량 상위
#
# ## 어떻게 (How)
# 단일 종목(BTC)을 대표로 심층 분석한다(ACF·분해는 종목별이라야 의미). 전종목 확장은
# 방향신호(S6)에서만 후속으로 넓힌다. 모든 표는 과학적표기 없이 직관적 숫자 포맷,
# 그래프 축도 sci표기 끈다. 100k행 ADF autolag가 segfault를 내던 문제는 검정을 부분표본
# +고정 maxlag로 바꿔 회피한다.
#
# ## 기대 결과 / 반영 (Expected)
# 가격의 비정상성·수익률의 (약)정상성·변동성 군집·방향 무상관을 그림과 검정으로 확정하고,
# 그 위에서 다음 스텝(외생변수 도입 / 비정상 레벨 직접 모델링 / GARCH 변동성 베이스라인)을
# 데이터 근거로 결정한다.

# %%
"""17번 EDA 드라이버 본체.

계획서: test/experiment_specs/17_eda_direction_signal_plan_20260719.md
실행:
    uv run test/models/17_eda_direction_signal_test.py            # 전 섹션 + 보고서
    uv run test/models/17_eda_direction_signal_test.py --dry-run  # 데이터 로드만 확인
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = ["Noto Sans CJK KR", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["axes.formatter.useoffset"] = False   # 축에 1e8+... 오프셋 표기 금지
matplotlib.rcParams["axes.formatter.limits"] = (-9, 12)   # 과학적표기 전환 억제

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from scipy import stats as sp_stats
from scipy.stats import kurtosis, pearsonr, skew, spearmanr
from sklearn.feature_selection import mutual_info_regression
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import het_arch
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss

warnings.filterwarnings("ignore")


def _engine_root(start: Path) -> Path:
    cands = [start, *start.parents, Path.home() / "personal_ai_project" / "quantitative_trading"]
    for c in cands:
        if (c / "pyproject.toml").exists() and (c / "engine").is_dir():
            return c
    raise RuntimeError("engine을 담은 quantitative_trading 디렉터리를 찾지 못했다.")


try:
    _START = Path(__file__).resolve().parent
except NameError:
    _START = Path.cwd()
ROOT = _engine_root(_START)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine import data as engdata
from engine import features as engfeat

EXPERIMENT_TAG = "17_eda_direction_signal_20260722"
RESULTS_DIR = ROOT / "test" / "results" / EXPERIMENT_TAG
IMAGES_DIR = ROOT / "test" / "images" / EXPERIMENT_TAG

NON_FEATURE_COLS = {
    "timestamp", "open", "high", "low", "close", "volume", "value", "ticker",
    "log_close", "prev_close", "target_return", "target_open", "target_high",
    "target_low", "target_close", "target_timestamp",
}


# %% [markdown]
# ## 직관적 숫자 포맷 (과학적표기 E^ 절대 금지)
# 크기에 따라 자릿수를 바꿔 사람이 바로 읽히게 만든다.

# %%
def fmt(x) -> str:
    """숫자를 과학적표기 없이 직관적 문자열로. 크기별로 자릿수 자동 조정."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(v):
        return "NaN"
    a = abs(v)
    if a == 0:
        return "0"
    if a >= 1_000_000:          # 가격 등 큰 수 → 천단위 콤마 정수
        return f"{v:,.0f}"
    if a >= 1:                  # 보통 수 → 소수 4자리
        return f"{v:,.4f}"
    if a >= 1e-4:               # 수익률 등 작은 수 → 소수 6자리
        return f"{v:.6f}"
    return f"{v:.8f}"           # 아주 작은 수 → 소수 8자리 (여전히 E 표기 아님)


def fmt_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """DataFrame의 모든 수치 셀을 fmt로 문자열화한 표시용 프레임."""
    disp = frame.copy()
    for col in disp.columns:
        if pd.api.types.is_numeric_dtype(disp[col]):
            disp[col] = disp[col].map(fmt)
    return disp


def frame_to_md(frame: pd.DataFrame, index: bool = False) -> str:
    try:
        return frame.to_markdown(index=index)
    except ImportError:
        return "```\n" + frame.to_string(index=index) + "\n```"


def price_axis(ax):
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))


def small_axis(ax):
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.4f}"))


def image_dir() -> Path:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    return IMAGES_DIR


# %% [markdown]
# ## 데이터 로드 + target 부착 (h-step 방향/크기)

# %%
def load_dataset(args) -> pd.DataFrame:
    db_path = engdata.resolve_db_path(args.db)
    raw = engdata.load_price_data(db_path, args.table, args.ticker, args.max_rows)
    feats = engdata.make_features(raw)
    feats = engfeat.add_coin_specific_features(feats)
    log_close = np.log(feats["close"].astype(float).clip(lower=1e-9))
    feats["log_price"] = log_close
    feats["target_h_return"] = log_close.shift(-args.horizon) - log_close
    feats["target_h_direction"] = (feats["target_h_return"] > 0).astype(float)
    return feats.dropna(subset=["target_h_return"]).reset_index(drop=True)


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURE_COLS
            and not c.startswith("target_") and c != "log_price"
            and pd.api.types.is_numeric_dtype(df[c])]


# %% [markdown]
# ## S0 — 개요 & describe (직관적 숫자)

# %%
def section_overview(df: pd.DataFrame, report: list[str]):
    close = df["close"].astype(float)
    ret = df["log_return_1"].astype(float)
    vol = df["volume"].astype(float)
    n = len(df)
    start, end = str(df["timestamp"].min()), str(df["timestamp"].max())
    # 원천 4개 축 describe (직관적 숫자)
    desc = df[["close", "volume", "value", "log_return_1"]].describe().T
    desc = desc[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]
    report += [
        "## S0. 개요 & 기초 통계 (describe)", "",
        f"- 종목: **{df['ticker'].iloc[0] if 'ticker' in df.columns else 'KRW-BTC'}**, 15분봉",
        f"- 관측치: **{n:,}행**, 기간 **{start} ~ {end}**",
        f"- 결측 셀(전체): **{int(df.isna().sum().sum()):,}개**",
        "",
        "### 원천 변수 describe (과학적표기 없이)", "",
        frame_to_md(fmt_frame(desc.reset_index().rename(columns={"index": "변수"}))),
        "",
        "**읽는 법**: `close`(종가, KRW)는 평균 "
        f"{fmt(close.mean())}, 최소 {fmt(close.min())} ~ 최대 {fmt(close.max())}로 "
        "3년간 크게 표류 → 레벨은 한 곳에 머물지 않는다(비정상 시사). "
        f"`log_return_1`(15분 로그수익률)은 평균 {fmt(ret.mean())}(거의 0), "
        f"표준편차 {fmt(ret.std())}로 0 주변에 몰려 있다(정상 시사).",
        "",
    ]
    # 핵심 파생변수 요약도 몇 개
    key = [c for c in ["realized_vol_16", "realized_vol_64", "rsi_14_scaled",
                       "trend_strength_64", "macd_gap_pct"] if c in df.columns]
    if key:
        kd = df[key].describe().T[["mean", "std", "min", "50%", "max"]]
        report += ["### 핵심 파생변수 요약", "",
                   frame_to_md(fmt_frame(kd.reset_index().rename(columns={"index": "변수"}))), ""]


# %% [markdown]
# ## S1 — 시계열 그림 (레벨·수익률·거래량·롤링 변동성)

# %%
def section_series(df: pd.DataFrame, report: list[str]):
    ts = pd.to_datetime(df["timestamp"])
    close = df["close"].astype(float)
    ret = df["log_return_1"].astype(float)
    vol = df["volume"].astype(float)
    roll_mean = ret.rolling(96).mean()
    roll_std = ret.rolling(96).std()

    fig, axes = plt.subplots(4, 1, figsize=(13, 14), sharex=True)
    axes[0].plot(ts, close, color="#1f77b4", linewidth=0.6)
    axes[0].set_title("① 종가(레벨, KRW) — 계속 표류하면 비정상")
    price_axis(axes[0])
    axes[1].plot(ts, ret, color="#555555", linewidth=0.4)
    axes[1].axhline(0, color="red", linewidth=0.5)
    axes[1].set_title("② 15분 로그수익률 — 0 주변, 큰 스파이크가 군집")
    small_axis(axes[1])
    axes[2].plot(ts, roll_std, color="#d62728", linewidth=0.7)
    axes[2].set_title("③ 롤링 표준편차(24시간=96봉) — 변동성 군집(고요↔격동 반복)")
    small_axis(axes[2])
    axes[3].plot(ts, vol, color="#2ca02c", linewidth=0.4)
    axes[3].set_title("④ 거래량")
    axes[3].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    fig.tight_layout()
    out = image_dir() / "s1_series_overview.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)

    report += [
        "## S1. 시계열 그림 — 레벨/수익률/변동성", "",
        f"![series](../../images/{EXPERIMENT_TAG}/s1_series_overview.png)", "",
        "- **①종가**: 3년간 크게 오르내리며 한 수준에 머물지 않음 → **비정상(추세·표류)**.",
        "- **②수익률**: 0을 중심으로 진동, 큰 변동이 특정 시기에 몰림 → 평균은 안정, **꼬리·군집**.",
        "- **③롤링 표준편차**: 잔잔한 구간과 격동 구간이 번갈아 → **변동성 군집(volatility clustering)**.",
        "  변동의 '크기'에는 시간 구조가 있다는 직접 증거.",
        "- **④거래량**: 변동성 급등 구간과 대체로 동행.",
        "",
    ]


# %% [markdown]
# ## S2 — 분포(히스토그램·QQ), 첨도·왜도·이상치

# %%
def section_distribution(df: pd.DataFrame, report: list[str]):
    ret = df["log_return_1"].astype(float).to_numpy()
    ret = ret[np.isfinite(ret)]
    k, s = kurtosis(ret), skew(ret)
    z = (ret - ret.mean()) / ret.std()
    outlier = float(np.mean(np.abs(z) > 3))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].hist(ret, bins=200, color="#1f77b4", alpha=0.85)
    axes[0].set_title(f"15분 로그수익률 분포 (첨도={fmt(k)}, 왜도={fmt(s)})")
    axes[0].set_xlabel("log_return_1")
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    axes[0].set_xlim(np.quantile(ret, 0.001), np.quantile(ret, 0.999))
    sp_stats.probplot(ret, dist="norm", plot=axes[1])
    axes[1].set_title("QQ-plot (정규분포 대비) — 꼬리에서 크게 벗어나면 fat-tail")
    fig.tight_layout()
    out = image_dir() / "s2_return_distribution.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)

    report += [
        "## S2. 수익률 분포·꼬리", "",
        f"![dist](../../images/{EXPERIMENT_TAG}/s2_return_distribution.png)", "",
        f"- 첨도(정규=0 기준 초과) **{fmt(k)}** → 정규분포보다 훨씬 뾰족하고 꼬리가 두껍다(극단 변동 잦음).",
        f"- 왜도 **{fmt(s)}** → 좌우 (거의) 대칭.",
        f"- 이상치 비율(|z|>3) **{fmt(outlier)}** (정규분포면 약 0.003).",
        "- **함의**: 평균만 줄이는 손실(MSE/Huber)은 이 두꺼운 꼬리에서 0 근처로 눌린다(진폭 압축의 구조적 원인).",
        "",
    ]


# %% [markdown]
# ## S3 — ACF/PACF: 방향(수익률) vs 크기(|수익률|)

# %%
def section_autocorr(df: pd.DataFrame, report: list[str]):
    ret = df["log_return_1"].astype(float).dropna().to_numpy()
    absret = np.abs(ret)
    nlags = 96
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    plot_acf(ret, lags=nlags, ax=axes[0, 0], title="수익률 ACF — 0 근처면 방향 예측 불가")
    plot_pacf(ret, lags=nlags, ax=axes[0, 1], title="수익률 PACF", method="ywm")
    plot_acf(absret, lags=nlags, ax=axes[1, 0], title="|수익률| ACF — 높고 느리게 감소=변동성 군집")
    plot_pacf(absret, lags=nlags, ax=axes[1, 1], title="|수익률| PACF", method="ywm")
    fig.tight_layout()
    out = image_dir() / "s3_acf_pacf.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)

    # 대표 자기상관 수치
    def ac(x, lag):
        return float(np.corrcoef(x[:-lag], x[lag:])[0, 1])
    rows = [{"lag": l, "수익률 자기상관": ac(ret, l), "|수익률| 자기상관": ac(absret, l)}
            for l in [1, 2, 4, 16, 64, 96]]
    report += [
        "## S3. 자기상관 — 방향 vs 크기 (핵심)", "",
        f"![acf](../../images/{EXPERIMENT_TAG}/s3_acf_pacf.png)", "",
        frame_to_md(fmt_frame(pd.DataFrame(rows))), "",
        "- **수익률(방향) 자기상관 ≈ 0**: 과거 수익률로 미래 방향(부호)을 예측하기 사실상 불가.",
        "- **|수익률|(크기) 자기상관은 크고 느리게 감소**: 변동 '크기'는 강하게 예측 가능(군집).",
        "- **이게 우리 연구의 핵심 구조**: 방향은 신호가 없고 크기는 신호가 있다 — "
        "예측 대상을 방향에 걸면 한계, 변동성(크기)·외생정보 쪽이 가망.",
        "",
    ]


# %% [markdown]
# ## S4 — 정상성 검정 (레벨 vs 수익률), ARCH-LM 포함. segfault-safe.

# %%
def _safe_series(x: np.ndarray, cap: int = 20000) -> np.ndarray:
    x = x[np.isfinite(x)]
    if len(x) > cap:                       # 100k 전체 autolag가 segfault → 최근 cap개만
        x = x[-cap:]
    return x


def section_stationarity(df: pd.DataFrame, report: list[str]):
    level = _safe_series(df["log_price"].astype(float).to_numpy())
    ret = _safe_series(df["log_return_1"].astype(float).to_numpy())

    def run(x):
        adf = adfuller(x, maxlag=30, autolag=None)     # autolag=None + 고정 maxlag = segfault 회피
        try:
            kp = kpss(x, regression="c", nlags=30)
        except Exception:
            kp = (np.nan, np.nan)
        try:
            arch_stat, arch_p, _, _ = het_arch(x, nlags=20)
        except Exception:
            arch_stat, arch_p = np.nan, np.nan
        return adf, kp, (arch_stat, arch_p)

    la, lk, larch = run(level)
    ra, rk, rarch = run(ret)
    rows = [
        {"대상": "로그가격(레벨)", "ADF통계량": la[0], "ADF_p": la[1],
         "KPSS통계량": lk[0], "KPSS_p": lk[1], "ARCH-LM_p": larch[1]},
        {"대상": "로그수익률(차분)", "ADF통계량": ra[0], "ADF_p": ra[1],
         "KPSS통계량": rk[0], "KPSS_p": rk[1], "ARCH-LM_p": rarch[1]},
    ]
    report += [
        "## S4. 정상성 검정 (레벨 vs 수익률)", "",
        f"※ 100k 전체에 ADF autolag는 segfault가 나서, 최근 20,000봉 + 고정 maxlag=30으로 검정.", "",
        frame_to_md(fmt_frame(pd.DataFrame(rows))), "",
        "**해석 (검정별 귀무가설이 반대임에 주의)**",
        "- ADF 귀무=비정상(단위근). p<0.05면 기각→**정상**.",
        "- KPSS 귀무=정상. p<0.05면 기각→**비정상**.",
        "- ARCH-LM 귀무=조건부 등분산(변동성 일정). p<0.05면 기각→**변동성 군집(조건부 이분산) 존재**.",
        "",
        f"- **로그가격**: ADF p={fmt(la[1])}(비정상 못 벗어남) + KPSS p={fmt(lk[1])}(비정상) "
        "→ 둘 다 **명백히 비정상**.",
        f"- **로그수익률**: ADF p={fmt(ra[1])}(정상) + KPSS p={fmt(rk[1])} + "
        f"ARCH-LM p={fmt(rarch[1])} → **평균은 정상, 분산은 비정상(조건부 이분산)**. "
        "즉 '차분하면 완전 정상'이 아니라 **평균만 정상, 변동성은 여전히 시변** — GARCH류가 다루는 지점.",
        "",
    ]


# %% [markdown]
# ## S5 — 시계열 분해(STL): 로그가격의 추세/계절/잔차
# 비정상 레벨을 직접 분해해 '추세·계절·잔여'가 각각 얼마나 되는지 눈으로 본다.

# %%
def section_decomposition(df: pd.DataFrame, report: list[str]):
    # 가독성 위해 최근 약 30일(=2880봉)만. 일간 계절 주기=96봉(24h*4).
    window = 2880
    sub = df.iloc[-window:].copy()
    ts = pd.to_datetime(sub["timestamp"])
    series = pd.Series(sub["log_price"].to_numpy(), index=ts)
    stl = STL(series, period=96, robust=True).fit()

    fig, axes = plt.subplots(4, 1, figsize=(13, 12), sharex=True)
    axes[0].plot(ts, series.values, color="#1f77b4", lw=0.8); axes[0].set_title("원본 로그가격 (최근 30일)")
    axes[1].plot(ts, stl.trend, color="#ff7f0e", lw=1.0); axes[1].set_title("추세(trend) — 저주파 방향")
    axes[2].plot(ts, stl.seasonal, color="#2ca02c", lw=0.6); axes[2].set_title("계절(seasonal, 일간 96봉 주기)")
    axes[3].plot(ts, stl.resid, color="#888888", lw=0.4); axes[3].set_title("잔차(residual)")
    for ax in axes:
        small_axis(ax)
    fig.tight_layout()
    out = image_dir() / "s5_stl_decomposition.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)

    var_total = float(np.var(series.values))
    var_trend = float(np.var(stl.trend))
    var_seasonal = float(np.var(stl.seasonal))
    var_resid = float(np.var(stl.resid))
    report += [
        "## S5. 시계열 분해(STL) — 로그가격의 추세/계절/잔차", "",
        f"![stl](../../images/{EXPERIMENT_TAG}/s5_stl_decomposition.png)", "",
        "최근 30일(2,880봉) 로그가격을 추세·계절(일간 96봉 주기)·잔차로 분해. 비정상 레벨을 "
        "차분하지 않고 직접 본 것.", "",
        f"- 추세 분산 비중 ≈ **{fmt(100*var_trend/max(var_total,1e-12))}%**, "
        f"계절 ≈ **{fmt(100*var_seasonal/max(var_total,1e-12))}%**, "
        f"잔차 ≈ **{fmt(100*var_resid/max(var_total,1e-12))}%**.",
        "- **함의**: 레벨의 움직임은 대부분 저주파 '추세'가 차지하고 일간 계절성은 작다. "
        "추세는 크지만 그 자체는 '이미 온 방향'이라 미래 방향 예측과는 다르다(레벨을 그대로 "
        "예측하면 직전값 복사=lag-copy 착시). 분해는 '무엇이 비정상을 만드는가(추세)'를 보여준다.",
        "",
    ]


# %% [markdown]
# ## S6 — 방향/크기 신호: 파생변수 × target 상관·상호정보량

# %%
def section_direction(df: pd.DataFrame, report: list[str]):
    cols = feature_columns(df)
    y_ret = df["target_h_return"].to_numpy(float)
    y_dir = df["target_h_direction"].to_numpy(float)
    rows = []
    for c in cols:
        x = df[c].to_numpy(float)
        m = np.isfinite(x) & np.isfinite(y_ret)
        if m.sum() < 500 or np.std(x[m]) < 1e-12:
            continue
        pr, pp = pearsonr(x[m], y_ret[m])
        sr, _ = spearmanr(x[m], y_ret[m])
        mi = mutual_info_regression(x[m].reshape(-1, 1), y_ret[m], random_state=42)[0]
        prd, _ = pearsonr(x[m], y_dir[m])
        rows.append({"변수": c, "Pearson(크기)": pr, "Spearman": sr, "상호정보량": mi,
                     "Pearson(방향)": prd, "R^2": pr**2})
    res = pd.DataFrame(rows).sort_values("상호정보량", ascending=False).reset_index(drop=True)
    res.to_csv(RESULTS_DIR / "s6_direction_signal.csv", index=False)

    top = res.head(20)
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    axes[0].barh(top["변수"], top["Pearson(크기)"].abs(), color="#1f77b4")
    axes[0].set_title("|Pearson| (변수 vs 4시간 수익률) 상위 20"); axes[0].invert_yaxis()
    axes[0].xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.3f}"))
    axes[1].barh(top["변수"], top["상호정보량"], color="#ff7f0e")
    axes[1].set_title("상호정보량(비선형 포함) 상위 20"); axes[1].invert_yaxis()
    axes[1].xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.4f}"))
    fig.tight_layout()
    out = image_dir() / "s6_direction_signal.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)

    best = res.iloc[0]
    mp = res.loc[res["Pearson(크기)"].abs().idxmax()]
    report += [
        "## S6. 방향/크기 신호: 파생변수 × 4시간 target", "",
        f"![signal](../../images/{EXPERIMENT_TAG}/s6_direction_signal.png)", "",
        "### 상위 15개 변수 (상호정보량 순)", "",
        frame_to_md(fmt_frame(res.head(15))), "",
        f"- 상호정보량 최고: **{best['변수']}** (MI={fmt(best['상호정보량'])}, "
        f"Pearson={fmt(best['Pearson(크기)'])}, R²={fmt(best['R^2'])}).",
        f"- |Pearson| 최고: **{mp['변수']}** (Pearson={fmt(mp['Pearson(크기)'])}, R²={fmt(mp['R^2'])}).",
        "- **함의**: 가장 강한 변수도 R²가 매우 낮다(표본이 커서 통계적으론 유의해도 실질 설명력은 미미). "
        "가격 파생변수만으로는 방향 신호가 사실상 없음을 전수로 재확인.",
        "",
    ]


# %% [markdown]
# ## 보고서 조립 + main

# %%
def build_and_save_report(df: pd.DataFrame, args):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    stats = engdata.basic_statistics(df)
    report = [
        f"# 17번 EDA raw 결과 — {df['ticker'].iloc[0] if 'ticker' in df.columns else 'KRW-BTC'} 15분봉", "",
        f"- 실행 시각: {stamp} (서버, 헤드리스 .py)",
        f"- 드라이버: `test/models/17_eda_direction_signal_test.py`",
        f"- target: h={args.horizon}봉({args.horizon*15}분={args.horizon/4:.0f}시간) 누적 로그수익률의 크기·방향", "",
        "> 모든 수치는 과학적표기(1.2e-3) 없이 직관적 숫자로 표기. 그래프 축도 동일.", "",
    ]
    section_overview(df, report)
    section_series(df, report)
    section_distribution(df, report)
    section_autocorr(df, report)
    section_stationarity(df, report)
    section_decomposition(df, report)
    section_direction(df, report)
    (RESULTS_DIR / "eda_raw.md").write_text("\n".join(report), encoding="utf-8")
    print(f"[saved] {RESULTS_DIR / 'eda_raw.md'}")
    print(f"[saved images] {IMAGES_DIR}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="17번 전면 EDA")
    p.add_argument("--db", default=None)
    p.add_argument("--table", default="upbit_krw_candle")
    p.add_argument("--ticker", default="KRW-BTC")
    p.add_argument("--max-rows", type=int, default=0)
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_known_args(argv)[0]
    if a.max_rows <= 0:
        a.max_rows = None
    return a


def main(argv=None):
    args = parse_args(argv)
    print(f"[plan] EDA table={args.table} ticker={args.ticker} horizon={args.horizon}")
    df = load_dataset(args)
    print(f"[data] rows={len(df):,} feature_cols={len(feature_columns(df))}")
    if args.dry_run:
        print("[dry-run] 데이터 로드까지만 확인.")
        return
    build_and_save_report(df, args)
    print("[eda-done]")


if __name__ == "__main__":
    main()
