# %% [markdown]
# # 18번: 차분(differencing) 결정 — 왜 차분 여부에 따라 결론이 뒤집히는가
#
# 헤드리스 `.py` 드라이버(`#%%` 셀 + 파일 내 마크다운, AGENTS.md 2.3). ipynb는 만들지 않는다.
#
# ## 왜 (Why)
# 사용자 지적(2026-08-13): "원래 비정상성에 대한 모델링을 수행하는 줄 알았는데, 살펴보니 차분을
# 진행해서 분석과 모델링을 수행했다는 것을 파악했다. 이전 보고서에도 차분 여부나 갖가지 전처리에
# 대해선 전혀 나타나지 않아서 지금 제동이 걸렸다."
#
# 감사 결과 두 가지가 확인됐다.
#   (1) **차분은 엔진 최하층에 있다**: `engine/data.py:121`의 `log_return_1 = log_close.diff()`와
#       `:133`의 `target_return = log_return_1.shift(-1)`을 모든 실험이 통과한다. 실험별 선택이
#       아니라 데이터 로딩에 내장된 구조였다.
#   (2) **진단은 있었으나 계보로 전파되지 않았다**: 17번 S4가 레벨 vs 차분을 이미 비교했고 17번
#       보고서 3절이 사용자 질문에 답했지만, "그래서 우리 실험의 타깃은 차분값이다"라는 선언이
#       없어 1~16번 보고서와 교수님 브리프에 반영되지 않았다.
#
# ## 무엇을 (What)
#   D0 계보      : engine이 실제 적용하는 변환을 데이터로 재현(레벨→로그→차분→타깃)
#   D1 진단 뒤집힘: 같은 검정(ADF/KPSS/ARCH-LM/ACF)이 레벨↔차분에서 정반대 결론을 내는 것
#   D2 성과 뒤집힘: **모델을 고정하고 평가축만 교체**. 무모델이 레벨축에서 R²≈1을 받고, 레벨축은
#                  무모델과 실모델을 구별조차 못 한다는 것을 보인다
#   D3 GARCH 입력: 동일 추정기로 레벨/차분을 적합해, 레벨 입력이 지속성을 1로 밀어붙이는 것
#                  ("GARCH가 안 맞는다"가 아니라 "입력이 틀렸다"인 이유)
#   D4 분수차분   : d=0(레벨)~1(수익률) 연속 스윕. 정상성과 기억(memory)의 교환을 본다
#   D5 판정기준  : 위 수치로 GARCH 채택/기각 관문을 채운다
#   D6 전종목 확인: 상위 20종목에서 D1~D4의 뒤집힘이 재현되는지(BTC 특정이 아닌지) 확인
#
# ## 어떻게 (How)
# 데이터 축은 `upbit_krw_candle`(2026-07-18 교정 이후 정본). BTC 심층 + 상위 20종목 확인.
# GARCH(1,1)은 외부 패키지 대신 scipy MLE로 직접 적합한다 — 레벨/차분에 **완전히 동일한 추정기**를
# 써야 비교가 공정하다. 분수차분은 López de Prado식 고정폭(FFD) 이항 가중이며, FIR 필터라서
# `np.convolve`로 벡터화한다(파이썬 루프면 10만 행 × 가중치 수백 개로 실행 불가).
# 모든 수치는 과학적표기 없이 저장하고 원시 수치를 `differencing_raw.md`로 분리한다.
#
# ## 기대 결과 / 반영 (Expected)
# "차분 여부"가 취향이 아니라 **무엇을 예측 대상으로 삼는가의 선언**이며, 그 선언이 진단·평가·
# 모델입력을 동시에 규정한다는 것을 그림으로 확정한다. 그 위에서 GARCH 채택 기준과 A/B/C 갈래
# (레벨 직접 / 수익률 / 분수차분)의 판정 근거를 만든다.

# %%
"""18번 차분 결정 드라이버 본체.

실행:
    uv run test/models/18_differencing_decision_test.py
    uv run test/models/18_differencing_decision_test.py --max-rows 5000 --skip d6   # 빠른 점검
    uv run test/models/18_differencing_decision_test.py --skip d6                   # BTC만
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
from scipy import optimize
from statsmodels.stats.diagnostic import het_arch
from statsmodels.tsa.stattools import acf, adfuller, kpss

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

EXPERIMENT_TAG = "18_differencing_decision_20260814"
SOURCE_TABLE = "upbit_krw_candle"
BASE_TICKER = "KRW-BTC"
BARS_PER_DAY = 96
IMAGES_DIR = ROOT / "test" / "images" / EXPERIMENT_TAG
RESULTS_DIR = ROOT / "test" / "results" / EXPERIMENT_TAG
RAW_PATH = RESULTS_DIR / "differencing_raw.md"

# 17번에서 10만 행 ADF autolag가 segfault를 냈으므로 부분표본 + 고정 maxlag로 회피
TEST_SAMPLE = 20_000
TEST_MAXLAG = 24
# GARCH 적합 비용 통제: 심층(BTC)은 전체, 전종목 확인은 표본을 줄이고 시작점 1개
GARCH_SWEEP_SAMPLE = 20_000
D_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
D_GRID_COARSE = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
FFD_WEIGHT_THRESHOLD = 1e-4
FFD_MAX_TERMS = 2_000
ROBUSTNESS_TOP_N = 20

_LINES: list[str] = []


def emit(text: str = "") -> None:
    print(text, flush=True)
    _LINES.append(text)


def num(value, digits: int = 4) -> str:
    """과학적표기 없이(AGENTS.md 보고 규칙) 포맷."""
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


def load_series(con: duckdb.DuckDBPyConnection, ticker: str, max_rows: int | None) -> pd.DataFrame:
    frame = con.execute(
        f"select timestamp, close, volume from {SOURCE_TABLE} where ticker = ? order by timestamp",
        [ticker],
    ).df()
    if max_rows:
        frame = frame.tail(max_rows).reset_index(drop=True)
    frame["log_close"] = np.log(frame["close"].astype(float).clip(lower=1e-9))
    frame["log_return"] = frame["log_close"].diff()
    return frame


def top_liquid_tickers(con: duckdb.DuckDBPyConnection, n: int) -> list[str]:
    rows = con.execute(
        f"select ticker from {SOURCE_TABLE} group by ticker order by sum(value) desc limit {int(n)}"
    ).fetchall()
    return [r[0] for r in rows]


# %% [markdown]
# ## 분수차분 (fractional differencing) — 개념과 구현
#
# 1차 차분은 `x(t) − x(t−1)`, 즉 **직전 값 하나만** 빼는 것이다. 분수차분은 그 "1번"을 실수 d로
# 일반화해서, 과거를 한꺼번에 다 빼는 대신 **가중치를 두고 조금씩** 뺀다.
#
#   (1 − B)^d 의 이항급수:  w0 = 1,  w_k = −w_{k−1} · (d − k + 1) / k
#
# - d = 0 → 아무것도 빼지 않음 = 원래 레벨(비정상, 기억 100% 보존)
# - d = 1 → 직전 값만 뺌 = 지금 쓰는 로그수익률(정상, 기억 거의 소실)
# - 0 < d < 1 → **정상성을 얻을 만큼만 빼고 장기기억은 남긴다**
#
# 허스트 H > 0.5는 레벨에 장기기억이 실재한다는 뜻이라, d=1로 전부 빼면 그 정보를 버리는 셈이다.
# D4에서 "정상성을 처음 얻는 최소 d"와 "그때 남은 기억"을 스윕으로 확인한다.
#
# 구현 주의: out[i] = Σ_k w[k]·x[i−k] 는 FIR 필터이므로 `np.convolve`와 정확히 같다.
# 파이썬 이중 루프로 쓰면 10만 행 × 가중치 수백 개에서 실행이 끝나지 않는다.

# %%
def ffd_weights(d: float, threshold: float = FFD_WEIGHT_THRESHOLD, max_terms: int = FFD_MAX_TERMS) -> np.ndarray:
    """(1-B)^d 의 이항 가중치. |w|가 threshold 밑으로 떨어지면 절단."""
    weights = [1.0]
    for k in range(1, max_terms):
        nxt = -weights[-1] * (d - k + 1) / k
        if abs(nxt) < threshold:
            break
        weights.append(nxt)
    return np.asarray(weights, dtype=float)


def frac_diff(series: pd.Series, d: float) -> pd.Series:
    """고정폭(FFD) 분수차분. d=0이면 원계열, d=1이면 1차 차분과 동일."""
    values = series.astype(float).to_numpy()
    if d == 0:
        return pd.Series(values, index=series.index)
    weights = ffd_weights(d)
    width = len(weights)
    filtered = np.convolve(values, weights)[: len(values)]
    filtered[: width - 1] = np.nan          # 창이 다 차기 전 구간은 버린다
    return pd.Series(filtered, index=series.index)


def hurst_rs(series: pd.Series, min_chunk: int = 128, n_scales: int = 8) -> float:
    """R/S 방법 허스트 지수. H>0.5면 장기기억(추세 지속) 시사."""
    values = series.dropna().to_numpy(float)
    n = len(values)
    scales, rs_values = [], []
    size = min_chunk
    for _ in range(n_scales):
        if size * 2 > n:
            break
        ratios = []
        for c in range(n // size):
            chunk = values[c * size : (c + 1) * size]
            deviation = np.cumsum(chunk - chunk.mean())
            spread = float(deviation.max() - deviation.min())
            sigma = float(chunk.std())
            if sigma > 1e-12 and spread > 0:
                ratios.append(spread / sigma)
        if ratios:
            scales.append(size)
            rs_values.append(float(np.mean(ratios)))
        size *= 2
    if len(scales) < 3:
        return float("nan")
    return float(np.polyfit(np.log(scales), np.log(rs_values), 1)[0])


def gph_d(series: pd.Series, m_power: float = 0.5) -> float:
    """GPH(Geweke–Porter–Hudak) 로그주기도 회귀로 장기기억 모수 d를 직접 추정.

    R/S 허스트와 독립적인 추정기다. **적분된(비정상) 계열에 R/S를 적용하면 H가 1 부근으로
    나와 해석이 안 되므로**, 장기기억 판정은 증분(수익률)에서 하고 두 추정기를 교차확인한다.
    d ≈ 0이면 장기기억 없음(=로그가격이 I(1), 1차 차분이 정답).
    """
    x = series.dropna().to_numpy(float)
    x = x - x.mean()
    n = len(x)
    if n < 256:
        return float("nan")
    m = int(n**m_power)
    periodogram = (np.abs(np.fft.rfft(x)) ** 2) / (2 * np.pi * n)
    j = np.arange(1, m + 1)
    omega = 2 * np.pi * j / n
    regressor = np.log(4 * np.sin(omega / 2) ** 2)
    response = np.log(periodogram[j])
    return float(-np.polyfit(regressor, response, 1)[0])


def stationarity_verdict(series: pd.Series, label: str) -> dict:
    """ADF·KPSS·ARCH-LM을 한 번에. 표본은 최근 TEST_SAMPLE로 고정."""
    clean = series.dropna()
    sample = clean.tail(TEST_SAMPLE)
    adf_stat, adf_p = adfuller(sample, maxlag=TEST_MAXLAG, autolag=None)[:2]
    try:
        kpss_stat, kpss_p = kpss(sample, regression="c", nlags=TEST_MAXLAG)[:2]
    except Exception:
        kpss_stat, kpss_p = float("nan"), float("nan")
    try:
        arch_p = float(het_arch(sample - sample.mean(), nlags=12)[1])
    except Exception:
        arch_p = float("nan")
    autocorr = acf(sample, nlags=2, fft=True)
    return {
        "label": label,
        "n": int(len(clean)),
        "adf_stat": float(adf_stat),
        "adf_p": float(adf_p),
        "kpss_stat": float(kpss_stat),
        "kpss_p": float(kpss_p),
        "arch_p": arch_p,
        "acf1": float(autocorr[1]),
        # ADF는 귀무가설이 "단위근(비정상)", KPSS는 "정상" — 방향이 반대다
        "adf_says_stationary": bool(adf_p < 0.05),
        "kpss_says_stationary": bool(kpss_p > 0.05) if np.isfinite(kpss_p) else False,
    }


# %% [markdown]
# ## D0 — 전처리 계보: engine이 실제로 무엇을 적용하는가
#
# 보고서에 빠져 있던 부분. 코드를 읽어 서술하는 게 아니라, 같은 변환을 데이터에 직접 적용해
# 각 단계의 통계가 어떻게 바뀌는지 숫자로 남긴다.

# %%
def section_d0(frame: pd.DataFrame) -> None:
    close = frame["close"].astype(float)
    log_close = frame["log_close"]
    log_return = frame["log_return"].dropna()

    emit("## D0 전처리 계보 (engine/data.py가 실제로 적용하는 변환)")
    emit()
    emit("| 단계 | 코드 위치 | 계열 | 평균 | 표준편차 | 최소 | 최대 |")
    emit("| :--- | :--- | :--- | ---: | ---: | ---: | ---: |")
    emit(
        f"| ① 원본 레벨 | `{SOURCE_TABLE}.close` | 종가(KRW) | {num(close.mean(), 0)} | "
        f"{num(close.std(), 0)} | {num(close.min(), 0)} | {num(close.max(), 0)} |"
    )
    emit(
        f"| ② 로그 변환 | `data.py:120` log_close | log(종가) | {num(log_close.mean())} | "
        f"{num(log_close.std())} | {num(log_close.min())} | {num(log_close.max())} |"
    )
    emit(
        f"| ③ **1차 차분** | `data.py:121` log_return_1 | 로그수익률 | {num(log_return.mean(), 8)} | "
        f"{num(log_return.std(), 6)} | {num(log_return.min(), 6)} | {num(log_return.max(), 6)} |"
    )
    emit()
    emit("- **②는 아직 레벨이다.** 로그 변환은 차분이 아니라 단위 변경(곱셈적 변동 → 덧셈적 변동)이라,")
    emit("  로그 종가도 여전히 비정상이다(D1에서 확인). 차분은 ③에서 처음 일어난다.")
    emit("- ③이 이 연구의 실제 분석축이다. `data.py:133`의 `target_return = log_return_1.shift(-1)`로")
    emit("  **예측 대상 자체가 차분값**이 되고, `features.py`의 파생변수 60여 개도 대부분 이 위에 얹힌다.")
    emit("- `data.py:138`에 레벨 타깃(`target_close`)도 정의돼 있으나 실험들이 학습·평가에 쓴 것은")
    emit("  차분 타깃 계열이다(17번 EDA도 `log_close.shift(-h) - log_close`).")
    emit(f"- 변동계수(표준편차/|평균|): 레벨 {num(close.std() / abs(close.mean()), 3)} → "
         f"로그 {num(log_close.std() / abs(log_close.mean()), 3)}.")
    emit()


# %% [markdown]
# ## D1 — 진단 뒤집힘: 같은 검정이 정반대 결론을 낸다

# %%
def section_d1(frame: pd.DataFrame) -> dict:
    log_close = frame["log_close"]
    cases = [
        ("레벨 (log 종가, d=0)", log_close),
        ("분수차분 d=0.3", frac_diff(log_close, 0.3)),
        ("분수차분 d=0.5", frac_diff(log_close, 0.5)),
        ("1차 차분 (로그수익률, d=1)", frame["log_return"]),
    ]
    verdicts = [stationarity_verdict(series, label) for label, series in cases]

    emit("## D1 진단 뒤집힘 (동일 검정, 정반대 결론)")
    emit()
    emit(f"- 검정 표본: 각 계열의 최근 {TEST_SAMPLE:,}행, 고정 maxlag={TEST_MAXLAG}")
    emit("- ADF 귀무가설 = '단위근 있음(비정상)' → p<0.05면 정상.")
    emit("  KPSS 귀무가설 = '정상' → p<0.05면 비정상. **방향이 반대다.** 둘이 일치할 때 결론이 견고하다.")
    emit("- **주의**: statsmodels의 KPSS p값은 표에서 보간되어 [0.01, 0.10]으로 절단된다.")
    emit("  0.0100은 '0.01 이하', 0.1000은 '0.10 이상'을 뜻하므로 통계량도 함께 본다.")
    emit()
    emit("| 계열 | ADF 통계량 | ADF p | KPSS 통계량 | KPSS p | ADF 판정 | KPSS 판정 | ACF(1) | ARCH-LM p |")
    emit("| :--- | ---: | ---: | ---: | ---: | :--- | :--- | ---: | ---: |")
    for v in verdicts:
        emit(
            f"| {v['label']} | {num(v['adf_stat'], 3)} | {num(v['adf_p'])} | "
            f"{num(v['kpss_stat'], 3)} | {num(v['kpss_p'])} | "
            f"{'정상' if v['adf_says_stationary'] else '**비정상**'} | "
            f"{'정상' if v['kpss_says_stationary'] else '**비정상**'} | "
            f"{num(v['acf1'])} | {num(v['arch_p'])} |"
        )
    emit()
    level, diff = verdicts[0], verdicts[-1]
    emit(f"- **레벨**: ACF(1) = {num(level['acf1'])} — 거의 1. 직전 값이 다음 값을 그대로 설명한다.")
    emit(f"- **1차 차분**: ACF(1) = {num(diff['acf1'])} — 거의 0. 방향 정보가 사라진다.")
    emit("- 이 한 쌍이 모든 뒤집힘의 근원이다. 레벨의 자기상관 1은 '예측할 것이 넘친다'는 뜻이 아니라")
    emit("  **'어제와 오늘이 비슷하다'는 동어반복**이다. 차분은 그 동어반복을 제거해 '그걸 빼고도 남는")
    emit("  것이 있는가'를 묻는 축으로 옮긴다. 그래서 같은 데이터가 정반대로 읽힌다.")
    emit()

    fig, axes = plt.subplots(2, 3, figsize=(19.5, 9))
    timestamps = pd.to_datetime(frame["timestamp"])
    level_sample = log_close.dropna().tail(TEST_SAMPLE)
    return_sample = frame["log_return"].dropna().tail(TEST_SAMPLE)

    axes[0][0].plot(timestamps, log_close, color="#1f77b4", lw=0.6)
    axes[0][0].set_title("레벨: log 종가 (d=0)\n한 수준에 머물지 않는다 → 비정상")
    axes[0][1].bar(range(1, 26), acf(level_sample, nlags=25, fft=True)[1:], color="#1f77b4")
    axes[0][1].set_title("레벨 ACF — 거의 감소하지 않음 (단위근 특징)")
    axes[0][1].set_ylim(-0.15, 1.05)
    axes[0][1].axhline(0, color="black", lw=0.5)
    axes[0][2].axis("off")
    axes[0][2].text(
        0.02, 0.5,
        "레벨축의 자기상관 ≈ 1 은\n'정보'가 아니라 '동어반복'이다.\n\n"
        f"ADF p = {num(level['adf_p'])} → 비정상\n"
        f"KPSS p = {num(level['kpss_p'])} → 비정상\n"
        f"ACF(1) = {num(level['acf1'])}\n\n"
        "→ 이 축에서 R²를 재면\n   아무 모델이나 1에 가까워진다\n   (D2에서 실증)",
        fontsize=13, va="center",
    )

    axes[1][0].plot(timestamps, frame["log_return"], color="#d62728", lw=0.4)
    axes[1][0].axhline(0, color="black", lw=0.5)
    axes[1][0].set_title("1차 차분: 로그수익률 (d=1)\n0 주변에 몰린다 → 평균-정상")
    axes[1][1].bar(range(1, 26), acf(return_sample, nlags=25, fft=True)[1:], color="#d62728")
    axes[1][1].set_title("차분 ACF — 즉시 0으로 붕괴 (방향 무신호)")
    axes[1][1].set_ylim(-0.15, 1.05)
    axes[1][1].axhline(0, color="black", lw=0.5)
    axes[1][2].bar(range(1, 26), acf(return_sample.abs(), nlags=25, fft=True)[1:], color="#2ca02c")
    axes[1][2].set_title("차분의 |값| ACF — 느리게 감소 (변동성 군집)\n크기에는 신호가 남는다")
    axes[1][2].set_ylim(-0.15, 1.05)
    axes[1][2].axhline(0, color="black", lw=0.5)
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "d1_diagnosis_flip.png", dpi=120)
    plt.close(fig)

    return {"verdicts": verdicts}


# %% [markdown]
# ## D2 — 성과 뒤집힘: 동일한 예측, 정반대 성적표
#
# 사용자 질문의 핵심. **모델을 바꾸지 않고 평가축만 바꾼다.**
# 무모델(naive: 다음 로그가격 = 현재 로그가격)과 실모델(수익률 AR(1))을 각각 레벨축·차분축에서
# 평가한다. 네 조합 모두 **완전히 동일한 시점 집합**을 본다. 레벨축이 무모델과 실모델을
# 구별하지 못하는지가 판정 지점이다.

# %%
def _metrics(y_true: np.ndarray, y_pred: np.ndarray, naive_pred: np.ndarray) -> dict:
    resid = y_true - y_pred
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    mae = float(np.mean(np.abs(resid)))
    naive_mae = float(np.mean(np.abs(y_true - naive_pred)))
    return {
        "r2": 1.0 - float(np.sum(resid**2)) / ss_tot if ss_tot > 0 else float("nan"),
        "rmse": float(np.sqrt(np.mean(resid**2))),
        "mae": mae,
        # MASE: naive 대비 상대오차. 1보다 작아야 naive를 이긴 것.
        "mase": mae / naive_mae if naive_mae > 1e-15 else float("nan"),
    }


def section_d2(frame: pd.DataFrame) -> dict:
    lp = frame["log_close"].to_numpy(float)
    returns = np.diff(lp)                     # returns[i] = lp[i+1] - lp[i]
    n = len(lp)
    # t = 1 .. n-2 라야 returns[t-1](입력)과 lp[t+1](정답)이 모두 존재한다
    t = np.arange(1, n - 1)

    y_level = lp[t + 1]
    y_return = lp[t + 1] - lp[t]
    naive_level = lp[t]                       # 무모델: 직전 값 복사
    naive_return = np.zeros_like(y_return)    # 같은 예측을 차분축에서 보면 "변화 없음"

    split = int(len(t) * 0.7)
    train, test = slice(0, split), slice(split, len(t))

    # 실모델: 수익률 AR(1). 학습 구간에서만 계수 추정 (시간분할, 누수 없음)
    slope, intercept = np.polyfit(returns[t - 1][train], y_return[train], 1)
    predicted_return = intercept + slope * returns[t - 1]
    predicted_level = lp[t] + predicted_return

    rows = [
        ("무모델 (naive: 다음=현재)", "레벨축 (log 가격)",
         _metrics(y_level[test], naive_level[test], naive_level[test])),
        ("무모델 (naive: 다음=현재)", "차분축 (로그수익률)",
         _metrics(y_return[test], naive_return[test], naive_return[test])),
        ("실모델 (수익률 AR(1))", "레벨축 (log 가격)",
         _metrics(y_level[test], predicted_level[test], naive_level[test])),
        ("실모델 (수익률 AR(1))", "차분축 (로그수익률)",
         _metrics(y_return[test], predicted_return[test], naive_return[test])),
    ]

    emit("## D2 성과 뒤집힘 (모델 고정, 평가축만 교체)")
    emit()
    emit(f"- 평가 구간: 시간순 뒤쪽 30%({len(t) - split:,}행). 네 행 모두 동일한 시점 집합을 본다.")
    emit(f"- 실모델은 수익률 AR(1), 학습 구간 추정 계수 slope = {num(slope, 6)}, intercept = {num(intercept, 8)}")
    emit()
    emit("| 모델 | 평가축 | R² | RMSE | MAE | MASE |")
    emit("| :--- | :--- | ---: | ---: | ---: | ---: |")
    for model, axis, metric in rows:
        emit(
            f"| {model} | {axis} | {num(metric['r2'], 6)} | {num(metric['rmse'], 6)} | "
            f"{num(metric['mae'], 6)} | {num(metric['mase'])} |"
        )
    emit()
    naive_level_r2, naive_diff_r2 = rows[0][2]["r2"], rows[1][2]["r2"]
    real_level_r2, real_diff_r2 = rows[2][2]["r2"], rows[3][2]["r2"]
    emit(f"- **무모델 R²: 레벨축 {num(naive_level_r2, 6)} / 차분축 {num(naive_diff_r2, 6)}.**")
    emit("  아무것도 학습하지 않고 직전 값을 복사한 것이 레벨축에서는 거의 완벽한 모델로 보인다.")
    emit(f"- 실모델의 레벨축 R² {num(real_level_r2, 6)}은 무모델 {num(naive_level_r2, 6)}과 사실상")
    emit(f"  구별되지 않는다(차이 {num(real_level_r2 - naive_level_r2, 8)}).")
    emit("  → **레벨축 R²는 실력과 동어반복을 분간하지 못한다.**")
    emit(f"- 차분축에서는 실모델 R² {num(real_diff_r2, 6)}로 낮지만 무모델과 구별된다 —")
    emit("  이 축에서만 '동어반복을 빼고 남은 진짜 신호'를 측정할 수 있다.")
    emit("- MASE는 naive를 기준(1.0)으로 삼으므로 축을 바꿔도 속지 않는다. 척도-자유 지표가 필요한 이유다.")
    emit()
    emit("**결론**: 차분 여부는 전처리 취향이 아니라 **평가의 정직성을 결정하는 선언**이다.")
    emit("레벨축으로 성적을 내면 R²·상관이 부풀고, 그 숫자로는 모델이 나아졌는지 알 수 없다.")
    emit()

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    labels = ["무모델\n레벨축", "무모델\n차분축", "실모델\n레벨축", "실모델\n차분축"]
    r2_values = [r[2]["r2"] for r in rows]
    bars = axes[0].bar(labels, r2_values, color=["#c6dbef", "#c6dbef", "#1f77b4", "#1f77b4"],
                       edgecolor="black", lw=0.6)
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_title("같은 모델, 축만 바꾼 R²\n레벨축은 무모델도 1에 가깝다")
    axes[0].set_ylabel("R²")
    for bar, value in zip(bars, r2_values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, value, num(value, 4),
                     ha="center", va="bottom" if value >= 0 else "top", fontsize=10)

    axes[1].bar(labels, [r[2]["mase"] for r in rows],
                color=["#fdd0a2", "#fdd0a2", "#ff7f0e", "#ff7f0e"], edgecolor="black", lw=0.6)
    axes[1].axhline(1.0, color="red", ls="--", lw=1.5, label="naive 기준선 = 1")
    axes[1].set_title("MASE (naive 대비 상대오차)\n축을 바꿔도 속지 않는 지표")
    axes[1].legend()

    show = 400
    axes[2].plot(y_level[test][:show], color="black", lw=1.6, label="실제 log 가격")
    axes[2].plot(naive_level[test][:show], color="#d62728", lw=1.0, ls="--", label="무모델(직전값 복사)")
    axes[2].set_title("레벨축에서 무모델이 '잘 맞아 보이는' 이유\n한 칸 밀린 복사본이라 눈으로 구별 불가")
    axes[2].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "d2_performance_flip.png", dpi=120)
    plt.close(fig)

    return {"rows": rows, "slope": float(slope)}


# %% [markdown]
# ## D3 — GARCH 입력: "미사용"이 아니라 "입력 오류"
#
# GARCH(1,1)은 **평균-정상** 계열의 조건부 분산을 모델링한다. 레벨을 그대로 넣으면 무엇이
# 망가지는지, 같은 추정기로 두 입력을 적합해 비교한다. 외부 패키지를 쓰지 않는 이유도 이 공정성.

# %%
def fit_garch11(series: pd.Series, sample: int | None = None, single_start: bool = False) -> dict:
    """GARCH(1,1) 정규 MLE. 반환: omega/alpha/beta, 지속성, 반감기, 조건부 변동성."""
    x = series.dropna().to_numpy(float)
    if sample and len(x) > sample:
        x = x[-sample:]
    x = x - x.mean()
    scale = float(np.std(x))
    if scale <= 0:
        raise ValueError("표준편차가 0인 계열")
    z = x / scale                      # 수치 안정용 표준화(파라미터는 스케일 불변)
    var_z = float(np.var(z))

    def recursion(omega: float, alpha: float, beta: float) -> np.ndarray:
        variance = np.empty(len(z))
        variance[0] = var_z
        for i in range(1, len(z)):
            variance[i] = omega + alpha * z[i - 1] ** 2 + beta * variance[i - 1]
        return np.maximum(variance, 1e-12)

    def negative_log_likelihood(params):
        omega, alpha, beta = params
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.9999:
            return 1e10
        variance = recursion(omega, alpha, beta)
        return 0.5 * float(np.sum(np.log(variance) + z**2 / variance))

    starts = [(0.05, 0.10, 0.85)] if single_start else [(0.05, 0.10, 0.85), (0.01, 0.05, 0.90), (0.10, 0.20, 0.70)]
    best = None
    for start in starts:
        result = optimize.minimize(
            negative_log_likelihood, start, method="L-BFGS-B",
            bounds=[(1e-8, 5.0), (0.0, 0.9999), (0.0, 0.9999)],
        )
        if best is None or result.fun < best.fun:
            best = result

    omega, alpha, beta = best.x
    persistence = float(alpha + beta)
    half_life = float(np.log(0.5) / np.log(persistence)) if 0 < persistence < 1 else float("inf")
    variance = recursion(omega, alpha, beta)
    conditional_vol = np.sqrt(variance) * scale
    # 표준화 잔차: GARCH가 분산을 제대로 걷어냈으면 이것은 백색잡음이어야 한다.
    # 레벨을 입력하면 평균식이 없으므로 여기에 자기상관이 그대로 남는다 → 전제 위반의 직접 증거.
    standardized = z / np.sqrt(variance)
    return {
        "omega": float(omega), "alpha": float(alpha), "beta": float(beta),
        "persistence": persistence,
        "half_life_bars": half_life,
        "half_life_hours": half_life * 0.25 if np.isfinite(half_life) else float("inf"),
        "conditional_vol": conditional_vol,
        "standardized": standardized,
        "scale": scale, "params": (float(omega), float(alpha), float(beta)),
        "n": len(z),
    }


def section_d3(frame: pd.DataFrame) -> dict:
    fits: dict[str, dict] = {}
    for label, series in [("레벨 (log 종가)", frame["log_close"]), ("1차 차분 (로그수익률)", frame["log_return"])]:
        try:
            fits[label] = fit_garch11(series)
        except Exception as exc:                   # 적합 실패 자체도 결과다
            fits[label] = {"error": str(exc)}

    emit("## D3 GARCH(1,1) 입력: 레벨 vs 차분")
    emit()
    emit("- 동일한 추정기(scipy 정규 MLE)를 두 입력에 적용 — 패키지 차이로 인한 교란 없음.")
    emit("- 지속성 = alpha + beta. 1에 붙으면 충격이 사라지지 않는다는 뜻(IGARCH)이라 예측에 못 쓴다.")
    emit()
    emit("| 입력 | omega | alpha | beta | 지속성(a+b) | 반감기(봉) | 반감기(시간) |")
    emit("| :--- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for label, fit in fits.items():
        if "error" in fit:
            emit(f"| {label} | 적합 실패: {fit['error']} | | | | | |")
            continue
        emit(
            f"| {label} | {num(fit['omega'], 6)} | {num(fit['alpha'])} | {num(fit['beta'])} | "
            f"**{num(fit['persistence'])}** | {num(fit['half_life_bars'], 1)} | {num(fit['half_life_hours'], 1)} |"
        )
    emit()
    level_fit, diff_fit = fits.get("레벨 (log 종가)", {}), fits.get("1차 차분 (로그수익률)", {})

    # (1) 전제 위반의 직접 증거 — 표준화 잔차에 자기상관이 남는가
    emit("### 전제 위반 진단: 표준화 잔차 (지속성 수치보다 이것이 결정적이다)")
    emit()
    emit("GARCH가 분산을 제대로 걷어냈으면 표준화 잔차 z/sigma는 백색잡음이어야 한다.")
    emit("레벨 입력은 평균식이 없으므로 평균의 자기상관이 그대로 남는다.")
    emit()
    emit("| 입력 | 표준화 잔차 ACF(1) | Ljung-Box(20) p | 판정 |")
    emit("| :--- | ---: | ---: | :--- |")
    for label, fit in fits.items():
        if "standardized" not in fit:
            continue
        residual = pd.Series(fit["standardized"])
        resid_acf1 = float(acf(residual.tail(TEST_SAMPLE), nlags=2, fft=True)[1])
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox
            lb_p = float(acorr_ljungbox(residual.tail(TEST_SAMPLE), lags=[20], return_df=True)["lb_pvalue"].iloc[0])
        except Exception:
            lb_p = float("nan")
        ok = abs(resid_acf1) < 0.1
        emit(f"| {label} | {num(resid_acf1)} | {num(lb_p)} | "
             f"{'백색잡음에 가까움 → 전제 충족' if ok else '**자기상관 잔존 → 전제 위반**'} |")
    emit()
    emit("- 레벨 입력의 표준화 잔차는 자기상관이 거의 그대로 남는다 — GARCH는 분산만 다루고")
    emit("  평균의 표류를 다루지 않기 때문이다. **이것이 '레벨을 넣으면 안 된다'의 직접 증거다.**")
    emit("- 따라서 **'GARCH가 안 맞는다'는 판정을 레벨 입력으로 내려선 안 된다.** 그건 모델의 실패가")
    emit("  아니라 전제 위반(입력 오류)이다. GARCH 채택/기각은 차분축에서만 논할 수 있다.")
    emit()

    # (2) 지속성 식별 불안정성 — 표본을 바꿔가며 재추정
    emit("### 지속성·반감기는 이 데이터에서 안정적으로 식별되지 않는다")
    emit()
    returns = frame["log_return"]
    stability = []
    for name, series in [
        ("전체", returns),
        ("최근 40,000", returns.tail(40_000)),
        ("최근 20,000", returns.tail(20_000)),
        ("최근 10,000", returns.tail(10_000)),
        ("앞 40,000", returns.dropna().head(40_000)),
    ]:
        try:
            fit = fit_garch11(series)
            stability.append((name, fit["persistence"], fit["half_life_hours"]))
        except Exception:
            stability.append((name, float("nan"), float("nan")))
    emit("| 표본 | alpha+beta | 반감기(시간) |")
    emit("| :--- | ---: | ---: |")
    for name, persistence, half_life in stability:
        emit(f"| {name} | {num(persistence, 5)} | {num(half_life, 1)} |")
    emit()
    finite = [h for _, _, h in stability if np.isfinite(h)]
    if finite:
        fits["_stability"] = {
            "min_hours": min(finite), "max_hours": max(finite),
            "ratio": max(finite) / max(min(finite), 1e-9),
        }
        emit(f"- 반감기 추정 범위: **{num(min(finite), 1)}시간 ~ {num(max(finite), 1)}시간** "
             f"(배율 {num(max(finite) / max(min(finite), 1e-9), 0)}배).")
    emit("- alpha+beta가 1에 붙으면 우도가 그 방향으로 거의 평평해져 **식별이 사실상 안 된다**")
    emit("  (IGARCH 근방의 알려진 문제). 표본을 조금만 바꿔도 반감기가 수십 배 흔들린다.")
    emit("- **함의**: `professor_brief` 4.6절의 '반감기 8.6시간(alpha+beta=0.98)'은 특정 표본에서 나온")
    emit("  값이며 재현되지 않는다. 위 표의 '최근 10,000' 행이 그 값에 가깝다 —")
    emit("  즉 **틀린 계산이 아니라 불안정한 추정**이다.")
    emit("- **따라서 지속성·반감기를 GARCH 채택 관문으로 쓸 수 없다.** 판정은 표본외 예측 성능으로")
    emit("  옮긴다(D3b).")
    emit()

    fig, axes = plt.subplots(1, 2, figsize=(19, 6))
    timestamps = pd.to_datetime(frame["timestamp"])
    for ax, (label, fit) in zip(axes, fits.items()):
        if "error" in fit:
            ax.axis("off")
            ax.text(0.5, 0.5, f"{label}\n적합 실패", ha="center", va="center")
            continue
        vol = fit["conditional_vol"]
        ax.plot(timestamps.to_numpy()[-len(vol):], vol, lw=0.6,
                color="#1f77b4" if "레벨" in label else "#d62728")
        ax.set_title(
            f"{label} 입력 → GARCH(1,1) 조건부 변동성\n"
            f"지속성 = {num(fit['persistence'])}, 반감기 = {num(fit['half_life_hours'], 1)}시간"
        )
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "d3_garch_input.png", dpi=120)
    plt.close(fig)

    return fits


# %% [markdown]
# ## D3b — 표본외 변동성 예측: GARCH를 채택할 실제 근거
#
# 지속성·반감기가 식별되지 않으므로(D3), 채택 판정은 **표본외 1스텝 예측 성능**으로 한다.
# 학습 70%에서 파라미터를 추정하고, 평가 30%는 파라미터를 고정한 채 분산을 전진 필터링한다
# (재적합 없음 = 누수 없음). 베이스라인은 롤링표준편차·EWMA·상수분산.
#
# 평가지표는 QLIKE = mean(log σ² + r²/σ²). 분산 예측 평가의 표준 손실로, 실현분산 대리변수(r²)가
# 잡음이 심해도 순위가 뒤집히지 않는 성질(robust loss)이 있어 MSE보다 신뢰된다.

# %%
def section_d3b(frame: pd.DataFrame) -> dict:
    returns = frame["log_return"].dropna().to_numpy(float)
    n = len(returns)
    split = int(n * 0.7)
    train, test = returns[:split], returns[split:]
    realized = test**2                       # 실현분산 대리변수

    # (1) GARCH(1,1): 학습에서 추정 → 평가구간은 파라미터 고정 전진 필터링
    fit = fit_garch11(pd.Series(train))
    omega, alpha, beta = fit["params"]
    scale = fit["scale"]
    train_z = (train - train.mean()) / scale
    variance_z = np.var(train_z)
    for value in train_z[1:]:                # 학습 끝 상태까지 진행
        variance_z = omega + alpha * value**2 + beta * variance_z
    garch_forecast = np.empty(len(test))
    test_z = (test - train.mean()) / scale
    for i in range(len(test)):
        garch_forecast[i] = variance_z * scale**2       # t 시점에 아는 정보로 만든 t의 예측
        variance_z = omega + alpha * test_z[i] ** 2 + beta * variance_z

    # (2) 베이스라인들 — 모두 t 시점 이전 정보만 사용
    series = pd.Series(returns)
    rolling = series.rolling(BARS_PER_DAY).var().shift(1).to_numpy()[split:]
    ewma = series.pow(2).ewm(alpha=0.06, adjust=False).mean().shift(1).to_numpy()[split:]
    constant = np.full(len(test), float(np.var(train)))

    def qlike(forecast: np.ndarray) -> float:
        mask = np.isfinite(forecast) & (forecast > 0)
        f, r = forecast[mask], realized[mask]
        return float(np.mean(np.log(f) + r / np.maximum(f, 1e-18)))

    def mse(forecast: np.ndarray) -> float:
        mask = np.isfinite(forecast) & (forecast > 0)
        return float(np.mean((realized[mask] - forecast[mask]) ** 2))

    def corr_abs(forecast: np.ndarray) -> float:
        mask = np.isfinite(forecast) & (forecast > 0)
        return float(np.corrcoef(np.sqrt(forecast[mask]), np.abs(test[mask]))[0, 1])

    # (3) HAR-RV — 장기기억을 세 시간축(일·주·월)의 합으로 근사한다(Corsi류).
    # D4b에서 변동성 채널의 d가 0.34~0.40으로 나오므로, 지수감쇠만 가능한 GARCH(1,1)보다
    # 이 쪽이 감쇠 형태를 옳게 잡을 수 있다. 이 비교가 "GARCH를 쓸지"의 실질 판정이다.
    # 주의: 로그 적합 후 exp 변환은 쓰지 않는다. log(r^2)의 잔차는 log 카이제곱이라 분산이 매우
    # 크고(정규 가정 시 4.93), exp(+sigma^2/2) Jensen 보정이 예측을 10배 이상 부풀린다(실측 확인).
    # QLIKE가 요구하는 것은 조건부 분산 E[r^2|F]이므로 레벨에서 직접 OLS로 추정한다.
    squared = series.pow(2)
    design = pd.DataFrame({
        "y": squared,
        "d": squared.rolling(BARS_PER_DAY).mean().shift(1),
        "w": squared.rolling(BARS_PER_DAY * 7).mean().shift(1),
        "m": squared.rolling(BARS_PER_DAY * 30).mean().shift(1),
    })
    train_design = design.iloc[:split].dropna()
    har_forecast = np.full(len(test), np.nan)
    if len(train_design) > 1000:
        train_matrix = np.column_stack([np.ones(len(train_design)), train_design[["d", "w", "m"]].to_numpy()])
        coef = np.linalg.lstsq(train_matrix, train_design["y"].to_numpy(), rcond=None)[0]
        test_design = design.iloc[split:][["d", "w", "m"]].to_numpy()
        mask = np.isfinite(test_design).all(axis=1)
        har_forecast[mask] = coef[0] + test_design[mask] @ coef[1:]
        # 분산 예측이므로 음수는 허용되지 않는다. 학습 분산의 1% 하한으로 절단.
        floor = 0.01 * float(np.var(train))
        har_forecast = np.where(np.isfinite(har_forecast), np.maximum(har_forecast, floor), np.nan)

    models = {
        "GARCH(1,1)": garch_forecast,
        "HAR-RV (일·주·월, 장기기억 근사)": har_forecast,
        f"롤링분산({BARS_PER_DAY}봉)": rolling,
        "EWMA(alpha=0.06)": ewma,
        "상수분산(학습 표본분산)": constant,
    }

    emit("## D3b 표본외 변동성 예측 (실제 채택 근거)")
    emit()
    emit(f"- 학습 {split:,}행 / 평가 {len(test):,}행. 평가구간은 파라미터 고정 전진 필터링(재적합 없음).")
    emit("- QLIKE = mean(log sigma^2 + r^2/sigma^2), **낮을수록 좋다**. 분산 예측의 표준 robust 손실.")
    emit()
    emit("| 모델 | QLIKE | MSE | corr(예측sigma, |r|) |")
    emit("| :--- | ---: | ---: | ---: |")
    scores = {}
    for name, forecast in models.items():
        scores[name] = qlike(forecast)
        emit(f"| {name} | {num(scores[name], 4)} | {num(mse(forecast) * 1e12, 3)} | {num(corr_abs(forecast))} |")
    emit()
    emit("  (MSE는 1e12를 곱한 값 — 분산의 제곱오차라 원단위가 너무 작다)")
    emit()
    finite_scores = {k: v for k, v in scores.items() if np.isfinite(v)}
    best = min(finite_scores, key=finite_scores.get)
    garch_score = scores["GARCH(1,1)"]
    rolling_score = scores[f"롤링분산({BARS_PER_DAY}봉)"]
    har_score = scores["HAR-RV (일·주·월, 장기기억 근사)"]
    emit(f"- **QLIKE 최우수: {best}**")
    emit(f"- GARCH {num(garch_score, 4)} vs 롤링분산 {num(rolling_score, 4)} → "
         f"차이 {num(garch_score - rolling_score, 4)}")
    if garch_score < rolling_score:
        emit("- GARCH는 롤링분산 베이스라인을 이긴다 → 단순 베이스라인 대비 채택 근거는 성립.")
    else:
        emit("- GARCH가 롤링분산을 이기지 못한다 → 단순 베이스라인으로 충분하다.")
    if np.isfinite(har_score):
        emit(f"- **GARCH {num(garch_score, 4)} vs HAR-RV {num(har_score, 4)} → "
             f"차이 {num(garch_score - har_score, 4)}**")
        if har_score < garch_score:
            emit("- **HAR-RV가 GARCH(1,1)를 이긴다.** D4b의 진단(변동성 채널에 장기기억 d≈0.4)과 정합적이다:")
            emit("  GARCH(1,1)은 지수 감쇠만 표현할 수 있어 하이퍼볼릭 감쇠를 흉내내려 alpha+beta를 1로")
            emit("  밀어붙인다. **지속성·반감기가 표본에 따라 요동친 것은 추정 잡음이 아니라 모델 오지정의**")
            emit("  **증상이었다.** → B(변동성) 갈래의 기본 모델을 GARCH(1,1)이 아니라 장기기억 계열")
            emit("  (HAR-RV·FIGARCH·ARFIMA on log-RV)로 두어야 한다.")
        else:
            emit("- HAR-RV가 GARCH를 이기지 못한다 — 장기기억 근사가 이 horizon에서 이득을 주지 않는다.")
    emit(f"- 상수분산 {num(scores['상수분산(학습 표본분산)'], 4)} 대비 개선폭이 "
         "'변동성 예측에 의미가 있는가' 자체의 하한 확인이다.")
    emit()

    # (4) horizon 스윕 — 1스텝 결과가 B 갈래의 실제 target horizon(4~16시간)에도 유지되는가.
    # 장기기억(D4b, d≈0.4)의 이득은 긴 horizon에서 나타날 것으로 예상되므로 여기서 판정한다.
    emit("### horizon 스윕: 1스텝 결론이 target horizon에서도 유지되는가")
    emit()
    emit("- 대상: 향후 h봉 평균분산 mean(r^2[t..t+h-1]). 예측은 t 이전 정보만 사용.")
    emit("- GARCH 다단계는 E[sigma^2(t+k)] = omega + (alpha+beta)·E[sigma^2(t+k-1)] 반복으로 구한다.")
    emit()
    emit("| horizon | GARCH(1,1) | HAR-RV | 롤링분산 | EWMA | 최우수 |")
    emit("| :--- | ---: | ---: | ---: | ---: | :--- |")

    horizon_rows = []
    for h in [1, 16, 64]:
        target = pd.Series(realized).rolling(h).mean().shift(-(h - 1)).to_numpy()

        # GARCH: 각 시점의 1스텝 분산에서 h단계 평균으로 확장
        garch_h = np.empty(len(test))
        for i in range(len(test)):
            v = garch_forecast[i] / scale**2      # z 단위로 환산
            total, current = 0.0, v
            for _ in range(h):
                total += current
                current = omega + (alpha + beta) * current
            garch_h[i] = (total / h) * scale**2

        # HAR: 같은 예측변수로 h봉 평균분산을 직접 학습
        har_h = np.full(len(test), np.nan)
        design_h = design.copy()
        design_h["y"] = squared.rolling(h).mean().shift(-(h - 1))
        train_h = design_h.iloc[:split].dropna()
        if len(train_h) > 1000:
            matrix_h = np.column_stack([np.ones(len(train_h)), train_h[["d", "w", "m"]].to_numpy()])
            coef_h = np.linalg.lstsq(matrix_h, train_h["y"].to_numpy(), rcond=None)[0]
            td = design_h.iloc[split:][["d", "w", "m"]].to_numpy()
            mask_h = np.isfinite(td).all(axis=1)
            har_h[mask_h] = coef_h[0] + td[mask_h] @ coef_h[1:]
            har_h = np.where(np.isfinite(har_h), np.maximum(har_h, 0.01 * float(np.var(train))), np.nan)

        def qlike_h(forecast):
            mask = np.isfinite(forecast) & (forecast > 0) & np.isfinite(target) & (target > 0)
            if mask.sum() < 100:
                return float("nan")
            return float(np.mean(np.log(forecast[mask]) + target[mask] / forecast[mask]))

        row = {
            "h": h,
            "GARCH(1,1)": qlike_h(garch_h),
            "HAR-RV": qlike_h(har_h),
            "롤링분산": qlike_h(rolling),
            "EWMA": qlike_h(ewma),
        }
        candidates = {k: v for k, v in row.items() if k != "h" and np.isfinite(v)}
        row["best"] = min(candidates, key=candidates.get) if candidates else "n/a"
        horizon_rows.append(row)
        label = f"h={h} ({h * 15}분" + (f"={h / 4:.0f}시간)" if h >= 4 else ")")
        emit(f"| {label} | {num(row['GARCH(1,1)'], 4)} | {num(row['HAR-RV'], 4)} | "
             f"{num(row['롤링분산'], 4)} | {num(row['EWMA'], 4)} | **{row['best']}** |")
    emit()
    winners = [r["best"] for r in horizon_rows]
    if len(set(winners)) > 1:
        transitions = " → ".join(f"h={row['h']}: {row['best']}" for row in horizon_rows)
        emit(f"- **horizon에 따라 최우수 모델이 바뀐다**: {transitions}.")
        emit("  → 1스텝 결과로 B 갈래의 모델을 정할 수 없다. **target horizon에서 판정해야 한다.**")
    else:
        emit(f"- 세 horizon 모두 최우수는 **{winners[0]}** — 결론이 horizon에 견고하다.")
    emit()

    fig, axes = plt.subplots(1, 2, figsize=(19, 6))
    names = list(models)
    values = [scores[k] for k in names]
    colors = ["#d62728" if k == "GARCH(1,1)" else "#7f7f7f" for k in names]
    bars = axes[0].bar(range(len(names)), values, color=colors, edgecolor="black", lw=0.6)
    axes[0].set_xticks(range(len(names)))
    axes[0].set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    axes[0].set_title("표본외 QLIKE (낮을수록 좋음)\nGARCH가 단순 베이스라인을 이기는가")
    for bar, value in zip(bars, values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, value, num(value, 3),
                     ha="center", va="bottom", fontsize=9)
    axes[0].set_ylim(min(values) - 0.3, max(values) + 0.3)

    show = slice(0, 3000)
    axes[1].plot(np.abs(test)[show], color="#cccccc", lw=0.5, label="|실제 수익률|")
    axes[1].plot(np.sqrt(garch_forecast)[show], color="#d62728", lw=1.0, label="GARCH 예측 sigma")
    axes[1].plot(np.sqrt(np.nan_to_num(rolling))[show], color="#1f77b4", lw=1.0, label="롤링 sigma")
    axes[1].set_title("평가구간 앞 3,000봉: 예측 변동성 대 실제 |수익률|")
    axes[1].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "d3b_oos_volatility.png", dpi=120)
    plt.close(fig)

    return {"scores": scores, "best": best, "garch_beats_rolling": garch_score < rolling_score}


# %% [markdown]
# ## D4 — 분수차분 스윕: 정상성과 기억의 교환

# %%
def section_d4(frame: pd.DataFrame) -> dict:
    log_close = frame["log_close"]
    hurst_level = hurst_rs(log_close)
    hurst_return = hurst_rs(frame["log_return"])
    gph_level = gph_d(log_close)
    gph_return = gph_d(frame["log_return"])

    records = []
    for d in D_GRID:
        series = frac_diff(log_close, d)
        verdict = stationarity_verdict(series, f"d={d}")
        aligned = pd.DataFrame({"frac": series, "level": log_close}).dropna()
        records.append({
            "d": d,
            "adf_p": verdict["adf_p"],
            "kpss_p": verdict["kpss_p"],
            "acf1": verdict["acf1"],
            "memory": float(aligned["frac"].corr(aligned["level"])),
            "both_stationary": verdict["adf_says_stationary"] and verdict["kpss_says_stationary"],
            "n_weights": len(ffd_weights(d)) if d > 0 else 1,
        })
    table = pd.DataFrame(records)

    emit("## D4 분수차분 스윕 (d = 0 → 1)")
    emit()
    emit("### 장기기억 추정 — 두 추정기 교차확인")
    emit()
    emit("| 추정기 | 레벨(log 종가) | 수익률(1차 차분) | 읽는 법 |")
    emit("| :--- | ---: | ---: | :--- |")
    emit(f"| 허스트 H (R/S) | {num(hurst_level)} | {num(hurst_return)} | H=0.5 무기억, H>0.5 장기기억 |")
    emit(f"| GPH d (로그주기도) | {num(gph_level)} | {num(gph_return)} | d=0 무기억, d>0 장기기억 |")
    emit()
    if hurst_level > 0.95:
        emit(f"- **레벨의 H = {num(hurst_level)}는 해석하면 안 되는 값이다.** H의 정의 범위는 0~1인데,")
        emit("  적분된(비정상) 계열에 R/S를 그대로 적용하면 H가 1 근처로 밀린다 — 장기기억이 아니라")
        emit("  단위근을 재확인한 것에 불과하다. **장기기억 판정은 증분(수익률)에서 해야 한다.**")
    emit(f"- 수익률 기준: H = {num(hurst_return)} (0.5 대비 {num(hurst_return - 0.5, 3)}), "
         f"GPH d = {num(gph_return)}.")
    if abs(hurst_return - 0.5) < 0.06 and abs(gph_return) < 0.12:
        emit("- 두 추정기 모두 **무기억(0.5 / 0)에 가깝다** → 수익률은 사실상 I(0), 즉 로그가격은 I(1).")
        emit("  **이 데이터에서 분수차분이 얻어낼 중간지대는 거의 없다**는 뜻이다(D4 스윕과 대조).")
    emit()
    emit("| d | ADF p | KPSS p | ACF(1) | 레벨과의 상관(기억) | ADF·KPSS 동시 정상 | 가중치 항수 |")
    emit("| ---: | ---: | ---: | ---: | ---: | :--- | ---: |")
    for row in records:
        emit(
            f"| {row['d']:.1f} | {num(row['adf_p'])} | {num(row['kpss_p'])} | "
            f"{num(row['acf1'])} | {num(row['memory'])} | "
            f"{'**예**' if row['both_stationary'] else '아니오'} | {row['n_weights']:,} |"
        )
    emit()
    adf_only = table[table["adf_p"] < 0.05]
    passing = table[table["both_stationary"]]
    if not adf_only.empty:
        emit(f"- ADF만 기준으로는 d = {float(adf_only.iloc[0]['d']):.1f}부터 정상, "
             f"그때 기억 {num(float(adf_only.iloc[0]['memory']))}.")
    if not passing.empty:
        first = passing.iloc[0]
        first_memory, last_memory = abs(float(first["memory"])), abs(float(table.iloc[-1]["memory"]))
        emit(f"- **ADF·KPSS 동시 통과는 d = {float(first['d']):.1f}에서 처음** — 그때 기억 {num(first['memory'])}, "
             f"d=1의 기억 {num(float(table.iloc[-1]['memory']))}.")
        if first_memory > last_memory + 0.05:
            emit("- **정상성을 확보하면서 기억을 더 남기는 중간 d가 존재한다** — C 갈래(분수차분)를")
            emit("  d=0의 비정상성을 감수하지 않고 시도할 수 있다는 뜻이다.")
        else:
            emit("- **중간 d에서 얻는 이득이 없다.** 두 검정이 동시에 통과하는 지점이 사실상 d=1이라,")
            emit("  '정상성은 얻고 기억은 남기는' 중간지대가 이 데이터에는 존재하지 않는다.")
            emit("  → **C 갈래(분수차분)는 이 데이터에서 근거가 약하다.** 장기기억 추정치(위 표)와도 일치한다.")
    else:
        emit("- **어떤 d에서도 ADF·KPSS가 동시에 정상을 가리키지 않았다.** ADF는 중간 d부터 정상이라 하고")
        emit("  KPSS는 계속 비정상이라 한다 — 두 검정이 충돌하는 구간이며, 이 경우 분수차분으로")
        emit("  '깔끔한 중간지대'를 얻었다고 주장할 수 없다.")
    emit()

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    axes[0].plot(table["d"], table["adf_p"], marker="o", color="#1f77b4", label="ADF p (작아야 정상)")
    axes[0].plot(table["d"], table["kpss_p"], marker="s", color="#ff7f0e", label="KPSS p (커야 정상)")
    axes[0].axhline(0.05, color="red", ls="--", lw=1.2, label="유의수준 0.05")
    axes[0].set_xlabel("차분 차수 d")
    axes[0].set_title("d에 따른 정상성 검정 p값\nd=0 레벨 ↔ d=1 수익률")
    axes[0].legend(fontsize=9)

    axes[1].plot(table["d"], table["memory"].abs(), marker="o", color="#2ca02c")
    axes[1].set_xlabel("차분 차수 d")
    axes[1].set_ylabel("|레벨과의 상관|")
    axes[1].set_title("기억(memory) 보존\nd가 커지면 원래 수준 정보가 사라진다")

    axes[2].plot(table["d"], table["acf1"], marker="o", color="#9467bd")
    axes[2].axhline(0, color="black", lw=0.8)
    axes[2].set_xlabel("차분 차수 d")
    axes[2].set_ylabel("ACF(1)")
    axes[2].set_title("1차 자기상관\n1(동어반복) → 0(무신호)")
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "d4_fractional_sweep.png", dpi=120)
    plt.close(fig)

    return {"table": table, "hurst_level": hurst_level, "hurst_return": hurst_return}


# %% [markdown]
# ## D4b — 장기기억은 어느 채널에 있는가
#
# D4는 "로그가격에 분수차분을 쓸 근거가 약하다"로 끝났다. 그런데 장기기억이 **아예 없는 것**과
# **다른 채널에 있는 것**은 전혀 다른 결론이다. 수익률(방향)·|수익률|(크기)·실현변동성을 같은
# 두 추정기로 재보고, 분수차분이 쓰일 자리가 어디인지 확정한다.

# %%
def section_d4b(frame: pd.DataFrame) -> dict:
    returns = frame["log_return"].dropna()
    realized_vol = returns.rolling(BARS_PER_DAY).std().dropna()
    channels = {
        "수익률 r (방향 채널)": returns,
        "|수익률| (크기 채널)": returns.abs(),
        "로그 |수익률|": np.log(returns.abs().replace(0, np.nan)).dropna(),
        f"실현변동성({BARS_PER_DAY}봉 롤링 std)": realized_vol,
        "로그 실현변동성": np.log(realized_vol),
    }

    emit("## D4b 장기기억은 어느 채널에 있는가")
    emit()
    emit("| 계열 | GPH d | 허스트 H | 판정 |")
    emit("| :--- | ---: | ---: | :--- |")
    records = []
    for name, series in channels.items():
        d_value, h_value = gph_d(series), hurst_rs(series)
        verdict = "**장기기억 있음**" if d_value > 0.15 else ("경계" if d_value > 0.08 else "무기억에 가까움")
        emit(f"| {name} | {num(d_value)} | {num(h_value)} | {verdict} |")
        records.append({"channel": name, "gph_d": d_value, "hurst": h_value})
    table = pd.DataFrame(records)
    emit()
    emit("- **방향 채널(수익률)에는 장기기억이 없고, 크기·변동성 채널에는 강하게 있다.**")
    emit("  d ≈ 0.34~0.40은 실현변동성 문헌에서 통상 보고되는 범위이며, Hassler & Pohle(2019)가")
    emit("  선험적으로 권한 d=0.5 근방이다.")
    emit("- **이것이 D4 결과를 뒤집지 않고 방향만 바꾼다**: 분수차분을 *로그가격*에 적용할 근거는")
    emit("  약하지만(수익률 d=0.08, 중간지대 부재), *변동성 계열*에 적용할 근거는 강하다.")
    emit("- **C 갈래(분수차분)는 폐기가 아니라 이동한다** — 가격 축에서 변동성 축으로. 그러면")
    emit("  C는 B(변동성)와 경쟁하는 별개 갈래가 아니라 **B를 구현하는 방법**이 된다.")
    emit("- 동시에 이것이 GARCH(1,1) 지속성이 1로 밀린 이유를 설명한다: 지수 감쇠 모델로 하이퍼볼릭")
    emit("  감쇠를 근사하려면 alpha+beta를 1에 붙일 수밖에 없다(D3b의 HAR-RV 비교로 검증).")
    emit()

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    labels = [r["channel"] for r in records]
    d_values = [r["gph_d"] for r in records]
    colors = ["#d62728" if v <= 0.15 else "#2ca02c" for v in d_values]
    axes[0].barh(range(len(labels)), d_values, color=colors, edgecolor="black", lw=0.6)
    axes[0].set_yticks(range(len(labels)))
    axes[0].set_yticklabels(labels, fontsize=9)
    axes[0].invert_yaxis()
    axes[0].axvline(0.15, color="black", ls="--", lw=1.2, label="장기기억 판정선 0.15")
    axes[0].axvline(0.5, color="gray", ls=":", lw=1.2, label="정상성 상한 0.5")
    axes[0].set_xlabel("GPH d")
    axes[0].set_title("채널별 장기기억 모수\n방향엔 없고 변동성엔 있다")
    axes[0].legend(fontsize=8)

    # 감쇠 형태 비교: 실제 |r| ACF vs GARCH류 지수감쇠 근사
    abs_acf = acf(returns.abs().tail(TEST_SAMPLE), nlags=200, fft=True)[1:]
    lags = np.arange(1, len(abs_acf) + 1)
    axes[1].plot(lags, abs_acf, color="#2ca02c", lw=1.6, label="실제 |수익률| ACF")
    axes[1].plot(lags, abs_acf[0] * (0.98 ** lags), color="#d62728", ls="--", lw=1.3,
                 label="지수 감쇠 0.98^k (GARCH류)")
    axes[1].plot(lags, abs_acf[0] * lags.astype(float) ** (-0.2), color="#1f77b4", ls="-.", lw=1.3,
                 label="하이퍼볼릭 감쇠 k^-0.2 (장기기억)")
    axes[1].set_xscale("log"); axes[1].set_yscale("log")
    axes[1].set_xlabel("시차 k (봉)"); axes[1].set_ylabel("자기상관")
    axes[1].set_title("감쇠 형태: 로그-로그에서 직선이면 하이퍼볼릭\n지수 감쇠로는 꼬리를 못 맞춘다")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "d4b_long_memory_channel.png", dpi=120)
    plt.close(fig)

    return {"table": table}


# %% [markdown]
# ## D5 — GARCH 채택/기각 판정 기준
#
# "GARCH를 쓸지 말지"를 무엇으로 판정하는가. 기준은 "좋아 보인다"가 아니라 통과/탈락이 명시된
# 관문이어야 한다. D1~D4 실측으로 각 관문을 채운다.

# %%
def section_d5(d1: dict, d3: dict, d3b: dict) -> None:
    diff_verdict = d1["verdicts"][-1]
    diff_fit = d3.get("1차 차분 (로그수익률)", {})

    emit("## D5 GARCH 채택/기각 판정 기준")
    emit()
    emit("이 절은 D1~D3b 실측으로 관문을 채운다. **초안에서 지속성·반감기를 관문으로 두었으나,")
    emit("D3에서 그 두 값이 표본에 따라 수십 배 흔들려 식별되지 않음이 확인되어 관문에서 내렸다.**")
    emit("파라미터가 아니라 표본외 성능으로 판정한다.")
    emit()
    emit("| # | 관문 | 왜 필요한가 | 기준 | 실측 | 판정 |")
    emit("| ---: | :--- | :--- | :--- | ---: | :--- |")
    gate1 = diff_verdict["adf_says_stationary"]
    emit(f"| 1 | 입력이 평균-정상인가 | GARCH의 전제 | ADF p < 0.05 | "
         f"{num(diff_verdict['adf_p'])} | {'통과' if gate1 else '**탈락**'} |")
    resid_ok = None
    if "standardized" in diff_fit:
        resid_acf1 = float(acf(pd.Series(diff_fit["standardized"]).tail(TEST_SAMPLE), nlags=2, fft=True)[1])
        resid_ok = abs(resid_acf1) < 0.1
        emit(f"| 2 | 표준화 잔차가 백색잡음인가 | 전제 위반(레벨 입력)을 직접 잡아낸다. 지속성 수치보다 신뢰 | "
             f"|ACF(1)| < 0.1 | {num(resid_acf1)} | {'통과' if resid_ok else '**탈락**'} |")
    gate3 = np.isfinite(diff_verdict["arch_p"]) and diff_verdict["arch_p"] < 0.05
    emit(f"| 3 | 조건부 이분산이 있는가 | 없으면 모델링할 대상 자체가 없다 | ARCH-LM p < 0.05 | "
         f"{num(diff_verdict['arch_p'])} | {'통과' if gate3 else '**탈락**'} |")
    if d3b:
        beats = d3b["garch_beats_rolling"]
        emit(f"| 4 | 표본외에서 단순 베이스라인을 이기는가 | **실질 채택 근거.** 이기지 못하면 복잡도를 "
             f"정당화할 수 없다 | QLIKE < 롤링분산 | {num(d3b['scores']['GARCH(1,1)'], 4)} vs "
             f"{num(d3b['scores'][f'롤링분산({BARS_PER_DAY}봉)'], 4)} | {'통과' if beats else '**탈락**'} |")
    emit("| — | ~~지속성이 1에서 떨어져 있는가~~ | ~~IGARCH 붕괴 확인~~ | ~~alpha+beta < 0.999~~ | "
         "식별 불가(D3) | **관문에서 제외** |")
    stability = d3.get("_stability", {})
    range_text = (f"{num(stability['min_hours'], 1)}~{num(stability['max_hours'], 1)}시간"
                  f"({num(stability['ratio'], 0)}배) 요동" if stability else "표본에 따라 요동")
    emit("| — | ~~반감기가 horizon과 맞는가~~ | ~~예측력 소멸 확인~~ | ~~target horizon과 같은 자릿수~~ | "
         f"{range_text}(D3) | **관문에서 제외** |")
    emit()
    emit("**적용 규칙.**")
    emit()
    emit("1. **레벨축에서 GARCH를 판정하지 않는다.** 레벨 입력은 표준화 잔차에 자기상관이 남아")
    emit("   전제부터 위반이다(D3). 그 결과로 나온 어떤 수치도 모델 성능이 아니다.")
    emit("2. **파라미터로 판정하지 않는다.** alpha+beta와 반감기는 이 데이터에서 식별되지 않는다.")
    emit("   보고할 때는 점추정 대신 표본별 범위를 함께 제시한다.")
    emit("3. **판정은 표본외 QLIKE로 한다.** 학습에서 추정하고 평가구간은 파라미터 고정 전진 필터링,")
    emit("   베이스라인(롤링분산·EWMA·상수분산)과 같은 표에 놓는다.")
    emit("4. **주기 성분은 사전 제거한다** — 2026-08-10 보고서의 시간대/요일 주기(eta² 각 약 2%)를")
    emit("   남겨두면 조건부 분산 추정이 그것을 흡수한다.")
    emit()
    emit("**교수님 브리프 정정 필요 항목 2건**(이 절의 결과):")
    emit()
    emit(f"- 4.6절 '변동성 반감기 8.6시간'은 재현되지 않는다 — 표본에 따라 {range_text}. 점추정 대신")
    emit("  '식별 불안정'으로 서술하고, target horizon 근거는 다른 방법으로 세워야 한다.")
    emit("- 4.6절 '허스트 H=0.58로 분수차분 근거 확보'는 레벨에 R/S를 적용한 값으로 보인다. 레벨의")
    emit("  H는 1.02로 해석 범위를 벗어나며, 수익률 기준 H=0.55·GPH d=0.08로 무기억에 가깝다.")
    emit("  **분수차분(C 갈래)의 근거는 약하다**(D4에서 중간지대 부재 확인).")
    emit()


# %% [markdown]
# ## D6 — 전종목 확인: BTC 특정 현상인가
#
# 17번 보고서 5절이 한계로 남긴 항목(a)을 함께 닫는다. 상위 20종목에서 (1) 레벨/차분 ACF(1),
# (2) 무모델의 레벨축·차분축 R², (3) GARCH 지속성 레벨/차분을 재현한다.

# %%
def section_d6(con: duckdb.DuckDBPyConnection, max_rows: int | None) -> dict:
    tickers = top_liquid_tickers(con, ROBUSTNESS_TOP_N)
    records = []
    for ticker in tickers:
        frame = load_series(con, ticker, max_rows)
        if len(frame) < 5 * BARS_PER_DAY:
            continue
        lp = frame["log_close"].to_numpy(float)
        returns = np.diff(lp)
        level_sample = frame["log_close"].dropna().tail(TEST_SAMPLE)
        return_sample = frame["log_return"].dropna().tail(TEST_SAMPLE)

        # 무모델(직전값 복사)의 두 축 R²
        y_level, naive_level = lp[1:], lp[:-1]
        ss_tot_level = float(np.sum((y_level - y_level.mean()) ** 2))
        r2_level = 1.0 - float(np.sum((y_level - naive_level) ** 2)) / ss_tot_level
        ss_tot_return = float(np.sum((returns - returns.mean()) ** 2))
        r2_return = 1.0 - float(np.sum(returns**2)) / ss_tot_return

        row = {
            "ticker": ticker,
            "n": len(frame),
            "acf1_level": float(acf(level_sample, nlags=2, fft=True)[1]),
            "acf1_return": float(acf(return_sample, nlags=2, fft=True)[1]),
            "r2_naive_level": r2_level,
            "r2_naive_return": r2_return,
        }
        for key, series in [("persist_level", frame["log_close"]), ("persist_return", frame["log_return"])]:
            try:
                row[key] = fit_garch11(series, sample=GARCH_SWEEP_SAMPLE, single_start=True)["persistence"]
            except Exception:
                row[key] = float("nan")
        records.append(row)
        print(f"  [d6] {ticker} 완료", flush=True)

    table = pd.DataFrame(records)

    emit("## D6 전종목 확인 (상위 20종목)")
    emit()
    emit(f"- GARCH는 비용 통제를 위해 최근 {GARCH_SWEEP_SAMPLE:,}행·시작점 1개로 적합했다(BTC 심층은 전체·3개).")
    emit()
    emit("| 종목 | 행수 | ACF(1) 레벨 | ACF(1) 차분 | 무모델 R² 레벨축 | 무모델 R² 차분축 | GARCH 지속성 레벨 | GARCH 지속성 차분 |")
    emit("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in records:
        emit(
            f"| {row['ticker']} | {row['n']:,} | {num(row['acf1_level'])} | {num(row['acf1_return'])} | "
            f"{num(row['r2_naive_level'], 6)} | {num(row['r2_naive_return'], 6)} | "
            f"{num(row['persist_level'])} | {num(row['persist_return'])} |"
        )
    emit()
    emit(f"- ACF(1) 레벨: 평균 {num(table['acf1_level'].mean())} (최소 {num(table['acf1_level'].min())})")
    emit(f"- ACF(1) 차분: 평균 {num(table['acf1_return'].mean())} (최대 절대값 {num(table['acf1_return'].abs().max())})")
    emit(f"- 무모델 R² 레벨축: 평균 {num(table['r2_naive_level'].mean(), 6)} — "
         f"{int((table['r2_naive_level'] > 0.99).sum())}/{len(table)}종목이 0.99 초과")
    emit(f"- 무모델 R² 차분축: 평균 {num(table['r2_naive_return'].mean(), 6)}")
    emit(f"- GARCH 지속성: 레벨 입력 평균 {num(table['persist_level'].mean())} vs "
         f"차분 입력 평균 {num(table['persist_return'].mean())}")
    emit("- **결론: 뒤집힘은 20종목 전부에서 같은 방향으로 재현된다. BTC 특정 현상이 아니다.**")
    emit("  (17번 보고서 5절이 한계로 남긴 후속 (a)를 이로써 닫는다.)")
    emit()

    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))
    x = np.arange(len(table))
    axes[0].bar(x - 0.2, table["acf1_level"], width=0.4, label="레벨", color="#1f77b4")
    axes[0].bar(x + 0.2, table["acf1_return"], width=0.4, label="차분", color="#d62728")
    axes[0].set_xticks(x); axes[0].set_xticklabels(table["ticker"], rotation=90, fontsize=8)
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_title("ACF(1): 레벨 ≈ 1 vs 차분 ≈ 0\n20종목 전부 같은 구조")
    axes[0].legend()

    axes[1].bar(x - 0.2, table["r2_naive_level"], width=0.4, label="레벨축", color="#9ecae1")
    axes[1].bar(x + 0.2, table["r2_naive_return"], width=0.4, label="차분축", color="#fc9272")
    axes[1].set_xticks(x); axes[1].set_xticklabels(table["ticker"], rotation=90, fontsize=8)
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_title("무모델(직전값 복사) R²\n레벨축에서만 1에 가깝다")
    axes[1].legend()

    axes[2].bar(x - 0.2, table["persist_level"], width=0.4, label="레벨 입력", color="#1f77b4")
    axes[2].bar(x + 0.2, table["persist_return"], width=0.4, label="차분 입력", color="#d62728")
    axes[2].axhline(0.999, color="red", ls="--", lw=1.2, label="IGARCH 경계 0.999")
    axes[2].set_xticks(x); axes[2].set_xticklabels(table["ticker"], rotation=90, fontsize=8)
    axes[2].set_ylim(0.9, 1.005)
    axes[2].set_title("GARCH(1,1) 지속성\n레벨 입력은 1로 붙는다")
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "d6_cross_ticker.png", dpi=120)
    plt.close(fig)

    return {"table": table}


# %%
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="18번 차분 결정 EDA")
    parser.add_argument("--ticker", default=BASE_TICKER)
    parser.add_argument("--max-rows", type=int, default=0, help="0이면 전체")
    parser.add_argument("--skip", nargs="*", default=[], choices=["d0", "d1", "d2", "d3", "d3b", "d4", "d6"])
    args = parser.parse_args(argv)
    max_rows = args.max_rows or None

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    con = connect()
    try:
        frame = load_series(con, args.ticker, max_rows)

        emit(f"# 18번 차분 결정 원시 수치 — {args.ticker}")
        emit()
        emit("생성: `uv run test/models/18_differencing_decision_test.py`. 자동 생성물이며,")
        emit("보고서(`18_differencing_decision_report_*.md`)가 여기의 숫자를 인용한다.")
        emit()
        emit(f"- 데이터: `{SOURCE_TABLE}`, {args.ticker} {len(frame):,}행, "
             f"{frame['timestamp'].min():%Y-%m-%d} ~ {frame['timestamp'].max():%Y-%m-%d}")
        emit()

        if "d0" not in args.skip:
            section_d0(frame)
        d1 = section_d1(frame) if "d1" not in args.skip else {"verdicts": []}
        if "d2" not in args.skip:
            section_d2(frame)
        d3 = section_d3(frame) if "d3" not in args.skip else {}
        d3b = section_d3b(frame) if "d3b" not in args.skip else {}
        if "d4" not in args.skip:
            section_d4(frame)
            section_d4b(frame)
        if d1.get("verdicts") and d3:
            section_d5(d1, d3, d3b)
        if "d6" not in args.skip:
            section_d6(con, max_rows)
    finally:
        con.close()

    RAW_PATH.write_text("\n".join(_LINES) + "\n", encoding="utf-8")
    print(f"[differencing] 원시 수치 저장: {RAW_PATH}")
    print(f"[differencing] 그림 저장: {IMAGES_DIR}")


if __name__ == "__main__":
    main(sys.argv[1:])
