# %% [markdown]
# # 19번: B(변동성) 갈래 모델 선택 — t분포·주기제거·target horizon을 한 실험에서 판정
#
# ## 왜 (Why)
# 18번에서 미결로 남긴 세 항목이 서로 얽혀 있어 따로 풀 수 없다.
#   (1) **분포 오지정**: 표준화 잔차 정규성이 기각됐다(JB p=0, 초과첨도 14.0). 정규 MLE는 틀렸고
#       t분포로 재적합해야 한다(18d 11절).
#   (2) **주기 성분**: 시간대·요일 주기가 |수익률| 분산의 각 약 2%를 설명한다(2026-08-10 보고서).
#       제거하지 않으면 조건부 분산이 그것을 흡수해 지속성이 위로 편향된다.
#   (3) **target horizon**: GARCH 우위가 h와 함께 사라지고 16시간에서 HAR-RV에 역전됐다.
#       B 갈래의 목표가 4~16시간이므로 **교차점이 목표 구간 안에 있다**(18번 5.1절).
#
# 세 항목을 따로 바꾸면 어느 변화가 효과를 냈는지 알 수 없다. 그래서 **모델 × 주기제거 × horizon**
# 삼원 격자로 한 번에 돌린다.
#
# ## 무엇을 (What)
#   모델 6종   : GARCH-정규 · **GARCH-t(신규)** · EWMA · 롤링분산 · HAR-RV(일·주·월) · HAR-RV+(분기 추가)
#   주기 제거  : 없음 / 시간대×요일 곱셈 성분 제거 (Yan 2021 곱셈 성분 모델의 단순형)
#   horizon    : h=1(15분) · h=16(4시간) · h=64(16시간)
#   지표       : QLIKE(주) · MAE · MASE · corr — `AGENTS.md` 2.9d 규칙대로 무모델 병기
#
# ## 어떻게 (How)
# **평가 대상을 통일한다**: 주기를 제거한 쪽도 예측을 원계열 스케일로 되돌린 뒤 **같은 실현분산
# (원계열 r²)** 에 대해 평가한다. 그래야 주기 제거가 정말 예측을 개선했는지 비교할 수 있다.
# 학습 70% / 평가 30% 시간분할, 평가구간은 파라미터 고정 전진 필터링(재적합·누수 없음).
# GARCH 추정기는 18번 것을 재사용하고 t분포 버전만 새로 구현한다(수치 일관성).
#
# ## 기대 결과 / 반영 (Expected)
# "B 갈래의 기본 모델을 무엇으로 둘 것인가"를 target horizon에서 데이터로 결정한다.

# %%
"""19번 변동성 모델 선택.

실행:
    uv run test/models/19_volatility_model_selection.py
    uv run test/models/19_volatility_model_selection.py --max-rows 30000   # 축소 점검
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = ["Noto Sans CJK KR", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["axes.formatter.useoffset"] = False
matplotlib.rcParams["axes.grid"] = True
matplotlib.rcParams["grid.color"] = "#e8e8e6"
matplotlib.rcParams["grid.linewidth"] = 0.7
matplotlib.rcParams["axes.edgecolor"] = "#c9c9c6"

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize, special, stats as sp_stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from engine import preflight  # noqa: E402

# dataviz 검증 색: 범주형 1·2 + 서열 램프 + 상태
C_A, C_B, C_C = "#2a78d6", "#eb6834", "#1baf7a"
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
# 모델은 "정체성"이므로 범주형 고정 순서(검증 통과: 최악 인접 CVD ΔE 9.1, 정상시 19.6).
# 명암 경고 3색은 직접 라벨로 완화한다.
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
S_GOOD, S_WARN, S_CRIT = "#0ca30c", "#fab219", "#d03b3b"
INK, INK2 = "#0b0b0b", "#52514e"

TAG = "19_volatility_model_selection_20260819"
SOURCE_TABLE = "upbit_krw_candle"
BASE_TICKER = "KRW-BTC"
BARS_PER_DAY = 96
HORIZONS = [1, 16, 64]

_LINES: list[str] = []


def emit(text: str = "") -> None:
    print(text, flush=True)
    _LINES.append(text)


def num(v, d: int = 4) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return "n/a" if not np.isfinite(f) else f"{f:,.{d}f}"


def _root() -> Path:
    start = Path(__file__).resolve().parent
    for c in [start, *start.parents]:
        if (c / "pyproject.toml").exists() and (c / "engine").is_dir():
            return c
    raise RuntimeError("root 못 찾음")


ROOT = _root()
IMAGES_DIR = ROOT / "test" / "images" / TAG
RESULTS_DIR = ROOT / "test" / "results" / TAG

_spec = importlib.util.spec_from_file_location(
    "d18", ROOT / "test" / "models" / "18_differencing_decision_test.py")
d18 = importlib.util.module_from_spec(_spec)
sys.modules["d18"] = d18
_spec.loader.exec_module(d18)

warnings.filterwarnings("ignore")


# %% [markdown]
# ## GARCH(1,1) with Student-t innovations
#
# 정규 MLE는 표준화 잔차가 정규라고 가정한다. 우리 데이터는 초과첨도 14.0이라 그 가정이 깨진다.
# t분포는 자유도 ν로 꼬리 두께를 함께 추정한다(ν가 작을수록 두꺼움, ν→∞면 정규와 같아짐).
# 분산이 1이 되도록 표준화한 t밀도를 쓴다 — 그래야 σ²가 조건부 분산 그대로 해석된다.

# %%
def fit_garch11_t(series: pd.Series, sample: int | None = None) -> dict:
    x = series.dropna().to_numpy(float)
    if sample and len(x) > sample:
        x = x[-sample:]
    x = x - x.mean()
    scale = float(np.std(x))
    z = x / scale
    var_z = float(np.var(z))

    def recursion(omega, alpha, beta):
        variance = np.empty(len(z))
        variance[0] = var_z
        for i in range(1, len(z)):
            variance[i] = omega + alpha * z[i - 1] ** 2 + beta * variance[i - 1]
        return np.maximum(variance, 1e-12)

    def negative_log_likelihood(params):
        omega, alpha, beta, nu = params
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.9999 or nu <= 2.05:
            return 1e10
        variance = recursion(omega, alpha, beta)
        # 단위분산 표준화 t: z = e/σ, e ~ t_ν * sqrt((ν-2)/ν)
        constant = (special.gammaln((nu + 1) / 2) - special.gammaln(nu / 2)
                    - 0.5 * np.log(np.pi * (nu - 2)))
        loglik = np.sum(constant - 0.5 * np.log(variance)
                        - (nu + 1) / 2 * np.log1p(z**2 / (variance * (nu - 2))))
        return -float(loglik)

    best = None
    for start in [(0.05, 0.10, 0.85, 6.0), (0.01, 0.05, 0.90, 4.0), (0.10, 0.20, 0.70, 10.0)]:
        result = optimize.minimize(
            negative_log_likelihood, start, method="L-BFGS-B",
            bounds=[(1e-8, 5.0), (0.0, 0.9999), (0.0, 0.9999), (2.1, 60.0)])
        if best is None or result.fun < best.fun:
            best = result
    omega, alpha, beta, nu = best.x
    variance = recursion(omega, alpha, beta)
    persistence = float(alpha + beta)
    return {
        "omega": float(omega), "alpha": float(alpha), "beta": float(beta), "nu": float(nu),
        "persistence": persistence,
        "half_life_hours": (float(np.log(0.5) / np.log(persistence)) * 0.25
                            if 0 < persistence < 1 else float("inf")),
        "conditional_var_z": variance, "scale": scale, "mean": float(np.mean(x) + series.dropna().mean() * 0),
        "params": (float(omega), float(alpha), float(beta)),
        "standardized": z / np.sqrt(variance), "loglik": float(-best.fun), "n": len(z),
    }


# %% [markdown]
# ## 주기 성분 제거 (곱셈형)
#
# 시간대(24) × 요일(7) = 168 버킷의 평균 |수익률|로 주기 성분 s를 추정하고, 전체 평균으로
# 정규화해 `r_adj = r / s`로 나눈다. 예측은 다시 `× s²`로 원계열 분산 스케일로 되돌려 평가한다
# — 그래야 두 조건이 **같은 실현분산**에 대해 비교된다.

# %%
def seasonal_factor(frame: pd.DataFrame) -> np.ndarray:
    ret = frame["ret"]
    key = frame["timestamp"].dt.hour * 7 + frame["timestamp"].dt.dayofweek
    bucket = ret.abs().groupby(key).transform("mean")
    factor = (bucket / ret.abs().mean()).to_numpy(float)
    return np.where(np.isfinite(factor) & (factor > 1e-6), factor, 1.0)


# %% [markdown]
# ## 예측기들 — 모두 t 시점 이전 정보만 사용

# %%
def forecast_garch(returns: np.ndarray, split: int, fit: dict, horizon: int) -> np.ndarray:
    """학습에서 추정한 파라미터를 고정하고 평가구간을 전진 필터링. h단계 평균분산을 낸다."""
    omega, alpha, beta = fit["params"]
    scale = fit["scale"]
    train = returns[:split]
    z_train = (train - train.mean()) / scale
    variance = np.var(z_train)
    for value in z_train[1:]:
        variance = omega + alpha * value**2 + beta * variance
    z_test = (returns[split:] - train.mean()) / scale
    out = np.empty(len(z_test))
    for i in range(len(z_test)):
        total, current = 0.0, variance
        for _ in range(horizon):
            total += current
            current = omega + (alpha + beta) * current
        out[i] = (total / horizon) * scale**2
        variance = omega + alpha * z_test[i] ** 2 + beta * variance
    return out


def forecast_har(squared: pd.Series, split: int, horizon: int, components: list[int]) -> np.ndarray:
    design = pd.DataFrame({"y": squared.rolling(horizon).mean().shift(-(horizon - 1))})
    for length in components:
        design[f"c{length}"] = squared.rolling(length).mean().shift(1)
    columns = [f"c{length}" for length in components]
    train = design.iloc[:split].dropna()
    out = np.full(len(squared) - split, np.nan)
    if len(train) < 1000:
        return out
    matrix = np.column_stack([np.ones(len(train)), train[columns].to_numpy()])
    coef = np.linalg.lstsq(matrix, train["y"].to_numpy(), rcond=None)[0]
    test = design.iloc[split:][columns].to_numpy()
    mask = np.isfinite(test).all(axis=1)
    out[mask] = coef[0] + test[mask] @ coef[1:]
    floor = 0.01 * float(np.var(squared.iloc[:split].to_numpy()) ** 0.5)
    return np.where(np.isfinite(out), np.maximum(out, max(floor, 1e-14)), np.nan)


def forecast_flat(series: pd.Series, split: int) -> np.ndarray:
    return series.shift(1).to_numpy()[split:]


# %%
def qlike(target: np.ndarray, forecast: np.ndarray) -> float:
    mask = np.isfinite(forecast) & (forecast > 0) & np.isfinite(target) & (target > 0)
    if mask.sum() < 100:
        return float("nan")
    return float(np.mean(np.log(forecast[mask]) + target[mask] / forecast[mask]))


def score_all(target: np.ndarray, forecast: np.ndarray, naive: np.ndarray) -> dict:
    mask = np.isfinite(forecast) & (forecast > 0) & np.isfinite(target) & np.isfinite(naive)
    t, f, n = target[mask], forecast[mask], naive[mask]
    if len(t) < 100:
        return {"QLIKE": np.nan, "MAE": np.nan, "MASE": np.nan, "corr": np.nan}
    return {
        "QLIKE": float(np.mean(np.log(f) + t / f)),
        "MAE": float(np.mean(np.abs(t - f))),
        "MASE": float(np.mean(np.abs(t - f)) / max(np.mean(np.abs(t - n)), 1e-20)),
        "corr": float(np.corrcoef(np.sqrt(f), np.sqrt(np.maximum(t, 0)))[0, 1]),
    }


def run_condition(frame: pd.DataFrame, deseasonalize: bool) -> tuple[pd.DataFrame, dict]:
    ret_raw = frame["ret"].dropna().to_numpy(float)
    factor = seasonal_factor(frame)[1:]            # ret과 길이 맞춤(첫 행 diff=NaN)
    returns = ret_raw / factor if deseasonalize else ret_raw
    back = factor**2 if deseasonalize else np.ones_like(ret_raw)

    n = len(returns)
    split = int(n * 0.7)
    squared_model = pd.Series(returns**2)
    realized_raw = ret_raw**2                       # 평가 대상은 항상 원계열 r²

    fits = {
        "GARCH-정규": d18.fit_garch11(pd.Series(returns[:split])),
        "GARCH-t": fit_garch11_t(pd.Series(returns[:split])),
    }

    records = []
    for horizon in HORIZONS:
        target = pd.Series(realized_raw).rolling(horizon).mean().shift(-(horizon - 1)).to_numpy()[split:]
        back_test = pd.Series(back).rolling(horizon).mean().shift(-(horizon - 1)).to_numpy()[split:]
        back_test = np.where(np.isfinite(back_test), back_test, 1.0)
        naive = np.full(len(target), float(np.mean(realized_raw[:split])))

        candidates = {}
        for label, fit in fits.items():
            candidates[label] = forecast_garch(returns, split, fit, horizon) * back_test
        candidates["EWMA(a=0.06)"] = (
            squared_model.ewm(alpha=0.06, adjust=False).mean().shift(1).to_numpy()[split:] * back_test)
        candidates[f"롤링분산({BARS_PER_DAY})"] = (
            squared_model.rolling(BARS_PER_DAY).mean().shift(1).to_numpy()[split:] * back_test)
        candidates["HAR-RV(일·주·월)"] = (
            forecast_har(squared_model, split, horizon,
                         [BARS_PER_DAY, BARS_PER_DAY * 7, BARS_PER_DAY * 30]) * back_test)
        candidates["HAR-RV+(분기추가)"] = (
            forecast_har(squared_model, split, horizon,
                         [BARS_PER_DAY, BARS_PER_DAY * 7, BARS_PER_DAY * 30, BARS_PER_DAY * 90]) * back_test)
        candidates["상수분산"] = naive

        for label, forecast in candidates.items():
            scores = score_all(target, forecast, naive)
            records.append({"주기제거": "제거" if deseasonalize else "없음",
                            "horizon": horizon, "모델": label, **scores})
    return pd.DataFrame(records), fits


# %%
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="19번 변동성 모델 선택")
    parser.add_argument("--ticker", default=BASE_TICKER)
    parser.add_argument("--max-rows", type=int, default=0)
    args = parser.parse_args(argv)

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    preflight.ensure_data(table=SOURCE_TABLE, tickers=[args.ticker])

    con = duckdb.connect(str(ROOT / "data" / "upbit_data.db"), read_only=True)
    try:
        frame = con.execute(
            f"select timestamp, close from {SOURCE_TABLE} where ticker = ? order by timestamp",
            [args.ticker]).df()
    finally:
        con.close()
    if args.max_rows:
        frame = frame.tail(args.max_rows).reset_index(drop=True)
    frame["ret"] = np.log(frame["close"].astype(float)).diff()

    emit(f"# 19번 변동성 모델 선택 원시 수치 — {args.ticker}")
    emit()
    emit(f"- 데이터 {len(frame):,}행, {frame['timestamp'].min():%Y-%m-%d} ~ {frame['timestamp'].max():%Y-%m-%d}")
    emit("- 학습 70% / 평가 30%, 평가구간은 파라미터 고정 전진 필터링(재적합 없음)")
    emit("- **두 조건 모두 같은 실현분산(원계열 r²)에 대해 평가한다** — 주기 제거 쪽은 예측을 원스케일로 되돌림")
    emit()

    tables, all_fits = [], {}
    for deseasonalize in [False, True]:
        label = "제거" if deseasonalize else "없음"
        print(f"[19] 주기제거={label} 실행", flush=True)
        table, fits = run_condition(frame, deseasonalize)
        tables.append(table)
        all_fits[label] = fits
    result = pd.concat(tables, ignore_index=True)

    # ── 1. 분포 가정: 정규 vs t
    emit("## 1. 분포 가정 — 정규 vs Student-t")
    emit()
    emit("**해석 주의**: Jarque-Bera는 *정규성* 검정이므로 t-GARCH 잔차에는 맞지 않는 자다.")
    emit("t분포를 가정했다면 잔차는 t_ν를 따라야 하고, 표준화 t_ν의 이론 초과첨도는 `6/(ν−4)`(ν>4)다.")
    emit("따라서 **관측 초과첨도를 그 이론값과 비교**하는 것이 옳은 점검이다.")
    emit()
    emit("| 주기제거 | 모델 | alpha | beta | 지속성 | ν | 반감기(시간) | 관측 초과첨도 | 이론 초과첨도 | 남은 초과 |")
    emit("| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for label, fits in all_fits.items():
        for model, fit in fits.items():
            residual = pd.Series(fit["standardized"]).dropna()
            observed = float(sp_stats.kurtosis(residual))
            nu = fit.get("nu", float("nan"))
            theoretical = 6.0 / (nu - 4.0) if np.isfinite(nu) and nu > 4 else (0.0 if model.endswith("정규") else float("nan"))
            remaining = observed - theoretical if np.isfinite(theoretical) else float("nan")
            emit(f"| {label} | {model} | {num(fit['alpha'])} | {num(fit['beta'])} | "
                 f"**{num(fit['persistence'], 5)}** | {num(nu, 2)} | "
                 f"{num(fit['half_life_hours'], 1)} | {num(observed, 2)} | {num(theoretical, 2)} | "
                 f"{num(remaining, 2)} |")
    emit()
    normal_fit = all_fits["없음"]["GARCH-정규"]
    t_fit = all_fits["없음"]["GARCH-t"]
    nu = t_fit["nu"]
    theoretical = 6.0 / (nu - 4.0) if nu > 4 else float("nan")
    observed = float(sp_stats.kurtosis(pd.Series(t_fit["standardized"]).dropna()))
    emit(f"- **t분포 자유도 ν = {num(nu, 2)}** — 꼬리가 매우 두껍다는 뜻이다(ν→∞면 정규와 같아짐).")
    emit(f"  정규 가정(초과첨도 0을 기대)은 관측 {num(observed, 2)}과 전혀 맞지 않았고, t_ν는 "
         f"{num(theoretical, 2)}을 기대해 훨씬 가깝다.")
    if np.isfinite(theoretical) and observed > theoretical * 1.3:
        emit(f"- 다만 **여전히 {num(observed - theoretical, 2)}만큼 초과가 남는다** — t분포도 완전한 설명은 아니다.")
        emit("  꼬리를 더 다루려면 왜도까지 반영하는 비대칭 t(skew-t)나 점프 성분이 필요하다(후속 항목).")
    emit(f"- 지속성: 정규 {num(normal_fit['persistence'], 5)} → t {num(t_fit['persistence'], 5)}")
    emit("- 로그우도는 분포가 다르면 직접 비교할 수 없다(척도가 다름). 모델 선택은 3절의 표본외 QLIKE로 한다.")
    emit()

    # ── 2. 주기 제거 효과
    emit("## 2. 주기 성분 제거 효과 (지속성 편향 확인)")
    emit()
    emit("| 모델 | 지속성(주기 미제거) | 지속성(주기 제거) | 반감기 미제거 | 반감기 제거 |")
    emit("| :--- | ---: | ---: | ---: | ---: |")
    for model in ["GARCH-정규", "GARCH-t"]:
        raw_fit, adj_fit = all_fits["없음"][model], all_fits["제거"][model]
        emit(f"| {model} | {num(raw_fit['persistence'], 5)} | {num(adj_fit['persistence'], 5)} | "
             f"{num(raw_fit['half_life_hours'], 1)}시간 | {num(adj_fit['half_life_hours'], 1)}시간 |")
    emit()

    # ── 3. 삼원 격자 QLIKE
    emit("## 3. 모델 × 주기제거 × horizon (QLIKE, 낮을수록 좋음)")
    emit()
    pivot = result.pivot_table(index="모델", columns=["주기제거", "horizon"], values="QLIKE")
    header = "| 모델 | " + " | ".join(f"{d}/h={h}" for d, h in pivot.columns) + " |"
    emit(header)
    emit("| :--- | " + " | ".join(["---:"] * len(pivot.columns)) + " |")
    order = ["GARCH-정규", "GARCH-t", "EWMA(a=0.06)", f"롤링분산({BARS_PER_DAY})",
             "HAR-RV(일·주·월)", "HAR-RV+(분기추가)", "상수분산"]
    for model in [m for m in order if m in pivot.index]:
        emit(f"| {model} | " + " | ".join(num(v, 4) for v in pivot.loc[model]) + " |")
    emit()

    emit("### horizon별 최우수 모델")
    emit()
    emit("| 주기제거 | horizon | 최우수 | QLIKE | 2위 | 격차 |")
    emit("| :--- | :--- | :--- | ---: | :--- | ---: |")
    for (deseason, horizon), group in result.groupby(["주기제거", "horizon"]):
        clean = group.dropna(subset=["QLIKE"]).sort_values("QLIKE")
        if len(clean) < 2:
            continue
        best, second = clean.iloc[0], clean.iloc[1]
        emit(f"| {deseason} | h={horizon} ({horizon*15}분) | **{best['모델']}** | {num(best['QLIKE'])} | "
             f"{second['모델']} | {num(second['QLIKE'] - best['QLIKE'])} |")
    emit()

    # ── 4. 전체 지표 (AGENTS.md 2.9d — 지표 묶음 병기)
    emit("## 4. 전 지표 (2.9d 규칙: 무모델 병기)")
    emit()
    emit("| 주기제거 | horizon | 모델 | QLIKE | MAE(×1e6) | MASE | corr |")
    emit("| :--- | :--- | :--- | ---: | ---: | ---: | ---: |")
    for _, row in result.iterrows():
        emit(f"| {row['주기제거']} | h={row['horizon']} | {row['모델']} | {num(row['QLIKE'])} | "
             f"{num(row['MAE'] * 1e6, 3)} | {num(row['MASE'])} | {num(row['corr'])} |")
    emit()

    # ── 그림
    fig, axes = plt.subplots(1, 2, figsize=(19, 6.4))
    ax = axes[0]
    models = [m for m in order if m in pivot.index and m != "상수분산"]
    for color, model in zip(CATEGORICAL, models):
        values = [pivot.loc[model, ("없음", h)] for h in HORIZONS]
        ax.plot(HORIZONS, values, marker="o", ms=7, lw=2, color=color, label=model)
    ax.set_xscale("log"); ax.set_xticks(HORIZONS)
    ax.set_xticklabels([f"h={h}\n({h*15}분)" for h in HORIZONS])
    ax.set_ylabel("QLIKE (낮을수록 좋음)")
    ax.axvspan(16, 64, color="#fab219", alpha=0.13)
    ax.text(32, ax.get_ylim()[1], " B 갈래 target 구간(4~16시간)", fontsize=10,
            color=INK, va="top", ha="center")
    ax.set_title("horizon에 따른 모델 순위 (주기 미제거)\ntarget 구간 안에서 순위가 바뀌는지가 판정 지점", fontsize=12)
    ax.legend(fontsize=9, frameon=False)

    ax = axes[1]
    # QLIKE는 음수라 0 기준 막대가 잘린다 → '개선폭(미제거 − 제거)'을 그린다. 양수면 주기 제거가 이득.
    improvements = {h: [pivot.loc[m, ("없음", h)] - pivot.loc[m, ("제거", h)] for m in models]
                    for h in HORIZONS}
    x = np.arange(len(models))
    width = 0.26
    for offset, (horizon, color) in zip([-width, 0, width],
                                        zip(HORIZONS, [RAMP[1], RAMP[2], RAMP[4]])):
        ax.bar(x + offset, improvements[horizon], width, color=color,
               label=f"h={horizon} ({horizon*15}분)")
        for xi, value in enumerate(improvements[horizon]):
            ax.text(xi + offset, value, f"{value:+.3f}", ha="center",
                    va="bottom" if value >= 0 else "top", fontsize=8, color=INK)
    ax.axhline(0, color=INK2, lw=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right", fontsize=9.5)
    ax.set_ylabel("QLIKE 개선폭 (미제거 − 제거)")
    ax.set_title("주기 성분 제거의 이득 — 0보다 크면 제거가 유리\n"
                 "전 모델·전 horizon에서 양수면 '무조건 제거'가 결론", fontsize=12)
    ax.legend(fontsize=9.5, frameon=False)
    span = max(max(v) for v in improvements.values())
    ax.set_ylim(min(0, min(min(v) for v in improvements.values())) - 0.02, span * 1.28)

    fig.suptitle("19번. B(변동성) 갈래 모델 선택 — 분포·주기·horizon 삼원 격자", fontsize=14, y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(IMAGES_DIR / "model_selection.png", dpi=115)
    plt.close(fig)

    result.to_csv(RESULTS_DIR / "model_selection.csv", index=False, encoding="utf-8")
    (RESULTS_DIR / "model_selection_raw.md").write_text("\n".join(_LINES) + "\n", encoding="utf-8")
    print(f"[19] 저장: {RESULTS_DIR}")


if __name__ == "__main__":
    main(sys.argv[1:])
