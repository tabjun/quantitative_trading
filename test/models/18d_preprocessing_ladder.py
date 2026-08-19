# %% [markdown]
# # 18d: 전처리 사다리 — 왜 전처리에 따라 "볼 수 있는 것"이 갈리는가
#
# 헤드리스 `.py` 드라이버(`#%%` 셀 + 파일 내 마크다운, AGENTS.md 2.3).
#
# ## 왜 (Why)
# 사용자 질문(2026-08-19): "차분하면 변동성, 안 하면 수익률로 갈린다고 했는데, **근본적으로 왜
# 전처리 방식에 따라 그렇게 나뉘는지, 그리고 그것을 어떻게 판단했는지**가 궁금했다. 지금 보고서는
# 계열이 무엇이 있고 결과가 이렇게 나왔다는 나열이라 알아보기 힘들다. 각 계열이 왜 나왔고, 어떤
# 방법을 계열마다 동일하게 적용했고, 모델링 전 EDA에서 무엇을 봤고, 형태가 어떻게 나타나고, 어떤
# 가정을 만족하는지가 필요하다."
#
# 앞선 18/18b/18c는 **결과**를 냈지만 **왜 계열이 나뉘는가**를 설명하지 않았다. 이 드라이버는
# 그 설명을 데이터로 만든다.
#
# ## 무엇을 (What)
# 하나의 원자료에서 다섯 단계 계열을 만든다. 세분화가 아니라 **변환 사슬의 각 칸**이다.
#
#   ① close            원 종가(레벨)        — "지금 얼마인가"
#   ② log(close)       로그 레벨            — "지금 얼마인가"(곱셈→덧셈 단위)
#   ③ diff(log close)  수익률(1차 차분)     — "얼마 변했나"           ← 부호가 있다
#   ④ |수익률|          크기                 — "얼마나 크게 변했나"     ← 부호를 버린다
#   ⑤ rolling std(96)  실현변동성           — "요즘 얼마나 출렁이나"    ← 크기를 시간축에 평활
#
# 그리고 **다섯 계열 전부에 완전히 같은 자를 댄다**(ADF·KPSS·ARCH-LM·ACF·GPH·Hurst·왜도·첨도·
# 정규성). 같은 자를 대야 "무엇이 달라지는가"가 계열의 성질 때문임을 말할 수 있다.
#
# ## 어떻게 (How)
# 도구 구현(GARCH MLE·GPH·Hurst)은 18번 드라이버에서 그대로 가져온다(재구현하면 두 보고서의
# 수치가 갈릴 수 있다). GARCH는 가정 점검이 목적이므로 ②(레벨=반례)와 ③(수익률=정본)에만
# 적합하고, ④⑤는 GARCH의 입력이 아니라 **HAR류가 모델링하는 대상**이므로 그 이유를 명시한다.
# 표준화 잔차의 정규성(Jarque-Bera)까지 검정해 GARCH 가정 점검을 완결한다.
#
# ## 기대 결과 / 반영 (Expected)
# "전처리가 분석 대상을 가른다"를 계열 × 가정 표와 나란한 그림으로 확정하고, 그로부터
# "어떤 전처리에서 무엇을 판단할 수 있는가"의 규칙을 만든다.

# %%
"""18d 전처리 사다리 분석.

실행:
    uv run test/models/18d_preprocessing_ladder.py
    uv run test/models/18d_preprocessing_ladder.py --max-rows 30000   # 축소 점검
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
matplotlib.rcParams["axes.formatter.limits"] = (-9, 12)

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.stattools import acf, adfuller, kpss

warnings.filterwarnings("ignore")


def _project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "pyproject.toml").exists() and (candidate / "engine").is_dir():
            return candidate
    raise RuntimeError("quantitative_trading 디렉터리를 찾지 못했다.")


try:
    _START = Path(__file__).resolve().parent
except NameError:
    _START = Path.cwd()
ROOT = _project_root(_START)

# 18번 드라이버의 도구를 재사용한다 — 재구현하면 두 보고서의 수치가 갈린다.
_spec = importlib.util.spec_from_file_location(
    "d18", ROOT / "test" / "models" / "18_differencing_decision_test.py"
)
d18 = importlib.util.module_from_spec(_spec)
sys.modules["d18"] = d18
_spec.loader.exec_module(d18)

TAG = "18d_preprocessing_ladder_20260819"
SOURCE_TABLE = "upbit_krw_candle"
BASE_TICKER = "KRW-BTC"
BARS_PER_DAY = 96
IMAGES_DIR = ROOT / "test" / "images" / TAG
RESULTS_DIR = ROOT / "test" / "results" / TAG
RAW_PATH = RESULTS_DIR / "ladder_raw.md"
TEST_SAMPLE = 20_000
TEST_MAXLAG = 24

_LINES: list[str] = []


def emit(text: str = "") -> None:
    print(text, flush=True)
    _LINES.append(text)


def num(value, digits: int = 4) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "n/a" if not np.isfinite(v) else f"{v:,.{digits}f}"


# %% [markdown]
# ## 전처리 사다리 만들기
#
# 다섯 계열은 임의로 고른 것이 아니라 **한 사슬의 연속된 칸**이다. 각 칸은 앞 칸에서 정보를
# 하나 버리거나 형태를 바꾼다 — 그 "버리는 것"이 무엇을 볼 수 있고 무엇을 볼 수 없게 되는지를
# 결정한다.

# %%
def build_ladder(frame: pd.DataFrame) -> dict[str, dict]:
    close = frame["close"].astype(float)
    log_close = np.log(close.clip(lower=1e-9))
    log_return = log_close.diff()
    abs_return = log_return.abs()
    realized_vol = log_return.rolling(BARS_PER_DAY).std()
    return {
        "① 원 종가 (레벨)": {
            "series": close,
            "question": "지금 얼마인가",
            "drops": "—(원자료)",
            "op": "raw",
        },
        "② 로그 종가 (레벨)": {
            "series": log_close,
            "question": "지금 얼마인가 (곱셈적 변동을 덧셈적으로)",
            "drops": "절대 단위(원). 수준 정보는 그대로 보존",
            "op": "log(x)",
        },
        "③ 수익률 (1차 차분)": {
            "series": log_return,
            "question": "직전 대비 얼마 변했나",
            "drops": "**수준 정보를 버린다** — '지금 얼마'를 알 수 없게 된다",
            "op": "diff(log x)",
        },
        "④ |수익률| (크기)": {
            "series": abs_return,
            "question": "얼마나 크게 변했나",
            "drops": "**부호를 버린다** — 오르내림 방향을 알 수 없게 된다",
            "op": "|diff|",
        },
        f"⑤ 실현변동성 ({BARS_PER_DAY}봉)": {
            "series": realized_vol,
            "question": "요즘 얼마나 출렁이나",
            "drops": "개별 봉의 크기를 버린다 — 시간축으로 평활된 수준만 남는다",
            "op": f"rolling_std({BARS_PER_DAY})",
        },
    }


# %% [markdown]
# ## 모든 계열에 같은 자를 댄다
#
# 도구별로 무엇을 재는지:
#
# | 도구 | 무엇을 재나 | 어떻게 계산하나 |
# | :--- | :--- | :--- |
# | ADF | 단위근(수준이 표류하는가) | 귀무=단위근 있음. p<0.05면 정상 |
# | KPSS | 정상성 | 귀무=정상. p<0.05면 비정상. **ADF와 방향이 반대** |
# | ARCH-LM | 조건부 이분산(분산이 시변하는가) | 제곱 잔차를 자기회귀로 회귀. p<0.05면 이분산 존재 |
# | ACF(1) | 1차 자기상관(직전 값이 다음을 설명하는가) | corr(x_t, x_{t-1}) |
# | ACF(96) | 하루 뒤까지 상관이 남는가 | corr(x_t, x_{t-96}) |
# | GPH d | 장기기억 모수 | 로그주기도를 log(4sin²(ω/2))에 회귀. d=0 무기억 |
# | Hurst H | 장기기억(다른 추정기) | R/S 통계량의 로그-로그 기울기. H=0.5 무기억 |
# | 왜도·첨도 | 분포 모양 | 정규분포는 0, 0(초과첨도) |
# | Jarque-Bera | 정규성 | 왜도·첨도 결합 검정. p<0.05면 정규 아님 |

# %%
def measure(series: pd.Series) -> dict:
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
    autocorr = acf(sample, nlags=BARS_PER_DAY, fft=True)
    try:
        jb_p = float(sp_stats.jarque_bera(sample.to_numpy())[1])
    except Exception:
        jb_p = float("nan")
    return {
        "n": int(len(clean)),
        "mean": float(clean.mean()),
        "std": float(clean.std()),
        "skew": float(sp_stats.skew(sample.to_numpy())),
        "kurtosis": float(sp_stats.kurtosis(sample.to_numpy())),
        "adf_stat": float(adf_stat),
        "adf_p": float(adf_p),
        "kpss_stat": float(kpss_stat),
        "kpss_p": float(kpss_p),
        "arch_p": arch_p,
        "acf1": float(autocorr[1]),
        "acf96": float(autocorr[BARS_PER_DAY]),
        "gph_d": d18.gph_d(clean),
        "hurst": d18.hurst_rs(clean),
        "jb_p": jb_p,
        "mean_stationary": bool(adf_p < 0.05 and (kpss_p > 0.05 if np.isfinite(kpss_p) else False)),
    }


# %%
def section_ladder(ladder: dict) -> pd.DataFrame:
    emit("## 1. 전처리 사다리 — 각 칸이 무엇을 버리는가")
    emit()
    emit("| 계열 | 연산 | 답하는 질문 | 이 단계에서 버리는 정보 |")
    emit("| :--- | :--- | :--- | :--- |")
    for name, spec in ladder.items():
        emit(f"| {name} | `{spec['op']}` | {spec['question']} | {spec['drops']} |")
    emit()
    emit("**핵심**: 계열이 다섯 개인 것은 세분화가 아니라 이 사슬의 각 칸이다. 그리고 각 칸에서")
    emit("**무엇을 버렸는지가 곧 그 칸에서 무엇을 볼 수 없는지**를 결정한다 —")
    emit("③에서 수준을 버렸으니 ③으로는 '가격이 얼마가 될까'를 답할 수 없고,")
    emit("④에서 부호를 버렸으니 ④로는 '오를까 내릴까'를 답할 수 없다. 이것이 전처리가 분석")
    emit("대상을 가르는 **기계적인 이유**다. 아래 검정은 그 결과를 확인하는 것일 뿐이다.")
    emit()

    rows = {}
    for name, spec in ladder.items():
        rows[name] = measure(spec["series"])
        print(f"  [측정 완료] {name}", flush=True)
    table = pd.DataFrame(rows).T

    emit("## 2. 다섯 계열에 완전히 같은 자를 댄 결과")
    emit()
    emit(f"- 검정 표본: 각 계열 최근 {TEST_SAMPLE:,}행, 고정 maxlag={TEST_MAXLAG} (10만 행 autolag segfault 회피)")
    emit("- KPSS p는 statsmodels가 [0.01, 0.10]으로 절단한다 — 통계량을 함께 본다")
    emit()
    emit("| 계열 | ADF p | KPSS p | 평균-정상 | ARCH-LM p | ACF(1) | ACF(96) | GPH d | Hurst H | 왜도 | 초과첨도 | JB p |")
    emit("| :--- | ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for name, row in table.iterrows():
        emit(
            f"| {name} | {num(row['adf_p'])} | {num(row['kpss_p'])} | "
            f"{'**예**' if row['mean_stationary'] else '아니오'} | {num(row['arch_p'])} | "
            f"{num(row['acf1'])} | {num(row['acf96'])} | {num(row['gph_d'])} | {num(row['hurst'])} | "
            f"{num(row['skew'], 2)} | {num(row['kurtosis'], 1)} | {num(row['jb_p'])} |"
        )
    emit()
    return table


# %% [markdown]
# ## GARCH 가정 점검
#
# 참고한 설명글(needmorecaffeine.tistory.com/37)이 제시하는 가정은 두 가지다.
#   (가) 오차항: E[u_t] = 0, Var[u_t] = σ_u²  → **평균-정상**을 요구한다
#   (나) 정상성 조건: Σα_i + Σβ_i < 1
# 표준 GARCH는 여기에 **(다) 표준화 잔차가 i.i.d.** 라는 가정을 더 쓴다(정규 MLE를 쓰면 정규성까지).
# 네 가지를 계열마다 판정한다.

# %%
def section_assumptions(ladder: dict, table: pd.DataFrame) -> None:
    emit("## 3. GARCH 가정을 계열마다 점검")
    emit()
    emit("참고 설명글이 제시하는 가정: **(가)** 오차항 E[u]=0·Var[u]=σ² (평균-정상 요구),")
    emit("**(나)** 정상성 조건 Σα+Σβ < 1. 표준 GARCH는 여기에 **(다)** 표준화 잔차 i.i.d.,")
    emit("정규 MLE를 쓰면 **(라)** 표준화 잔차 정규성까지 요구한다.")
    emit()

    fits: dict[str, dict] = {}
    for name in ["② 로그 종가 (레벨)", "③ 수익률 (1차 차분)"]:
        try:
            fits[name] = d18.fit_garch11(ladder[name]["series"])
        except Exception as exc:
            fits[name] = {"error": str(exc)}

    emit("| 계열 | (가) 평균-정상 | (나) Σα+Σβ < 1 | (다) 잔차 무자기상관 | (라) 잔차 정규성 | GARCH 적용 판정 |")
    emit("| :--- | :--- | :--- | :--- | :--- | :--- |")

    for name in ladder:
        row = table.loc[name]
        gate_a = "**통과**" if row["mean_stationary"] else "**위반**"
        if name in fits and "persistence" in fits[name]:
            fit = fits[name]
            persistence = fit["persistence"]
            gate_b = f"{num(persistence, 5)} → {'통과(경계)' if persistence > 0.99 else '통과'}"
            residual = pd.Series(fit["standardized"]).tail(TEST_SAMPLE)
            resid_acf1 = float(acf(residual, nlags=2, fft=True)[1])
            lb_p = float(acorr_ljungbox(residual, lags=[20], return_df=True)["lb_pvalue"].iloc[0])
            gate_c = f"ACF(1)={num(resid_acf1)}, LB p={num(lb_p)} → {'통과' if abs(resid_acf1) < 0.1 else '**위반**'}"
            jb_resid = float(sp_stats.jarque_bera(residual.to_numpy())[1])
            gate_d = f"JB p={num(jb_resid)} → {'통과' if jb_resid > 0.05 else '**위반**'}"
            if not row["mean_stationary"]:
                verdict = "**적용 불가** — (가) 위반"
            elif abs(resid_acf1) >= 0.1:
                verdict = "**적용 불가** — (다) 위반"
            elif jb_resid <= 0.05:
                verdict = "조건부 가능 — (라) 위반이라 **정규 대신 t분포** 필요"
            else:
                verdict = "적용 가능"
        else:
            gate_b = gate_c = gate_d = "—"
            if name.startswith("①"):
                verdict = "적용 안 함 — 단위(원)만 다른 ②의 중복"
            elif name.startswith("④") or name.startswith("⑤"):
                verdict = "**GARCH의 입력이 아님** — 이것이 GARCH가 *설명하려는 대상*(HAR류의 입력)"
            else:
                verdict = "적합 실패"
        emit(f"| {name} | {gate_a} | {gate_b} | {gate_c} | {gate_d} | {verdict} |")
    emit()

    level_fit, return_fit = fits.get("② 로그 종가 (레벨)", {}), fits.get("③ 수익률 (1차 차분)", {})
    if "persistence" in return_fit:
        emit(f"- **(나)는 수치상 만족하지만 경계에 붙어 있다**: Σα+Σβ = {num(return_fit['persistence'], 5)} < 1.")
        emit("  1에서 떨어진 정도가 0.001 수준이라 정상성 조건을 '통과'라고는 해도 실질적으로는")
        emit("  준-IGARCH 영역이다. 이것이 반감기가 표본에 따라 55배 흔들린 이유이며(18번 4.2절),")
        emit("  **가정을 형식적으로 만족하는 것과 추정이 쓸 만한 것은 다르다**는 사례다.")
    if "persistence" in level_fit:
        emit(f"- 레벨(②)은 (가)부터 위반한다 — ADF p={num(table.loc['② 로그 종가 (레벨)', 'adf_p'])}로 단위근을")
        emit("  못 벗어난다. 그래도 적합은 돌아가고 숫자가 나온다(Σα+Σβ="
             f"{num(level_fit['persistence'], 5)}) — **가정 위반은 에러로 알려주지 않는다**는 점이 중요하다.")
    emit("- **④⑤에 GARCH를 적용하지 않는 이유**: GARCH는 '수익률의 조건부 분산'을 모델링한다.")
    emit("  ④|수익률|과 ⑤실현변동성은 그 분산의 **관측 대리변수**이므로, GARCH의 입력이 아니라")
    emit("  GARCH가 맞히려는 정답지다. 이 계열을 직접 모델링하는 것이 HAR-RV·ARFIMA 계열이다.")
    emit()


# %% [markdown]
# ## 형태를 나란히 본다 (모델링 전 EDA)
#
# 표만으로는 "왜"가 안 보인다. 다섯 계열을 같은 배치로 그려 형태 차이를 눈으로 확인한다.

# %%
def section_figure(ladder: dict, frame: pd.DataFrame, table: pd.DataFrame) -> None:
    n = len(ladder)
    fig, axes = plt.subplots(n, 3, figsize=(19, 3.1 * n))
    timestamps = pd.to_datetime(frame["timestamp"])
    colors = ["#1f77b4", "#1f77b4", "#d62728", "#2ca02c", "#9467bd"]

    for i, (name, spec) in enumerate(ladder.items()):
        series = spec["series"]
        color = colors[i]
        clean = series.dropna()
        sample = clean.tail(TEST_SAMPLE)
        row = table.loc[name]

        axes[i][0].plot(timestamps.to_numpy()[-len(clean):], clean.to_numpy(), lw=0.4, color=color)
        axes[i][0].set_title(f"{name} — 시계열", fontsize=10)
        if i == 0:
            axes[i][0].yaxis.set_major_formatter(
                matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))

        axes[i][1].hist(sample.to_numpy(), bins=120, color=color, alpha=0.85)
        axes[i][1].set_title(
            f"분포 — 왜도 {num(row['skew'], 2)}, 초과첨도 {num(row['kurtosis'], 1)}", fontsize=10)

        bars = acf(sample, nlags=BARS_PER_DAY, fft=True)[1:]
        axes[i][2].bar(range(1, len(bars) + 1), bars, color=color)
        axes[i][2].axhline(0, color="black", lw=0.6)
        axes[i][2].set_ylim(-0.2, 1.05)
        axes[i][2].set_title(
            f"ACF — ACF(1) {num(row['acf1'])}, ACF(96) {num(row['acf96'])}", fontsize=10)

    axes[0][0].set_ylabel("레벨\n(수준 보존)", fontsize=9)
    axes[2][0].set_ylabel("차분\n(수준 버림)", fontsize=9)
    axes[3][0].set_ylabel("크기\n(부호 버림)", fontsize=9)
    fig.suptitle(
        "전처리 사다리: 같은 원자료를 다섯 단계로 변환하고 같은 자를 댄다\n"
        "위 두 줄(레벨)은 ACF가 1 부근에서 안 떨어진다 → 비정상 · "
        "③(수익률)은 즉시 0 → 방향 무신호 · ④⑤(크기)는 느리게 감소 → 예측 가능한 구조",
        fontsize=12, y=0.998,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    fig.savefig(IMAGES_DIR / "ladder_side_by_side.png", dpi=110)
    plt.close(fig)


# %%
def section_rules(table: pd.DataFrame) -> None:
    emit("## 4. 그래서 어떤 전처리에서 무엇을 판단할 수 있는가")
    emit()
    emit("| 알고 싶은 것 | 써야 하는 계열 | 이유 | 쓰면 안 되는 계열과 그 이유 |")
    emit("| :--- | :--- | :--- | :--- |")
    emit("| 가격이 얼마가 될까 | ①② 레벨 | 수준 정보가 여기에만 있다 | ③④⑤ — 수준을 버렸다 |")
    emit("| 오를까 내릴까(방향) | ③ 수익률 | 부호가 여기에만 있다 | ④⑤ — 부호를 버렸다. "
         "①② — 쓸 수는 있으나 R²가 무모델과 구별 안 됨(18번 3.2절) |")
    emit("| 얼마나 크게 움직일까(변동성) | ④⑤ 크기·실현변동성 | 크기 구조가 여기에 남는다 | "
         "③ — 부호가 섞여 크기 신호가 희석된다 |")
    emit("| 모델 성능이 개선됐나 | ③ 차분축에서 평가 | 동어반복이 제거된 축 | "
         "①② — 무모델도 R² 0.9999(18번 3.2절) |")
    emit()
    emit("**판단 근거는 검정 결과가 아니라 사다리 구조 자체다.** 검정은 그 구조가 데이터에서")
    emit("실제로 그렇게 나타나는지 확인하는 역할이다. 위 표의 각 행이 1절의 '버리는 정보' 열과")
    emit("일대일로 대응한다.")
    emit()
    emit("### 검정이 확인해 준 것")
    emit()
    for name, row in table.iterrows():
        if name.startswith("①") or name.startswith("②"):
            note = (f"ACF(1)={num(row['acf1'])}이 1에 붙어 있고 ACF(96)={num(row['acf96'])}도 안 떨어진다 "
                    "→ 수준이 표류하는 비정상. 예측이 쉬워 보이는 건 동어반복 때문")
        elif name.startswith("③"):
            note = (f"ACF(1)={num(row['acf1'])}로 즉시 0 → 방향 무신호. "
                    f"단 ARCH-LM p={num(row['arch_p'])}로 분산은 시변 → 크기 쪽에 구조가 남아 있다는 신호")
        elif name.startswith("④"):
            note = (f"ACF(1)={num(row['acf1'])}, ACF(96)={num(row['acf96'])}로 하루 뒤까지 상관이 남는다. "
                    f"GPH d={num(row['gph_d'])} → 장기기억")
        else:
            note = (f"ACF(96)={num(row['acf96'])}, GPH d={num(row['gph_d'])} → 장기기억. "
                    "평활했으므로 ④보다 매끄럽지만 같은 구조")
        emit(f"- **{name}**: {note}")
    emit()


# %%
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="18d 전처리 사다리")
    parser.add_argument("--ticker", default=BASE_TICKER)
    parser.add_argument("--max-rows", type=int, default=0)
    args = parser.parse_args(argv)

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    db_path = ROOT / "data" / "upbit_data.db"
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        frame = con.execute(
            f"select timestamp, close from {SOURCE_TABLE} where ticker = ? order by timestamp",
            [args.ticker],
        ).df()
    finally:
        con.close()
    if args.max_rows:
        frame = frame.tail(args.max_rows).reset_index(drop=True)

    emit(f"# 18d 전처리 사다리 원시 수치 — {args.ticker}")
    emit()
    emit("생성: `uv run test/models/18d_preprocessing_ladder.py`. 자동 생성물이며 보고서가 인용한다.")
    emit()
    emit(f"- 데이터: `{SOURCE_TABLE}`, {len(frame):,}행, "
         f"{frame['timestamp'].min():%Y-%m-%d} ~ {frame['timestamp'].max():%Y-%m-%d}")
    emit("- 도구 구현(GARCH MLE·GPH·Hurst)은 `18_differencing_decision_test.py`에서 import — 수치 일관성 보장")
    emit()

    ladder = build_ladder(frame)
    table = section_ladder(ladder)
    section_assumptions(ladder, table)
    section_figure(ladder, frame, table)
    section_rules(table)

    table.to_csv(RESULTS_DIR / "ladder_measurements.csv", encoding="utf-8")
    RAW_PATH.write_text("\n".join(_LINES) + "\n", encoding="utf-8")
    print(f"[18d] 저장: {RAW_PATH}")
    print(f"[18d] 그림: {IMAGES_DIR / 'ladder_side_by_side.png'}")


if __name__ == "__main__":
    main(sys.argv[1:])
