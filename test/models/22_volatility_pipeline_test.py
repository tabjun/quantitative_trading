# %% [markdown]
# # 22번: 변동성 예측 파이프라인 — 처리마다 EDA를 붙인 전면 비교
#
# 헤드리스 `.py` 드라이버(`#%%` 셀 + 파일 내 마크다운, AGENTS.md 2.3).
#
# ## 왜 (Why)
# 그동안 통계 검정 수치만 뽑고 **눈으로 보는 EDA를 건너뛰었다**(사용자 지적 2026-09-08).
# 기초통계량 파악은 거시 plot을 보는 것에서 시작해야 하며, **처리를 할 때마다 데이터가
# 어떻게 변하는지 매번 확인**해야 방향이 어긋나지 않는다.
#
# ## 무엇을 (What) — 파이프라인 각 단계마다 EDA
# ```
# [E0] 원본 EDA          거시 plot · 기초통계 · 분포
#   ↓ [P1] 로그+1차차분
# [E1] 변환 후 EDA       정상성 확보 확인 · 분포 변화
#   ↓ [P2] 주기 제거(전/후 두 조건)
# [E2] 주기 제거 EDA     주기가 실제로 사라졌는가
#   ↓ [P3] 모델군별 전처리
# [E3] 입력 분포 EDA     모델군마다 다른 입력이 어떻게 생겼는가
# ```
#
# ## 어떻게 (How) — 모델군별 가정 맞춤
# | 모델군 | 가정 | 전처리 |
# | :--- | :--- | :--- |
# | GARCH 계열 | 정상성·조건부이분산 | 원 수익률(표준화 금지) + t/skew-t 분포 |
# | HAR-RV | 선형성·등분산 | 로그 실현변동성 |
# | 트리 | 없음(비모수) | 원 스케일 + 시간구조 feature 주입 |
# | 딥러닝 | 스케일 민감·분포이동 | RevIN(시퀀스별 정규화) |
#
# ## 기대 결과
# 주기 제거 전/후 두 조건에서 모델 전면 비교. 꼬리(두꺼움)는 t분포로 이론적 보완만 하고,
# 완전한 처리는 후속 연구로 넘긴다(교수님 협의 2026-09-07).

# %%
"""22번 변동성 파이프라인 드라이버 — 1부: 처리 단계별 EDA.

실행:
    uv run test/models/22_volatility_pipeline_test.py --stage eda
    uv run test/models/22_volatility_pipeline_test.py --stage eda --quick
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

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as st
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
sys.path.insert(0, str(ROOT / "test" / "scripts"))
from report_header import render_standard_header, study_universe, num  # noqa: E402

EXPERIMENT_TAG = "22_volatility_pipeline_20260908"
DB = ROOT / "data" / "upbit_data.db"
IMAGES_DIR = ROOT / "test" / "images" / EXPERIMENT_TAG
RESULTS_DIR = ROOT / "test" / "results" / EXPERIMENT_TAG
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
BPD = 96  # 15분봉 하루
ACCENT, ACCENT2, MUTED = "#0E9384", "#C85A3E", "#8FA1A8"
_LINES: list[str] = []


def emit(t: str = "") -> None:
    print(t, flush=True)
    _LINES.append(t)


# %% [markdown]
# ## 데이터 적재

# %%
def load_prices(tickers: list[str]) -> pd.DataFrame:
    con = duckdb.connect(str(DB), read_only=True)
    q = f"""SELECT ticker, timestamp, close FROM upbit_krw_candle
            WHERE ticker IN ({','.join(['?'] * len(tickers))}) ORDER BY timestamp"""
    df = con.execute(q, tickers).df()
    con.close()
    return df.pivot(index="timestamp", columns="ticker", values="close")[tickers]


def diagnose(x: np.ndarray, label: str) -> dict:
    """한 계열의 기초통계 + 핵심 검정을 한 번에."""
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    sub = x[-40000:] if len(x) > 40000 else x
    d = {"단계": label, "n": len(x), "평균": x.mean(), "표준편차": x.std(),
         "왜도": float(st.skew(x)), "초과첨도": float(st.kurtosis(x)),
         "최소": x.min(), "최대": x.max()}
    for key, fn in (("ADF p", lambda: adfuller(sub, autolag="AIC")[1]),
                    ("KPSS p", lambda: kpss(sub, regression="c", nlags="auto")[1]),
                    ("ARCH p", lambda: het_arch(sub - sub.mean(), nlags=16)[1]),
                    ("LB(|x|) p", lambda: acorr_ljungbox(np.abs(sub), lags=[BPD],
                                                          return_df=True)["lb_pvalue"].iloc[0])):
        try:
            d[key] = float(fn())
        except Exception:
            d[key] = np.nan
    return d


def acf(x: np.ndarray, lag: int) -> float:
    x = x - x.mean()
    den = np.sum(x * x)
    return float(np.sum(x[lag:] * x[:-lag]) / den) if den > 1e-18 else np.nan


def seasonal_profile(r: pd.Series) -> tuple[pd.Series, pd.Series]:
    """시간대별·요일별 평균 |수익률| — 주기 성분의 모양."""
    a = r.abs()
    return a.groupby(a.index.hour).mean(), a.groupby(a.index.dayofweek).mean()


def remove_periodicity(r: pd.Series) -> pd.Series:
    """주기 제거: 각 (요일,시각) 슬롯의 평균 |수익률|로 나눠 표준화.

    목적은 STL 같은 분해가 아니라, **시계만 보면 알 수 있는 부분**을 덜어내는 것이다.
    9시는 원래 변동성이 높고 4시는 낮은데, 그대로 두면 모델이 그 '쉬운 것'만 학습해
    성능이 부풀고 GARCH 지속성이 왜곡된다(19번에서 반감기 54배 왜곡 확인).
    """
    slot = r.index.dayofweek * 24 + r.index.hour
    a = r.abs()
    scale = a.groupby(slot).transform("mean")
    overall = a.mean()
    out = r / scale.replace(0, np.nan) * overall
    return out.fillna(0.0)


# %% [markdown]
# ## E0~E2 — 원본 → 차분 → 주기 제거, 매 단계 EDA

# %%
def stage_eda(tickers: list[str], rep: str, quick: bool) -> None:
    px = load_prices(tickers).dropna(how="all")
    emit(f"## E0. 원본 데이터 — 거시 흐름부터 본다")
    emit()
    emit(f"- 종목 {len(tickers)}개 · 15분봉 · {px.index[0]} ~ {px.index[-1]} · {len(px):,}행")
    emit()

    # 각 종목 자기 첫 관측을 100으로(상장 시점이 달라 bfill로 채우면 기준이 뒤틀린다)
    first = px.apply(lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan)
    norm = px / first * 100
    med = norm.median(axis=1)
    emit(f"- 정규화 가격(각 종목 첫 관측=100) 중앙값: 고점 {num(med.max(),1)} "
         f"({med.idxmax()}) → 최종 {num(med.ffill().iloc[-1],1)}")
    last = norm.ffill().iloc[-1].dropna()
    emit(f"- 종목별 최종값 범위 {num(last.min(),1)} ~ {num(last.max(),1)} "
         f"(중앙값 {num(last.median(),1)}) — 편차가 크다")
    emit()

    # 대표 종목으로 처리 단계 추적
    close = px[rep].dropna()
    r0 = np.log(close).diff().dropna()          # P1: 로그 + 1차 차분
    r1 = remove_periodicity(r0)                  # P2: 주기 제거

    stages = {"P0 로그가격(레벨)": np.log(close).to_numpy(),
              "P1 로그수익률(차분)": r0.to_numpy(),
              "P2 +주기제거": r1.to_numpy()}
    rows = [diagnose(v, k) for k, v in stages.items()]
    dd = pd.DataFrame(rows)

    emit("## E1. 처리 단계별 기초통계량과 검정 — 무엇이 어떻게 바뀌는가")
    emit()
    emit(f"대표 종목 **{rep}** 기준. 각 처리가 데이터를 어떻게 바꾸는지 매 단계 확인한다.")
    emit()
    emit("| 단계 | n | 표준편차 | 왜도 | 초과첨도 | ADF p | KPSS p | ARCH p | LB(&#124;x&#124;) p |")
    emit("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for _, x in dd.iterrows():
        emit(f"| {x['단계']} | {int(x['n']):,} | {num(x['표준편차'],6)} | {num(x['왜도'],3)} | "
             f"**{num(x['초과첨도'],1)}** | {num(x['ADF p'],4)} | {num(x['KPSS p'],4)} | "
             f"{num(x['ARCH p'],4)} | {num(x['LB(|x|) p'],4)} |")
    emit()
    emit("**읽는 법**")
    emit("- ADF p<0.05 = 정상 / KPSS p>0.05 = 정상 → **P0에서 P1로 갈 때 정상성이 확보**되는지 본다.")
    emit("- ARCH p<0.05 = 조건부 이분산 존재 → 변동성 모델을 쓸 근거.")
    emit("- LB(|x|) p<0.05 = 변동성에 자기상관 존재(변동성 군집).")
    emit()

    # 주기 제거 효과
    hod0, dow0 = seasonal_profile(r0)
    hod1, dow1 = seasonal_profile(pd.Series(r1.values, index=r0.index))
    emit("## E2. 주기 제거 효과 — 실제로 주기가 사라졌는가")
    emit()
    emit("| 구분 | 시간대 최대/최소 | 요일 최대/최소 |")
    emit("| :--- | ---: | ---: |")
    emit(f"| 제거 전 | **{num(hod0.max()/hod0.min(),2)}배** | {num(dow0.max()/dow0.min(),2)}배 |")
    emit(f"| 제거 후 | **{num(hod1.max()/hod1.min(),2)}배** | {num(dow1.max()/dow1.min(),2)}배 |")
    emit()
    emit("주기 제거는 STL 같은 분해가 아니라 **시계만 보면 알 수 있는 부분을 덜어내는 것**이다. "
         "9시는 원래 변동성이 높고 4시는 낮은데, 그대로 두면 모델이 그 '쉬운 것'만 학습해 성능이 "
         "부풀고 GARCH 지속성이 왜곡된다(19번에서 반감기 54배 왜곡 확인).")
    emit()

    # ── 그림 1: 처리 단계 추적 ──
    fig, ax = plt.subplots(3, 3, figsize=(18, 12))
    for i, (name, v) in enumerate(stages.items()):
        idx = close.index if i == 0 else r0.index
        vv = v[-len(idx):] if len(v) >= len(idx) else v
        ax[i, 0].plot(idx[-len(vv):], vv, lw=0.4, color=ACCENT if i else MUTED)
        ax[i, 0].set_title(f"({chr(65+i)}-1) {name} — 시계열", fontsize=11)
        ax[i, 0].grid(alpha=.2)

        lim = np.quantile(np.abs(vv[np.isfinite(vv)]), 0.999)
        sel = vv[(vv > -lim) & (vv < lim)] if i else vv
        ax[i, 1].hist(sel, bins=150, density=True, color=ACCENT, alpha=.6)
        if i:
            xs = np.linspace(-lim, lim, 400)
            ax[i, 1].plot(xs, st.norm.pdf(xs, np.mean(vv), np.std(vv)), "r--", lw=1.6,
                          label="정규분포")
            ax[i, 1].legend(fontsize=8)
        k = st.kurtosis(vv[np.isfinite(vv)])
        ax[i, 1].set_title(f"({chr(65+i)}-2) 분포 (초과첨도 {k:.1f})", fontsize=11)

        if i:
            lags = list(range(1, 97))
            ax[i, 2].plot(lags, [acf(np.abs(vv), L) for L in lags], color=ACCENT2, lw=1.5,
                          label="|x| 자기상관")
            ax[i, 2].plot(lags, [acf(vv, L) for L in lags], color=MUTED, lw=1.2, label="x 자기상관")
            ax[i, 2].axhline(0, color="black", lw=.7)
            ax[i, 2].legend(fontsize=8)
            ax[i, 2].set_title(f"({chr(65+i)}-3) 자기상관 (lag 1~96)", fontsize=11)
        else:
            ax[i, 2].axis("off")
        ax[i, 2].grid(alpha=.2)
    fig.suptitle(f"그림 1. 처리 단계별 데이터 변화 — {rep}", fontsize=15, y=1.00)
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "fig1_pipeline_stages.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    emit(f"![그림 1](../../images/{EXPERIMENT_TAG}/fig1_pipeline_stages.png)")
    emit()

    # ── 그림 2: 거시 흐름 + 주기 제거 ──
    fig2, ax2 = plt.subplots(2, 2, figsize=(16, 9))
    b = ax2[0, 0]
    for t in norm.columns:
        b.plot(norm.index, norm[t], lw=0.5, color=MUTED, alpha=.4)
    b.plot(med.index, med, color=ACCENT, lw=2.5, label="중앙값")
    b.set_yscale("log"); b.axhline(100, color="black", ls=":", lw=.8)
    b.legend(fontsize=9); b.grid(alpha=.2)
    b.set_title("(1-A) 거시 흐름 — 정규화 가격(로그축)\n회색=개별 종목, 청록=중앙값")

    vol30 = np.log(px).diff().rolling(30 * BPD, min_periods=BPD).std() * np.sqrt(BPD * 365)
    b = ax2[0, 1]
    for t in vol30.columns:
        b.plot(vol30.index, vol30[t] * 100, lw=0.5, color=MUTED, alpha=.35)
    b.plot(vol30.index, vol30.median(axis=1) * 100, color=ACCENT2, lw=2.5, label="중앙값")
    b.legend(fontsize=9); b.grid(alpha=.2); b.set_ylabel("연율 변동성 (%)")
    b.set_title("(1-B) 30일 이동 변동성\n변동성 자체가 시간에 따라 크게 변한다")

    b = ax2[1, 0]
    w = 0.4
    xs = np.arange(24)
    b.bar(xs - w/2, hod0.values * 100, w, color=MUTED, label="제거 전")
    b.bar(xs + w/2, hod1.reindex(range(24)).values * 100, w, color=ACCENT, label="제거 후")
    b.set_xlabel("시각 (KST)"); b.set_ylabel("평균 |수익률| (%)")
    b.legend(fontsize=9)
    b.set_title(f"(1-C) 시간대 주기 제거 효과\n{num(hod0.max()/hod0.min(),2)}배 → "
                f"{num(hod1.max()/hod1.min(),2)}배")

    b = ax2[1, 1]
    lags = list(range(1, 97))
    b.plot(lags, [acf(np.abs(r0.to_numpy()), L) for L in lags], color=MUTED, lw=1.8, label="제거 전 |r|")
    b.plot(lags, [acf(np.abs(r1.to_numpy()), L) for L in lags], color=ACCENT, lw=1.8, label="제거 후 |r|")
    b.axhline(0, color="black", lw=.7); b.legend(fontsize=9); b.grid(alpha=.2)
    b.set_xlabel("lag (봉)")
    b.set_title("(1-D) 변동성 자기상관 — 주기 제거 전후\n제거 후에도 남는 것이 진짜 변동성 군집")
    fig2.suptitle("그림 2. 거시 구조와 주기 제거", fontsize=15, y=1.00)
    fig2.tight_layout()
    fig2.savefig(IMAGES_DIR / "fig2_macro_periodicity.png", dpi=120, bbox_inches="tight")
    plt.close(fig2)
    emit(f"![그림 2](../../images/{EXPERIMENT_TAG}/fig2_macro_periodicity.png)")
    emit()

    dd.to_csv(RESULTS_DIR / "stage_diagnostics.csv", index=False)


# %%
def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="eda", choices=["eda"])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--rep", default="KRW-BTC")
    a = ap.parse_args(argv)

    tickers, sel = study_universe()
    if a.quick:
        tickers = tickers[:5]

    emit("# 22번 — 변동성 예측 파이프라인: 처리마다 EDA")
    emit()
    emit(f"작성일 2026-09-08 · 브랜치 `stock` · 15분봉")
    emit()
    emit("> 그동안 통계 검정 수치만 뽑고 **눈으로 보는 EDA를 건너뛰었다**. 기초통계량 파악은 "
         "거시 plot에서 시작하며, **처리할 때마다 데이터가 어떻게 변하는지 매번 확인**한다.")
    emit()
    emit("---")
    emit()
    print("[header] 표준 헤더 생성 중...", flush=True)
    emit(render_standard_header(
        tickers=tickers, train_frac=0.70, rep_ticker=a.rep,
        analyzed_tickers=tickers, selection_reason=sel,
        transforms=[("로그 + 1차 차분", "정상성 확보(레벨은 단위근)"),
                    ("주기 제거(전/후 비교)", "시간대·요일 주기가 GARCH 지속성을 왜곡하고 "
                     "모델이 '쉬운 부분'만 학습해 성능이 부풀 수 있어 두 조건을 비교")]))
    emit("---")
    emit()

    stage_eda(tickers, a.rep, a.quick)

    (RESULTS_DIR / "pipeline_eda_raw.md").write_text("\n".join(_LINES) + "\n", encoding="utf-8")
    print(f"\n저장: {RESULTS_DIR / 'pipeline_eda_raw.md'}", flush=True)


if __name__ == "__main__":
    main()
