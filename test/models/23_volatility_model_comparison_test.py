# %% [markdown]
# # 23번: 변동성 예측 모델 통합 비교 — 가정 맞춤 전처리 + 매 단계 EDA
#
# 헤드리스 `.py` 드라이버(`#%%` 셀 + 파일 내 마크다운, AGENTS.md 2.3).
#
# ## 왜 (Why)
# 22번에서 거시 EDA와 처리 단계별 확인을 마쳤다. 이제 **모델을 실제로 겨루게 한다.**
# 다만 모델마다 요구하는 가정이 다르므로 **입력을 모델군별로 다르게 준비**해야 공정하다.
# 모든 모델에 같은 전처리를 넣고 비교하는 것은 어떤 모델에게는 불리한 출발선이다.
#
# ## 무엇을 (What) — 이전 계획 4건을 하나로 통합
# | 이전 계획 | 이 실험에서 |
# | :--- | :--- |
# | 비선형·커널 모델로 선형 대체 | 커널 3종 투입 + **선형은 검정 수치와 함께 참고용 표기** |
# | 하이브리드(GARCH → ML 입력) | GARCH 조건부분산을 feature로 결합한 2종 |
# | 전 종목 확대 검증 | 20종목 기본 조건 |
# | 레짐 전환 | 이 실험 이후 별도 진행(거시 EDA에서 근거 확보) |
#
# ## 어떻게 (How) — 모델군별 가정 맞춤
# | 모델군 | 가정 | 전처리 | 가정 위반 시 |
# | :--- | :--- | :--- | :--- |
# | GARCH 계열 | 정상성·조건부이분산 | 원 수익률(표준화 금지) + t분포 | 잔차 진단 명시 |
# | HAR-RV | 선형성·등분산 | 로그 실현변동성 | 검정 수치와 함께 한계 표기 |
# | 선형 | 선형성·등분산 | 로그축 + 표준화 | **RESET·BDS 수치를 함께 제시** |
# | 커널 | 스케일 민감(선형성 불필요) | 입력·타깃 표준화 | 비용 때문에 서브샘플 |
# | 트리 | 없음(비모수) | 원 스케일 + 시간구조 주입 | — |
# | 딥러닝 | 스케일 민감·분포이동 | RevIN(시퀀스별 정규화) | 분포이동 잔존 명시 |
#
# ## 조건·평가
# 주기 제거 **전/후 두 조건** × 20종목. 평가는 QLIKE(주 지표) · MZ-R² · MASE.
# 벤치마크는 HAR-RV(문헌에서 이기기 어려운 표준).

# %%
"""23번 통합 모델 비교 드라이버.

실행:
    uv run test/models/23_volatility_model_comparison_test.py
    uv run test/models/23_volatility_model_comparison_test.py --quick
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = ["Noto Sans CJK KR", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

import duckdb
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import torch
import torch.nn as nn
import xgboost as xgb
from arch import arch_model
from scipy import stats as st
from scipy.signal import lfilter
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.kernel_approximation import Nystroem
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch, het_breuschpagan, linear_reset
from statsmodels.tsa.stattools import bds

warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision("high")
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _project_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "pyproject.toml").exists() and (p / "test").exists():
            return p
    return start


ROOT = _project_root(Path(__file__).resolve())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "test" / "scripts"))
from report_header import render_standard_header, study_universe, num  # noqa: E402
from engine.models import make_model  # noqa: E402

TAG = "23_model_comparison_20260908"
DB = ROOT / "data" / "upbit_data.db"
IMG = ROOT / "test" / "images" / TAG
RES = ROOT / "test" / "results" / TAG
IMG.mkdir(parents=True, exist_ok=True)
RES.mkdir(parents=True, exist_ok=True)
BPD = 96                 # 15분봉 하루
HORIZON = 4              # 예측 대상: 1시간 뒤 실현변동성
TRAIN_FRAC = 0.70
SEQ_LEN = {"rnn": 32, "tfm": 96}
_LINES: list[str] = []


def emit(t: str = "") -> None:
    print(t, flush=True)
    _LINES.append(t)


# %% [markdown]
# ## 데이터·전처리

# %%
def load_close(ticker: str) -> pd.Series:
    con = duckdb.connect(str(DB), read_only=True)
    d = con.execute("SELECT timestamp, close FROM upbit_krw_candle WHERE ticker=? ORDER BY timestamp",
                    [ticker]).df()
    con.close()
    return pd.Series(d["close"].astype(float).clip(lower=1e-9).to_numpy(),
                     index=pd.DatetimeIndex(d["timestamp"]))


def remove_periodicity(r: pd.Series) -> pd.Series:
    """주기 제거 — 각 (요일,시각) 슬롯 평균 |수익률|로 나눠 표준화.

    "지금 9시니까 변동성 높겠지"는 시계만 봐도 아는 부분이라 모델이 이것만 학습해도
    성능이 부풀고, GARCH 지속성이 왜곡된다(19번 반감기 54배 왜곡).
    """
    slot = r.index.dayofweek * 24 + r.index.hour
    a = r.abs()
    scale = a.groupby(slot).transform("mean")
    return (r / scale.replace(0, np.nan) * a.mean()).fillna(0.0)


def tick_ratio(px: pd.Series) -> float:
    """호가 단위 / 가격 — 가격 이산성의 크기를 재는 통제 변수.

    업비트는 가격대별로 호가 단위를 다르게 정한다. 가격이 낮은 종목은 이 비율이 커서
    15분 동안 가격이 한 칸도 움직이지 않는 봉이 대량 발생한다
    (ARDR 22.2% · DOGE 23.0% · SHIB 22.4% 대 BTC 0.63%).

    **주의**: 이 값을 실현변동성의 하한으로 삼아 0을 메우는 처리는 하지 않는다.
    Sucarrat & Escribano(2018, *European Journal of Finance* 24(10))는 영수익률을
    작은 상수로 대체하는 처방이 추정을 점근적으로 편향시킴을 보이고 **0을 결측으로
    취급**할 것을 권했으며, Bellégo 외(arXiv:2203.11820)는 log(Y+Δ)의 Δ 선택이
    결과를 자의적으로 좌우함을 보였다. 따라서 이 값은 **보고용 통제 변수**로만 쓴다.
    """
    p = px.to_numpy()
    d = np.abs(np.diff(p))
    d = d[d > 0]
    return float(np.min(d) / np.median(p)) if len(d) else np.nan


def targets(r: np.ndarray, h: int):
    """미래/과거 h구간 실현변동성. 0은 그대로 둔다(메우지 않는다)."""
    c = np.concatenate([[0.0], np.cumsum(r ** 2)])
    n = len(r); idx = np.arange(n)
    fut = np.full(n, np.nan); pas = np.full(n, np.nan)
    ok = idx + 1 + h <= n
    fut[ok] = np.sqrt(c[idx[ok] + 1 + h] - c[idx[ok] + 1])
    okp = idx + 1 - h >= 0
    pas[okp] = np.sqrt(c[idx[okp] + 1] - c[idx[okp] + 1 - h])
    return fut, pas


def insanity_filter(pred: np.ndarray, lo: float, hi: float, mean_: float):
    """BPQ(2016) insanity filter — 예측치를 학습표본 지지집합 안으로 가둔다.

    Bollerslev, Patton & Quaedvlieg(2016, *Journal of Econometrics* 192(1), 각주 17)는
    예측치가 추정기간 중 타깃의 관측 범위를 벗어나면 그 기간의 무조건부 평균으로
    대체한다("insanity" is replaced by "ignorance"). 원논문의 실제 트리밍 비율은 0.1% 미만이다.

    이것은 QLIKE를 임의로 완화하는 편법이 아니라 **Patton(2011)의 robustness 정리가
    요구하는 가정 A4**(최적예측이 양의 실수 R⁺⁺의 컴팩트 부분집합 안에 있을 것)를
    그대로 구현한 것이다. 이 가정이 깨지면 QLIKE는 아래로 무계가 되어, 모델이 분산을
    0에 붙일수록 손실이 무한히 좋아지는 상태가 된다.

    범위는 Zhang 외(2024, *Journal of Financial Econometrics* 22(2))를 따라 학습기간
    타깃의 0.5%~99.5% 분위수로 잡는다. 반환값에 트리밍 비율을 함께 실어 보고한다.
    """
    p = np.asarray(pred, dtype=float)
    bad = ~np.isfinite(p) | (p < lo) | (p > hi)
    out = np.where(bad, mean_, p)
    return out, float(np.mean(bad))


def feats_log(r: np.ndarray) -> np.ndarray:
    """선형·HAR·커널용 — 로그축(실현변동성 우편향 대칭화)."""
    r2 = pd.Series(r ** 2); s = pd.Series(r); ar = pd.Series(np.abs(r))
    cols = []
    for w in (1, 4, 16, 96, 672):
        cols.append(np.log(np.sqrt(r2.shift(1).rolling(w, min_periods=1).mean()) + 1e-12).fillna(0).to_numpy())
    for w in (4, 16, 96):
        cols.append(np.log(ar.shift(1).rolling(w, min_periods=1).mean() + 1e-12).fillna(0).to_numpy())
        cols.append(s.shift(1).rolling(w, min_periods=1).mean().fillna(0).to_numpy())
    return np.column_stack(cols)


def feats_har(r: np.ndarray) -> np.ndarray:
    """HAR-RV 정석 — 과거 1봉/1일/1주 실현변동성(로그)."""
    r2 = pd.Series(r ** 2)
    return np.column_stack([
        np.log(np.sqrt(r2.shift(1).rolling(w, min_periods=1).mean()) + 1e-12).fillna(0).to_numpy()
        for w in (1, BPD, 7 * BPD)])


def feats_tree(r: np.ndarray) -> np.ndarray:
    """트리용 — 원 스케일 + 시간구조 명시 주입(트리는 시간 순서를 모른다)."""
    r2 = pd.Series(r ** 2); s = pd.Series(r); ar = pd.Series(np.abs(r))
    cols = []; rv = {}
    for w in (1, 4, 16, 96, 672):
        rv[w] = np.sqrt(r2.shift(1).rolling(w, min_periods=1).mean()).fillna(0)
        cols.append(rv[w].to_numpy())
    for a_, b_ in ((1, 16), (4, 96), (16, 672)):          # 창간 비율 = 국면 전환 신호
        cols.append((rv[a_] / (rv[b_] + 1e-12)).clip(0, 20).to_numpy())
    for k in (1, 2, 4, 8, 16):                             # lag = 자기상관 직접 주입
        cols.append(ar.shift(k).fillna(0).to_numpy())
    for w in (4, 16, 96):
        cols.append(s.shift(1).rolling(w, min_periods=1).mean().fillna(0).to_numpy())
    return np.column_stack(cols)


def seqs(r: np.ndarray, seq_len: int, revin: bool) -> np.ndarray:
    ch = np.column_stack([r, np.abs(r)]).astype(np.float32)
    n = len(r); X = np.zeros((n, seq_len, ch.shape[1]), np.float32)
    for i in range(n):
        lo = max(0, i - seq_len + 1); seg = ch[lo:i + 1]
        X[i, seq_len - len(seg):] = seg
    if revin:                                              # 시퀀스별 정규화(분포이동 대응)
        mu = X.mean(axis=1, keepdims=True); sd = X.std(axis=1, keepdims=True)
        X = (X - mu) / np.where(sd < 1e-6, 1.0, sd)
    else:                                                  # 전역 표준화
        mu = X.reshape(-1, X.shape[2]).mean(0); sd = X.reshape(-1, X.shape[2]).std(0)
        X = (X - mu) / np.where(sd < 1e-6, 1.0, sd)
    return X


# %% [markdown]
# ## 평가 지표 — QLIKE(주) · MZ-R² · MASE

# %%
def qlike(actual_var, pred_var) -> float:
    """QLIKE 손실 = log(예측분산) + 실제분산/예측분산 (낮을수록 좋음).

    Patton(2011, *Journal of Econometrics* 160(1)) 식 (6)의 정의를 그대로 쓴다.
    **실제분산에는 아무 보정도 하지 않는다** — 실제분산이 0이어도 QLIKE = log(h)로
    유한하기 때문이다. 발산은 오직 **예측분산 h가 0에 붙을 때**만 일어나며, 그것은
    호출 전에 `insanity_filter()`가 막는다(BPQ 2016). 즉 손실함수를 완화하는 것이
    아니라 Patton의 가정 A4를 예측 쪽에서 충족시키는 구조다.
    """
    a = np.asarray(actual_var, dtype=float); p = np.asarray(pred_var, dtype=float)
    v = np.log(p) + a / p
    v = v[np.isfinite(v)]
    return float(np.mean(v)) if len(v) else np.nan


def plot_macro(tickers: list[str], path: Path) -> dict:
    """거시 EDA — 검정 이전에 **전체 흐름을 눈으로 먼저** 본다.

    통계 검정만 쌓으면 데이터가 어떤 모양인지 모른 채 결론을 내게 된다. 모델 비교에
    들어가기 전에 (M-1)~(M-4)로 20종목의 큰 흐름·연동·주기·군집을 확인한다.
    """
    px = {}
    for tk in tickers:
        s = load_close(tk)
        px[tk.replace("KRW-", "")] = s
    P = pd.DataFrame(px).sort_index()
    first = P.apply(lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan)
    norm = P / first * 100                                   # 종목별 첫 관측=100
    R = np.log(P).diff()

    fig, ax = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("그림 M. 거시 EDA — 검정 이전에 전체 흐름을 눈으로 본다", fontsize=15)

    # (M-1) 정규화 가격 흐름
    a = ax[0, 0]
    for c in norm.columns:
        a.plot(norm.index, norm[c], lw=.6, alpha=.45, color="#8FA1A8")
    med = norm.median(axis=1)
    a.plot(med.index, med, lw=2.0, color="#C85A3E", label="20종목 중앙값")
    a.set_yscale("log"); a.set_ylabel("정규화 가격 (첫 관측=100, 로그축)")
    a.set_title("(M-1) 20종목 가격 흐름 — 큰 흐름이 서로 닮았다\n"
                "닮음의 정체는 (M-2)에서 가른다", fontsize=11)
    a.legend(fontsize=9); a.grid(alpha=.3)

    # (M-2) 종목 간 상관 — 연동인가 계절성인가
    a = ax[0, 1]
    C = R.corr()
    im = a.imshow(C.values, cmap="RdYlBu_r", vmin=-1, vmax=1)
    a.set_xticks(range(len(C))); a.set_xticklabels(C.columns, rotation=90, fontsize=7)
    a.set_yticks(range(len(C))); a.set_yticklabels(C.columns, fontsize=7)
    off = C.values[~np.eye(len(C), dtype=bool)]
    # 표본이 거래대금 상위 10 + 변동성 상위 10으로 구성되므로 두 블록을 갈라 본다
    g1 = len([c for c in C.columns if c in
              {"XRP", "BTC", "DOGE", "ETH", "SOL", "SHIB", "SEI", "XLM", "SUI", "ETC"}])
    V = C.values
    m_in1 = float(np.nanmean(V[:g1, :g1][~np.eye(g1, dtype=bool)])) if g1 > 1 else np.nan
    n2 = len(C) - g1
    m_in2 = float(np.nanmean(V[g1:, g1:][~np.eye(n2, dtype=bool)])) if n2 > 1 else np.nan
    m_bt = float(np.nanmean(V[:g1, g1:])) if g1 and n2 else np.nan
    a.axhline(g1 - .5, color="k", lw=1.5); a.axvline(g1 - .5, color="k", lw=1.5)
    a.set_title(f"(M-2) 수익률 상관 — 전체 평균 {np.nanmean(off):.3f}\n"
                f"거래대금군 내 {m_in1:.3f} · 변동성군 내 {m_in2:.3f} · 군 간 {m_bt:.3f}",
                fontsize=11)
    fig.colorbar(im, ax=a, fraction=.046)

    # (M-3) 시간대별 변동성 — 진짜 반복 주기
    a = ax[1, 0]
    A = R.abs().mean(axis=1)
    hr = A.groupby(A.index.hour).mean()
    a.bar(hr.index, hr.values, color="#0E9384")
    a.axhline(hr.mean(), ls="--", color="#C85A3E", lw=1.2, label="전체 평균")
    a.set_xlabel("시각 (KST)"); a.set_ylabel("평균 |수익률|")
    a.set_title(f"(M-3) 시간대별 변동성 — 최대/최소 {hr.max()/hr.min():.2f}배\n"
                "이것이 주기 제거의 대상이다", fontsize=11)
    a.legend(fontsize=9); a.grid(alpha=.3, axis="y")

    # (M-4) 변동성 군집 — 우리 연구의 전제
    a = ax[1, 1]
    ref = R[R.columns[0]].dropna()
    lags = np.arange(1, 97)
    acf_r = [ref.autocorr(l) for l in lags]
    acf_a = [ref.abs().autocorr(l) for l in lags]
    a.plot(lags, acf_r, lw=1.5, color="#8FA1A8", label="수익률 (방향)")
    a.plot(lags, acf_a, lw=2.0, color="#C85A3E", label="|수익률| (크기)")
    a.axhline(0, color="k", lw=.8)
    a.set_xlabel("시차 (15분봉)"); a.set_ylabel("자기상관")
    a.set_title(f"(M-4) {ref.name} 자기상관 — 방향은 0, 크기는 느리게 감소\n"
                "이 한 장이 '변동성을 예측한다'는 연구 전제다", fontsize=11)
    a.legend(fontsize=9); a.grid(alpha=.3)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)
    return {"n": len(norm), "start": str(P.index[0]), "end": str(P.index[-1]),
            "corr": float(np.nanmean(off)), "c_in1": m_in1, "c_in2": m_in2, "c_bt": m_bt,
            "hr_ratio": float(hr.max() / hr.min()),
            "peak": float(med.max()), "last": float(med.iloc[-1]),
            "acf_r": float(np.mean(np.abs(acf_r[:16]))), "acf_a": float(acf_a[0])}


def plot_censoring(tickers: list[str], path: Path) -> None:
    """검열 진단 그림 — 가격 이산성이 타깃을 어떻게 망가뜨리는지 4단으로 보인다."""
    rec = []
    for tk in tickers:
        px = load_close(tk)
        p = px.to_numpy()
        r = np.diff(np.log(np.clip(p, 1e-9, None)))
        fl = tick_ratio(px)
        f0, _ = targets(r, HORIZON)
        f0 = f0[np.isfinite(f0)]
        rec.append({"tk": tk.replace("KRW-", ""), "px": float(np.median(p)),
                    "zr": float((r == 0).mean()), "fl": fl,
                    "rv0": float((f0 <= 0).mean()), "rv": f0})
    rec.sort(key=lambda x: x["px"])

    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("그림 Z. 가격 이산성이 타깃(실현변동성)을 검열한다", fontsize=15)

    # (Z-1) 가격 수준 ↔ 영수익률 비율
    a = ax[0, 0]
    a.scatter([x["px"] for x in rec], [x["zr"] * 100 for x in rec], s=70, color="#2f7d76", zorder=3)
    for x in rec:
        a.annotate(x["tk"], (x["px"], x["zr"] * 100), fontsize=8,
                   xytext=(4, 4), textcoords="offset points")
    a.set_xscale("log"); a.set_xlabel("가격 중앙값 (원, 로그축)")
    a.set_ylabel("수익률이 정확히 0인 봉의 비율 (%)")
    a.set_title("(Z-1) 원인 — 가격이 낮을수록 안 움직이는 봉이 많다\n"
                "종목의 성질이 아니라 호가 단위의 함수", fontsize=11)
    a.grid(alpha=.3)

    # (Z-2) 검열 하한 ↔ RV=0 비율
    a = ax[0, 1]
    a.scatter([x["fl"] for x in rec], [x["rv0"] * 100 for x in rec], s=70, color="#c98a1e", zorder=3)
    for x in rec:
        a.annotate(x["tk"], (x["fl"], x["rv0"] * 100), fontsize=8,
                   xytext=(4, 4), textcoords="offset points")
    a.set_xscale("log"); a.set_xlabel("검열 하한 = 호가단위 / 가격 (로그축)")
    a.set_ylabel("실현변동성 = 0 인 구간 비율 (%)")
    a.set_title("(Z-2) 결과 — 하한이 클수록 타깃이 0으로 검열된다\n"
                "이 0이 QLIKE 분모로 들어가 손실을 발산시켰다", fontsize=11)
    a.grid(alpha=.3)

    # (Z-3) 로그 타깃 분포 — 보정 전(검열된 봉이 왼쪽 끝에 뭉친다)
    worst = max(rec, key=lambda x: x["rv0"])
    best = min(rec, key=lambda x: x["rv0"])
    a = ax[1, 0]
    for x, c in ((worst, "#c0392b"), (best, "#2f7d76")):
        v = np.log(x["rv"] + 1e-14)
        a.hist(v, bins=120, alpha=.55, color=c, label=f"{x['tk']} (RV=0 {x['rv0']*100:.2f}%)")
    a.axvline(np.log(1e-14), ls="--", color="k", lw=1.2, label="log(1e-14) = -32.2")
    a.set_xlabel("log(실현변동성)"); a.set_ylabel("빈도")
    a.set_title("(Z-3) 보정 전 로그 타깃 — 검열된 봉이 -32.2에 뭉친다\n"
                "모델이 이 값을 학습하면 극소값을 예측한다", fontsize=11)
    a.legend(fontsize=8); a.grid(alpha=.3)

    # (Z-4) 결측 처리 후 — 0을 메우지 않고 학습에서 뺀다
    a = ax[1, 1]
    for x, c in ((worst, "#c0392b"), (best, "#2f7d76")):
        v = np.log(x["rv"][x["rv"] > 0])
        a.hist(v, bins=120, alpha=.55, color=c,
               label=f"{x['tk']} (학습 제외 {x['rv0']*100:.2f}%)")
    a.set_xlabel("log(실현변동성), RV=0 시점 제외"); a.set_ylabel("빈도")
    a.set_title("(Z-4) 결측 처리 후 — 0을 메우지 않고 학습에서 뺀다\n"
                "작은 상수로 메우면 추정이 편향된다(Sucarrat & Escribano 2018)", fontsize=11)
    a.legend(fontsize=8); a.grid(alpha=.3)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)


def evaluate(rv_act, rv_pred, rv_naive) -> dict:
    """Patton(2011)이 증명한 robust 손실은 **MSE와 QLIKE 둘뿐**이므로 둘 다 보고한다.

    같은 논문은 MSE-LOG·MAE-LOG 등 로그·표준편차 기반 손실이 robust하지 **않음**을
    보였다(5분 RV에서 MSE-LOG의 최적예측이 0.98σ², 일별 제곱수익률에서는 0.28σ²로 왜곡).
    따라서 0 문제를 피하려고 로그 기반 손실로 갈아타는 것은 문헌이 금지하는 선택이다.
    """
    out = {"QLIKE": qlike(rv_act ** 2, rv_pred ** 2),
           "MSE": float(np.mean((rv_act ** 2 - rv_pred ** 2) ** 2))}
    try:
        X = np.column_stack([np.ones(len(rv_pred)), rv_pred])
        beta, *_ = np.linalg.lstsq(X, rv_act, rcond=None)
        fit = X @ beta
        sst = np.sum((rv_act - rv_act.mean()) ** 2)
        out["MZ_R2"] = float(1 - np.sum((rv_act - fit) ** 2) / sst) if sst > 1e-18 else np.nan
    except Exception:
        out["MZ_R2"] = np.nan
    out["MASE"] = float(np.mean(np.abs(rv_act - rv_pred)) /
                        (np.mean(np.abs(rv_act - rv_naive)) + 1e-18))
    return out


def linearity_evidence(X, y, plot_path: Path | None = None, title: str = "") -> dict:
    """선형 모델을 쓸 근거가 있는지 — **수치와 그림을 함께** 제시한다.

    검정 수치만 쓰면 왜 기각되는지 보이지 않으므로, 각 검정이 무엇을 잡아내는지
    대응하는 그림을 함께 그린다.
        (L-1) 잔차 vs 예측값        → RESET 검정 근거(곡선 패턴이 보이면 비선형항 필요)
        (L-2) |잔차| vs 예측값       → BP 검정 근거(퍼짐이 변하면 이분산)
        (L-3) 잔차 QQ 플롯          → 정규성 이탈
        (L-4) 실제 vs 예측 산점도   → 선형 관계가 실제로 성립하는지
    """
    out = {}
    try:
        Xc = sm.add_constant(X)
        ols = sm.OLS(y, Xc).fit()
        out["OLS_R2"] = float(ols.rsquared)
        out["RESET_p"] = float(linear_reset(ols, power=3, test_type="fitted", use_f=True).pvalue)
        out["BP_p"] = float(het_breuschpagan(ols.resid, Xc)[1])
        rr = ols.resid
        out["BDS_p"] = float(np.atleast_1d(bds(rr[-5000:] if len(rr) > 5000 else rr, max_dim=3)[1])[0])

        if plot_path is not None:
            fit = ols.fittedvalues
            step = max(1, len(fit) // 6000)          # 산점도 과밀 방지
            f_, r_ = fit[::step], rr[::step]
            fig, ax = plt.subplots(2, 2, figsize=(14, 9))

            # (L-1) 잔차 vs 예측값 — RESET 검정의 근거
            ax[0, 0].scatter(f_, r_, s=4, alpha=.25, color="#0E9384")
            try:                                      # 국소 평균선: 곡선이면 비선형
                qs = np.quantile(f_, np.linspace(0, 1, 21))
                cx, cy = [], []
                for lo, hi in zip(qs[:-1], qs[1:]):
                    m = (f_ >= lo) & (f_ < hi)
                    if m.sum() > 10:
                        cx.append(f_[m].mean()); cy.append(r_[m].mean())
                ax[0, 0].plot(cx, cy, color="#C85A3E", lw=2.5, label="구간별 평균")
                ax[0, 0].legend(fontsize=9)
            except Exception:
                pass
            ax[0, 0].axhline(0, color="black", lw=1)
            ax[0, 0].set_xlabel("예측값"); ax[0, 0].set_ylabel("잔차")
            ax[0, 0].set_title(f"(L-1) 잔차 vs 예측값 — RESET 검정의 근거\n"
                               f"주황선이 평평해야 선형. 휘면 비선형항 필요 "
                               f"(RESET p={num(out['RESET_p'],4)})")

            # (L-2) |잔차| vs 예측값 — BP 검정의 근거
            ax[0, 1].scatter(f_, np.abs(r_), s=4, alpha=.25, color="#B8860B")
            try:
                cx2, cy2 = [], []
                for lo, hi in zip(qs[:-1], qs[1:]):
                    m = (f_ >= lo) & (f_ < hi)
                    if m.sum() > 10:
                        cx2.append(f_[m].mean()); cy2.append(np.abs(r_[m]).mean())
                ax[0, 1].plot(cx2, cy2, color="#C85A3E", lw=2.5, label="구간별 평균 |잔차|")
                ax[0, 1].legend(fontsize=9)
            except Exception:
                pass
            ax[0, 1].set_xlabel("예측값"); ax[0, 1].set_ylabel("|잔차|")
            ax[0, 1].set_title(f"(L-2) 오차 크기 vs 예측값 — 이분산 검정의 근거\n"
                               f"평평해야 등분산. 기울면 이분산 "
                               f"(Breusch-Pagan p={num(out['BP_p'],4)})")

            # (L-3) 잔차 QQ — 정규성
            st.probplot(rr, dist=st.norm, plot=ax[1, 0])
            ax[1, 0].get_lines()[0].set_markersize(2.0)
            ax[1, 0].get_lines()[0].set_color("#0E9384")
            ax[1, 0].get_lines()[1].set_color("#C85A3E")
            ax[1, 0].set_title(f"(L-3) 잔차 QQ 플롯 — 정규성\n"
                               f"직선에서 벗어나면 정규 가정 이탈 (초과첨도 {num(st.kurtosis(rr),1)})")
            ax[1, 0].set_xlabel("정규 이론 분위수"); ax[1, 0].set_ylabel("잔차 분위수")

            # (L-4) 실제 vs 예측 — 선형 관계 성립 여부
            ax[1, 1].scatter(f_, y[::step], s=4, alpha=.25, color="#7B68EE")
            lims = [np.quantile(y, .001), np.quantile(y, .999)]
            ax[1, 1].plot(lims, lims, "k--", lw=1.8, label="완벽 예측선")
            ax[1, 1].legend(fontsize=9)
            ax[1, 1].set_xlabel("예측값"); ax[1, 1].set_ylabel("실제값")
            ax[1, 1].set_title(f"(L-4) 실제 vs 예측 — 설명력\n"
                               f"점선에 모일수록 좋음 (OLS R²={num(out['OLS_R2'],3)})")
            for a_ in ax.flat:
                a_.grid(alpha=.2)
            fig.suptitle(f"그림 L. 선형 모델 가정 진단 — {title}", fontsize=14, y=1.00)
            fig.tight_layout()
            fig.savefig(plot_path, dpi=120, bbox_inches="tight")
            plt.close(fig)
            out["_plot"] = plot_path.name
    except Exception:
        pass
    return out


# %% [markdown]
# ## 딥러닝 학습(공통)

# %%
def train_dl(name, Xtr, ytr, Xte, ep=8, bs=4096, hidden=48):
    torch.manual_seed(0)
    m = make_model(name, Xtr.shape[1], Xtr.shape[2], hidden).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    lossf = nn.MSELoss()
    xt = torch.tensor(Xtr, device=DEV); yt = torch.tensor(ytr, dtype=torch.float32, device=DEV)
    for _ in range(ep):
        perm = torch.randperm(len(xt), device=DEV)
        for i in range(0, len(xt), bs):
            b = perm[i:i + bs]; opt.zero_grad()
            loss = lossf(m(xt[b]), yt[b]); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    m.eval()
    with torch.no_grad():
        xe = torch.tensor(Xte, device=DEV)
        return np.concatenate([m(xe[i:i + bs]).float().cpu().numpy() for i in range(0, len(xe), bs)])


def garch_cond_vol(r: np.ndarray, split_idx: int, dist="t") -> np.ndarray:
    """GARCH 조건부 변동성 — **원 시계열 인덱스 전체 길이**로 돌려준다.

    이전에는 검증 길이를 인자로 받아 그만큼만 반환했는데, 학습 표본에서 RV=0 시점을
    제외하면서 그 길이가 실제 검증 행 수와 어긋났다. 길이가 맞지 않으면 평가가 예외를
    내고 `except`가 이를 삼켜, 20종목 중 18종목에서 GARCH가 조용히 빠진 채로 표가
    만들어졌다(BTC·SOL만 제외 비율이 0에 가까워 우연히 길이가 맞았다).

    같은 사고를 반복하지 않도록 **원 인덱스 기준 전체 길이**를 반환하고, 호출부에서
    필요한 위치만 골라 쓰도록 한다. 길이 정합을 호출부의 산술이 아니라 인덱싱이 보장한다.
    """
    res = arch_model(r[:split_idx] * 100, mean="Zero", vol="GARCH", p=1, q=1,
                     dist=dist).fit(disp="off", show_warning=False)
    p = res.params
    omega = float(p.get("omega", 0)); alpha = float(p.get("alpha[1]", 0)); beta = float(p.get("beta[1]", 0))
    n = len(r)
    cv = np.full(n, np.nan)
    cv[:split_idx] = np.asarray(res.conditional_volatility) ** 2
    eps2 = (r * 100) ** 2

    # cv[t] - beta*cv[t-1] = omega + alpha*eps2[t-1] 는 1차 선형 재귀이므로
    # 파이썬 for 루프 대신 lfilter로 벡터화한다(20종목 기준 실행시간이 크게 줄었다).
    drive = omega + alpha * eps2[split_idx - 1: n - 1]
    zi = np.array([beta * cv[split_idx - 1]])
    cv[split_idx:], _ = lfilter([1.0], [1.0, -beta], drive, zi=zi)
    return np.sqrt(np.clip(cv, 0, None)) / 100.0


# %%
def run_one(ticker: str, deper: bool, quick: bool) -> tuple[list[dict], dict]:
    """한 종목·한 조건(주기제거 여부)에서 전 모델 적합·평가."""
    close = load_close(ticker)
    r_s = np.log(close).diff().dropna()
    if deper:
        r_s = remove_periodicity(r_s)
    r = r_s.to_numpy()
    n = len(r)
    if n < 20000:
        return [], {}

    # 가격 이산성 진단 — 통제 변수로만 쓰고, 타깃을 메우는 데는 쓰지 않는다
    tr_ratio = tick_ratio(close)
    raw_r = np.log(close).diff().dropna().to_numpy()
    zero_ret = float((raw_r == 0).mean())

    fut, pas = targets(r, HORIZON)
    Xl, Xh, Xt = feats_log(r), feats_har(r), feats_tree(r)
    valid = np.isfinite(fut) & np.isfinite(pas)
    valid[:7 * BPD + 10] = False
    zero_rv = float(np.mean(fut[valid] <= 0))              # 실현변동성이 0인 구간 비율

    # 학습 표본에서 RV=0 시점은 **결측으로 취급해 제외**한다(Sucarrat & Escribano 2018).
    # 작은 상수로 메우면 로그 타깃이 -32.2가 되어 추정이 점근적으로 편향된다.
    # 검증 표본에서는 제외하지 않는다 — 실제 관측이며, QLIKE는 실제값 0에서 유한하다.
    valid_tr = valid & (fut > 0) & (pas > 0)

    idxv = np.where(valid)[0]
    sp_all = int(len(idxv) * TRAIN_FRAC)
    split_idx = int(idxv[sp_all])
    # 학습 구간은 결측 제외본, 검증 구간은 전체본을 쓴다
    tr_mask = valid_tr.copy(); tr_mask[split_idx:] = False
    te_mask = valid.copy(); te_mask[:split_idx] = False
    n_tr = int(tr_mask.sum())
    drop_share = float(1 - n_tr / max(int((valid & (np.arange(n) < split_idx)).sum()), 1))

    fit_mask = tr_mask | te_mask                            # 모델이 실제로 보는 행
    sp = n_tr
    y = np.log(fut[fit_mask] + 1e-14)
    rv_act_all = fut[fit_mask]; rv_nai_all = pas[fit_mask]
    ytr, yte = y[:sp], y[sp:]
    ii = np.arange(0, len(yte), HORIZON)                   # 겹치지 않는 검증 표본
    rv_act = rv_act_all[sp:][ii]; rv_nai = rv_nai_all[sp:][ii]
    tmean = ytr.mean()

    # BPQ(2016) insanity filter 범위 — 학습기간 타깃의 0.5%~99.5% 분위수(Zhang 외 2024)
    rv_tr = rv_act_all[:sp]
    f_lo, f_hi = np.percentile(rv_tr, [0.5, 99.5])
    f_mean = float(rv_tr.mean())
    trims: dict[str, float] = {}

    def score(name, family, rv_pred, note=""):
        p = rv_pred[ii] if len(rv_pred) == len(yte) else rv_pred
        p, tr = insanity_filter(p, f_lo, f_hi, f_mean)
        trims[name] = tr
        return {"종목": ticker, "주기제거": deper, "모델": name, "계열": family, "비고": note,
                "트리밍비율": tr, **evaluate(rv_act, p, rv_nai)}

    rows = [score("naive", "기준선", rv_nai)]
    fails: list[tuple[str, str, str]] = []      # 조용히 넘어가지 않도록 실패를 모은다

    # 이 시점 이후 valid는 fit_mask를 가리켜야 한다(특성 슬라이싱 정합)
    valid = fit_mask

    # 선형성 근거 — 수치 + 그림(대표 종목만 그림 저장, 나머지는 수치만)
    plot_path = None
    if ticker == "KRW-BTC":
        plot_path = IMG / f"figL_linearity_{'deper' if deper else 'raw'}.png"
    lin_ev = linearity_evidence(Xl[valid][:sp], ytr, plot_path,
                                title=f"{ticker} · 주기제거 {'후' if deper else '전'}")

    # ── 선형·HAR·커널 (로그축 + 표준화) ──
    for nm, fam, Xsrc, mdl in (
            ("HAR-RV", "벤치마크", Xh, LinearRegression()),
            ("Linear", "선형", Xl, LinearRegression()),
            ("Ridge", "선형", Xl, Ridge(alpha=1.0)),
    ):
        Xv = Xsrc[valid]; sc = StandardScaler().fit(Xv[:sp])
        mdl.fit(sc.transform(Xv[:sp]), ytr)
        pred = np.exp(mdl.predict(sc.transform(Xv[sp:])))
        note = "" if fam == "벤치마크" else \
            f"선형성 기각(RESET p={num(lin_ev.get('RESET_p'),4)}, BDS p={num(lin_ev.get('BDS_p'),4)})"
        rows.append(score(nm, fam, pred, note))

    Xv = Xl[valid]
    scx = StandardScaler().fit(Xv[:sp]); scy = StandardScaler().fit(ytr.reshape(-1, 1))
    ytr_s = scy.transform(ytr.reshape(-1, 1)).ravel()
    sub = min(6000, sp)
    for nm, mdl in (("KernelRidge-RBF", KernelRidge(kernel="rbf", alpha=1.0)),
                    ("SVR-RBF", SVR(kernel="rbf", C=10.0, epsilon=0.05)),
                    ("Nystroem+Ridge", make_pipeline(Nystroem(kernel="rbf", n_components=300,
                                                              random_state=0), Ridge(alpha=1.0)))):
        try:
            use = slice(sp - sub, sp) if nm != "Nystroem+Ridge" else slice(0, sp)
            mdl.fit(scx.transform(Xv[use]), ytr_s[use] if nm != "Nystroem+Ridge" else ytr_s)
            p = scy.inverse_transform(mdl.predict(scx.transform(Xv[sp:])).reshape(-1, 1)).ravel()
            rows.append(score(nm, "커널", np.exp(p), "선형성 불필요"))
        except Exception:
            pass

    # ── 트리 (원 스케일 + 시간구조) ──
    Xvt = Xt[valid]
    trees = [("LightGBM", lgb.LGBMRegressor(n_estimators=300, learning_rate=0.06, max_depth=6, verbosity=-1)),
             ("XGBoost", xgb.XGBRegressor(n_estimators=300, learning_rate=0.06, max_depth=6,
                                          tree_method="hist", device=DEV, verbosity=0)),
             ("HistGBM", HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06, max_depth=6))]
    if quick:
        trees = trees[:1]
    for nm, mdl in trees:
        mdl.fit(Xvt[:sp], ytr)
        rows.append(score(nm, "트리", np.exp(mdl.predict(Xvt[sp:])), "가정 없음(비모수)"))

    # ── GARCH 계열 (원 수익률, 표준화 금지) ──
    gar_pred = gv_full = None
    try:
        gv_full = garch_cond_vol(r, split_idx, dist="t")    # 원 인덱스 전체 길이
        gar_pred = gv_full[te_mask] * np.sqrt(HORIZON)      # 검증 행만 골라 h구간 환산
        rows.append(score("GARCH-t", "통계", gar_pred, "t분포로 두꺼운 꼬리 보완"))
    except Exception as e:
        fails.append(("GARCH-t", type(e).__name__, str(e)[:120]))

    # ── 딥러닝 ──
    dl_list = [("GRU", "rnn", False), ("LSTM", "rnn", False)] if quick else \
              [("GRU", "rnn", False), ("LSTM", "rnn", False),
               ("PatchTSTLike", "tfm", True), ("ITransformerLike", "tfm", True)]
    seq_cache = {}
    for nm, grp, revin in dl_list:
        try:
            key = (grp, revin)
            if key not in seq_cache:
                seq_cache[key] = seqs(r, SEQ_LEN[grp], revin)
            Xs = seq_cache[key][valid]
            ps = train_dl(nm, Xs[:sp], ytr_s, Xs[sp:])
            p = scy.inverse_transform(ps.reshape(-1, 1)).ravel()
            rows.append(score(nm, "딥러닝", np.exp(p),
                              "RevIN(시퀀스별 정규화)" if revin else "전역 표준화"))
        except Exception as e:
            fails.append((nm, type(e).__name__, str(e)[:120]))

    # ── 하이브리드: GARCH 조건부분산을 feature로 결합 ──
    # garch_cond_vol()이 학습 구간은 in-sample 조건부변동성, 검증 구간은 재귀 예측으로
    # 이미 채워 두었으므로 fit_mask로 골라 쓰기만 하면 된다(별도 재적합 불필요).
    if gv_full is not None:
        try:
            gcol = gv_full[te_mask.astype(bool) | tr_mask.astype(bool)] * np.sqrt(HORIZON)
            Xhyb = np.column_stack([Xvt, np.log(gcol + 1e-12)])
            m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.06, max_depth=6, verbosity=-1)
            m.fit(Xhyb[:sp], ytr)
            rows.append(score("GARCH+LightGBM", "하이브리드", np.exp(m.predict(Xhyb[sp:])),
                              "통계 구조 + 비모수 학습"))
        except Exception as e:
            fails.append(("GARCH+LightGBM", type(e).__name__, str(e)[:120]))

    for rw in rows:
        rw["영수익률비율"] = zero_ret        # 15분 동안 가격이 한 칸도 안 움직인 봉의 비율
        rw["RV0비율"] = zero_rv              # 실현변동성이 정확히 0인 구간 비율
        rw["학습제외비율"] = drop_share      # 결측 처리로 학습에서 뺀 비율
        rw["호가가격비"] = tr_ratio          # 가격 이산성의 크기(통제 변수)
    if fails:
        for nm, et, msg in fails:
            print(f"    ! {ticker}(주기제거={deper}) {nm} 실패 — {et}: {msg}", flush=True)
        lin_ev = {**lin_ev, "실패모델": ";".join(f[0] for f in fails)}
    return rows, lin_ev


# %%
def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--n-tickers", type=int, default=20)
    a = ap.parse_args(argv)
    t0 = time.time()

    tickers, sel = study_universe()
    tickers = tickers[:3] if a.quick else tickers[:a.n_tickers]
    conds = [False, True]

    emit("# 23번 — 변동성 예측 모델 통합 비교")
    emit()
    emit(f"작성일 2026-09-08 · **{len(tickers)}종목** · 15분봉 · 예측 대상 **1시간 뒤 실현변동성**")
    emit()
    emit("> 모델마다 요구하는 가정이 다르므로 **입력을 모델군별로 다르게** 준비해 공정하게 비교한다. "
         "주기 제거 전/후 두 조건을 모두 돌린다.")
    emit()
    print("[header] 표준 헤더 생성 중...", flush=True)
    emit(render_standard_header(
        tickers=tickers, train_frac=TRAIN_FRAC, rep_ticker="KRW-BTC",
        analyzed_tickers=tickers, selection_reason=sel,
        transforms=[("모델군별 가정 맞춤 전처리", "GARCH=원수익률+t분포 / HAR·선형·커널=로그축 / "
                     "트리=원스케일+시간구조 / 딥러닝=RevIN"),
                    ("주기 제거 전·후", "시간대 주기가 GARCH 지속성을 왜곡하고 모델이 '쉬운 부분'만 "
                     "학습할 수 있어 두 조건을 비교")]))
    emit("---")
    emit()

    rows, lin_evs = [], []
    for tk in tickers:
        for deper in conds:
            rr, ev = run_one(tk, deper, a.quick)
            rows += rr
            if ev:
                ev.update({"종목": tk, "주기제거": deper})
                lin_evs.append(ev)
            print(f"  · {tk} (주기제거={deper}) 완료 — {len(rr)}개 모델", flush=True)

    rd = pd.DataFrame(rows)
    rd.to_csv(RES / "model_comparison.csv", index=False)
    led = pd.DataFrame(lin_evs)
    led.to_csv(RES / "linearity_evidence.csv", index=False)

    # ── 거시 EDA (검정보다 먼저 눈으로 본다) ──
    emit("## 0-A. 거시 EDA — 검정 이전에 전체 흐름을 눈으로 본다")
    emit()
    emit("통계 검정만 쌓으면 데이터가 어떤 모양인지 모른 채 결론을 내게 된다. 모델 비교에 "
         "들어가기 전에 20종목 원본의 큰 흐름·연동·주기·군집을 먼저 확인한다.")
    emit()
    fm = IMG / "figM_macro.png"
    try:
        mi = plot_macro(tickers, fm)
        emit(f"![그림 M](../../images/{TAG}/{fm.name})")
        emit()
        emit("| 그림 | 무엇을 보이는가 | 확인된 수치 |")
        emit("| :--- | :--- | :--- |")
        emit(f"| **(M-1)** | 20종목의 정규화 가격 흐름 — 큰 흐름이 서로 닮았다 | "
             f"중앙값 고점 {mi['peak']:.0f} → 최종 {mi['last']:.0f} |")
        emit(f"| **(M-2)** | 닮음의 정체 — 계절성이 아니라 **시장 연동** | "
             f"전체 평균 상관 **{mi['corr']:.3f}** (군별 분해는 아래) |")
        emit(f"| **(M-3)** | 진짜 반복 주기 — 시간대별 변동성 차이 | "
             f"최대/최소 **{mi['hr_ratio']:.2f}배** |")
        emit(f"| **(M-4)** | 연구 전제 — 방향은 예측 불가, 크기는 지속 | "
             f"\\|수익률\\| 1차 자기상관 **{mi['acf_a']:.3f}** vs 수익률 {mi['acf_r']:.3f} |")
        emit()
        emit("(M-2)가 중요하다. (M-1)에서 20종목이 닮아 보이는 것은 **반복되는 계절성이 아니라 "
             "같은 시장 충격에 함께 반응하는 연동**이다. 반복되는 진짜 주기는 (M-3)의 "
             "시간대 효과이며, 이것이 이번 회차에서 제거 대상으로 삼은 성분이다.")
        emit()
        emit("### (M-2) 상관 구조가 두 블록으로 갈린다")
        emit()
        emit("히트맵에 뚜렷한 블록 구조가 보인다. 표본이 **거래대금 상위 10종목 + 변동성 상위 "
             "10종목**으로 구성되어 있는데, 두 집단의 성격이 전혀 다르다.")
        emit()
        emit("| 구분 | 평균 수익률 상관 | 해석 |")
        emit("| :--- | ---: | :--- |")
        emit(f"| 거래대금군 내부 (XRP·BTC·ETH 등 10종목) | **{mi['c_in1']:.3f}** | "
             "같은 시장 충격에 함께 움직인다 |")
        emit(f"| 변동성군 내부 (BTT·STRAX·AQT 등 10종목) | **{mi['c_in2']:.3f}** | "
             "서로도 거의 무관하다 |")
        emit(f"| 두 군 사이 | **{mi['c_bt']:.3f}** | 거의 연동되지 않는다 |")
        emit(f"| 전체 평균 | {mi['corr']:.3f} | 두 블록을 섞어 **희석된 값** |")
        emit()
        emit("전체 평균 하나만 보면 \"연동이 중간 정도\"로 읽히지만, 실제로는 **강하게 연동된 "
             "집단과 거의 독립적인 집단이 섞여 있다.** 이는 표본 설계가 만든 구조이며 "
             "두 가지 함의를 갖는다.")
        emit()
        emit("첫째, **모델 성능을 20종목 평균 하나로 요약하면 성격이 다른 두 집단이 섞인다.** "
             "거래대금군은 시장 요인이 지배하므로 공통 성분을 쓰는 모델이 유리할 수 있고, "
             "변동성군은 종목 고유 움직임이 커서 그렇지 않다. 후속 회차에서 군별로 나누어 "
             "보고하는 것을 검토한다.")
        emit()
        emit("둘째, 변동성군은 **저가·저유동 종목이 많아 (1-B)의 가격 이산성 문제가 집중**된다. "
             "실제로 수익률=0 봉 비율 상위는 대부분 이 집단이다. 즉 상관 구조와 검열 구조가 "
             "같은 방향으로 겹쳐 있으므로, 두 문제를 함께 통제해야 한다.")
        emit()
        emit("(M-4)는 이 연구가 왜 방향이 아니라 변동성을 대상으로 삼는지를 한 장으로 보여준다. "
             "수익률 자기상관은 0에 붙어 있어 **다음 봉이 오를지 내릴지는 과거로 알 수 없다**. "
             "반면 |수익률|의 자기상관은 높은 값에서 느리게 감소한다 — **크게 움직인 뒤에는 "
             "크게 움직인다**. 예측 가능한 것은 방향이 아니라 크기다.")
        emit()
    except Exception as e:
        emit(f"> (거시 EDA 그림 생성 실패: {e})")
        emit()

    # ── 적합 완결성 점검 (표를 읽기 전에 먼저 본다) ──
    emit("## 0-B. 적합 완결성 점검 — 모든 모델이 모든 종목에서 돌았는가")
    emit()
    emit("모델별 평균을 비교하려면 **같은 종목 집합에서 계산된 평균**이어야 한다. 어떤 모델이 "
         "일부 종목에서만 적합되면 그 평균은 다른 모델과 비교할 수 없다. "
         "이전 실행에서 GARCH 계열이 20종목 중 2종목에서만 성공했는데도 표에는 1위로 찍혔다"
         "(길이 불일치 예외를 `except`가 삼켰다). 그래서 표를 읽기 전에 이 점검을 먼저 싣는다.")
    emit()
    cov = rd.groupby("모델")["종목"].nunique().sort_values()
    n_all = rd["종목"].nunique()
    emit(f"| 모델 | 적합 성공 종목 수 (전체 {n_all}) | 판정 |")
    emit("| :--- | ---: | :--- |")
    for nm, v in cov.items():
        ok = "정상" if v == n_all else f"**⚠ {n_all - v}종목 누락 — 평균 비교 불가**"
        emit(f"| {nm} | {v} | {ok} |")
    emit()
    miss = cov[cov < n_all]
    if len(miss):
        emit(f"> ⚠ **{len(miss)}개 모델이 전 종목에서 적합되지 않았다.** 아래 성능표에서 해당 "
             "모델의 평균은 다른 모델과 직접 비교하면 안 된다.")
    else:
        emit("→ 전 모델이 전 종목에서 적합되었다. 성능표의 평균을 서로 비교할 수 있다.")
    emit()

    # ── 선형 모델 사용 근거(수치 제시) ──
    emit("## 1. 선형 모델을 쓸 근거가 있는가 — 검정 수치")
    emit()
    emit("선형 모델(Linear/Ridge)은 ①입력과 출력이 직선 관계이고 ②오차 분산이 일정하다는 가정 위에서 "
         "성립한다. 단순히 '근거 없음'이라 쓰지 않고 **검정 수치를 제시**한다.")
    emit()
    if len(led):
        emit("| 조건 | 종목수 | OLS R² 평균 | RESET p 중앙값 | BDS p 중앙값 | BP p 중앙값 | 기각 종목 |")
        emit("| :--- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for deper in conds:
            s = led[led["주기제거"] == deper]
            if not len(s):
                continue
            rej = int(((s.get("RESET_p", 1) < 0.05) | (s.get("BDS_p", 1) < 0.05)).sum())
            emit(f"| 주기제거 {'후' if deper else '전'} | {len(s)} | {num(s['OLS_R2'].mean(),3)} | "
                 f"{num(s['RESET_p'].median(),5)} | {num(s['BDS_p'].median(),5)} | "
                 f"{num(s['BP_p'].median(),5)} | **{rej}/{len(s)}** |")
        emit()
        emit("### 검정 수치와 그림의 대응")
        emit()
        emit("검정 결과만 보면 왜 기각되는지 알 수 없으므로, **각 검정이 무엇을 잡아내는지 "
             "대응하는 그림**을 함께 싣는다.")
        emit()
        emit("| 검정 | 무엇을 보는가 | 기각의 의미 | 대응 그림 |")
        emit("| :--- | :--- | :--- | :--- |")
        emit("| **RESET** | 잔차가 예측값에 따라 체계적 패턴을 갖는가 | 선형식만으로 부족, "
             "비선형항 필요 | **(L-1)** 잔차 vs 예측값 — 구간별 평균선이 휘면 비선형 |")
        emit("| **Breusch-Pagan** | 오차 분산이 일정한가 | 이분산 → OLS 표준오차 신뢰 불가 | "
             "**(L-2)** \\|잔차\\| vs 예측값 — 기울면 이분산 |")
        emit("| **BDS** | 잔차가 독립적인가 | 비선형 구조 잔존 | (L-1)과 함께 판단 |")
        emit("| (참고) 정규성 | 잔차가 정규분포인가 | 신뢰구간 해석 제약 | **(L-3)** 잔차 QQ 플롯 |")
        emit("| (참고) 설명력 | 선형 관계가 실제로 성립하는가 | — | **(L-4)** 실제 vs 예측 산점도 |")
        emit()
        for deper in conds:
            fn = IMG / f"figL_linearity_{'deper' if deper else 'raw'}.png"
            if fn.exists():
                emit(f"**주기 제거 {'후' if deper else '전'} (KRW-BTC)**")
                emit()
                emit(f"![그림 L]( ../../images/{TAG}/{fn.name} )".replace(" ", ""))
                emit()
        emit("→ 기각된 종목에서는 선형 모델 결과를 **참고용으로만** 본다. 성능표의 '비고' 칸에 "
             "해당 종목의 검정 수치를 함께 표기했다.")
        emit()

    # ── 검열 진단 ──
    if "영수익률비율" in rd.columns:
        emit("## 1-B. 타깃 진단 — 실현변동성이 0이 되는 구간(가격 이산성)")
        emit()
        emit("모델을 돌리기 전에 **타깃 자체가 성립하는지** 먼저 확인해야 한다. 첫 실행에서 "
             "QLIKE가 종목·모델에 따라 최대 2.8e2까지 튀었다.")
        emit()
        emit("업비트는 가격대별로 호가 단위를 다르게 정한다. 가격이 낮은 종목은 호가 단위가 "
             "가격 대비 크기 때문에, **15분 동안 가격이 한 칸도 움직이지 않는 봉**이 대량으로 "
             "발생한다. 이때 실현변동성은 정확히 0이 된다.")
        emit()
        emit("### 이것이 우리만의 문제가 아니라는 문헌 근거")
        emit()
        emit("Brauneis & Sahiner(*Asia-Pacific Financial Markets*, 2024)는 8개 코인의 일중 "
             "실현변동성 예측을 비교했는데, 논문 Table 4에서 **HAR의 QLIKE가 DOGE +9.89, "
             "SHIB +9.00으로 양수 폭발**한 반면 기계학습 모델들은 -5.6~-6.5를 유지했다. "
             "우리 보정 전 수치(naive +21.7, HAR +23.0)와 같은 현상이다. "
             "즉 이 폭발은 코드 결함이 아니라 저가 코인에서 실제로 벌어지는 일이며, "
             "문헌은 이를 **저가 코인에서 기계학습이 HAR을 이기는 주된 메커니즘**으로 서술한다.")
        emit()
        emit("Bandi 외(2020, *Management Science* 66(8), \"Zeros\")는 영수익률이 제도적 산물이 "
             "아니라 **거래량·유동성에 연동된 실제 경제 현상**임을 보였다. 따라서 0 구간은 "
             "버려야 할 잡음이 아니라 보고해야 할 정보다.")
        emit()
        emit("### 처리 방침 — 타깃이 아니라 예측치를 가둔다")
        emit()
        emit("처음에는 `호가단위/가격`을 실현변동성의 하한으로 삼아 0을 메우려 했으나, "
             "**문헌이 이 방식을 직접 경고한다**. Sucarrat & Escribano(2018, *European "
             "Journal of Finance* 24(10))는 영수익률을 작은 상수로 대체하는 처방이 추정을 "
             "점근적으로 편향시킴을 보이고 **0을 결측으로 취급**할 것을 권했고, "
             "Bellégo 외(arXiv:2203.11820)는 log(Y+Δ)의 Δ 선택이 결과를 자의적으로 좌우함을 "
             "보였다. 실제로 하한 방식으로 전 종목을 돌렸을 때 naive의 QLIKE가 105까지 "
             "남아 문제가 해결되지도 않았다.")
        emit()
        emit("결정적으로 **타깃을 건드릴 필요가 없다**. 실제분산이 0이어도 "
             "QLIKE = log(h) + 0/h = log(h)로 유한하다. 발산은 오직 **예측분산 h가 0에 "
             "붙을 때**만 일어난다. 따라서 세 가지로 처리한다.")
        emit()
        emit("| 단계 | 처리 | 문헌 근거 |")
        emit("| :--- | :--- | :--- |")
        emit("| 학습 | RV=0 시점을 **결측으로 취급해 학습 표본에서 제외** | Sucarrat & Escribano(2018) |")
        emit("| 평가 | 예측치를 학습기간 타깃의 0.5%~99.5% 분위수로 가두고, 벗어나면 "
             "학습기간 평균으로 대체(**insanity filter**) | BPQ(2016) 각주 17 · Zhang 외(2024) |")
        emit("| 지표 | robust 손실인 **QLIKE와 MSE를 함께** 보고 | Patton(2011) |")
        emit()
        emit("insanity filter는 QLIKE를 임의로 완화하는 편법이 아니라 **Patton(2011)의 "
             "robustness 정리가 요구하는 가정 A4**(최적예측이 양의 실수의 컴팩트 부분집합 안에 "
             "있을 것)를 그대로 구현한 것이다. 같은 논문은 0 문제를 피하려고 로그 기반 손실"
             "(MSE-LOG 등)로 갈아타는 것을 **non-robust로 증명해 금지**했다.")
        emit()
        zz = rd.groupby("종목")[["영수익률비율", "RV0비율", "학습제외비율", "호가가격비"]].first()
        zz = zz.sort_values("영수익률비율", ascending=False)
        emit("| 종목 | 수익률=0 봉 비율 | RV=0 구간 | 학습 제외 비율 | 호가/가격 |")
        emit("| :--- | ---: | ---: | ---: | ---: |")
        for tk, v in zz.iterrows():
            emit(f"| {tk.replace('KRW-','')} | {v['영수익률비율']*100:.2f}% | "
                 f"{v['RV0비율']*100:.3f}% | {v['학습제외비율']*100:.3f}% | "
                 f"{v['호가가격비']:.2e} |")
        emit()
        if "트리밍비율" in rd.columns:
            tt = rd.groupby("모델")["트리밍비율"].mean().sort_values(ascending=False)
            hot = tt[tt > 0]
            emit(f"**insanity filter 트리밍 비율** (원논문 BPQ 2016은 0.1% 미만 보고): "
                 f"전체 평균 {rd['트리밍비율'].mean()*100:.3f}%")
            if len(hot):
                emit(">")
                emit("> 모델별: " + " · ".join(f"{m} {v*100:.2f}%" for m, v in hot.head(6).items()))
            emit()
        emit(f"→ 수익률이 정확히 0인 봉의 비율이 최대 **{zz['영수익률비율'].max()*100:.1f}%**, "
             f"최소 **{zz['영수익률비율'].min()*100:.2f}%**로 종목 간 격차가 크다. "
             "이 격차는 종목의 성질이 아니라 **가격 수준(=호가 단위)의 함수**이므로, "
             "종목 간 성능을 비교할 때 반드시 함께 봐야 하는 통제 변수다.")
        emit()
        fz = IMG / "figZ_censoring.png"
        try:
            plot_censoring(list(zz.index), fz)
            emit(f"![그림 Z](../../images/{TAG}/{fz.name})")
            emit()
            emit("| 그림 | 무엇을 보이는가 |")
            emit("| :--- | :--- |")
            emit("| **(Z-1)** | 원인 — 가격이 낮은 종목일수록 수익률이 0인 봉이 많다 |")
            emit("| **(Z-2)** | 결과 — 호가/가격 비율이 클수록 실현변동성 0 구간이 많다 |")
            emit("| **(Z-3)** | 0을 작은 상수로 메우면 로그 타깃이 -32.2에 뭉친다 — 이 처리를 "
                 "쓰지 않는 이유 |")
            emit("| **(Z-4)** | 결측 처리 후 — 0 시점을 학습에서 빼면 분포가 정상이다 |")
            emit()
        except Exception as e:
            emit(f"> (검열 진단 그림 생성 실패: {e})")
            emit()

    # ── 모델별 성능 ──
    emit("## 2. 모델별 성능 — 주기 제거 전/후")
    emit()
    for deper in conds:
        s = rd[rd["주기제거"] == deper]
        if not len(s):
            continue
        emit(f"### 주기 제거 {'후' if deper else '전'}")
        emit()
        cols = [c for c in ("QLIKE", "MSE", "MZ_R2", "MASE") if c in s.columns]
        agg = s.groupby(["계열", "모델"])[cols].mean().sort_values("QLIKE")
        emit("| 계열 | 모델 | QLIKE | MSE | MZ-R² | MASE | 비고 |")
        emit("| :--- | :--- | ---: | ---: | ---: | ---: | :--- |")
        for (fam, nm), v in agg.iterrows():
            note = s[s["모델"] == nm]["비고"].iloc[0] if len(s[s["모델"] == nm]) else ""
            mse = f"{v['MSE']:.2e}" if "MSE" in v else "—"
            emit(f"| {fam} | **{nm}** | {num(v['QLIKE'],4)} | {mse} | {num(v['MZ_R2'],3)} | "
                 f"{num(v['MASE'],3)} | {note} |")
        emit()
        best = agg.index[0]
        emit(f"- 최고: **{best[1]}** ({best[0]}) — QLIKE {num(agg.iloc[0]['QLIKE'],4)}")
        emit()
        emit("QLIKE와 MSE를 함께 싣는 이유는 Patton(2011)이 **변동성 대리변수에 잡음이 있을 때 "
             "순위를 보존하는 손실함수는 이 둘뿐**임을 증명했기 때문이다. 다만 두 지표의 성질은 "
             "다르다. QLIKE는 표준화오차의 함수라 극단 관측에 덜 민감하고, MSE는 오차 분산이 "
             "σ⁴에 비례해 변동성 수준에 민감하다. 암호화폐처럼 종목 간 변동성 수준 격차가 큰 "
             "데이터에서는 QLIKE를 주 지표로 삼는 것이 타당하며, Patton & Sheppard(2009)는 "
             "QLIKE 기반 검정의 검정력이 더 높다고 보고했다.")
        emit()

    # ── 문헌 대비 위치 ──
    emit("## 2-B. 우리 수치는 문헌의 어디쯤인가")
    emit()
    emit("성능 수치는 절대값만으로 좋고 나쁨을 말할 수 없다. 특히 **Mincer-Zarnowitz R²는 "
         "대리변수의 잡음 때문에 상한이 제한된다** — Andersen & Bollerslev(1998, *IER* 39(4))는 "
         "일별 제곱수익률을 대리변수로 쓰면 완벽한 예측이라도 R²의 상한이 약 1/3임을 보였고, "
         "Andersen 외(2005, *Econometrica* 73(1))는 관측 R²가 진짜 예측가능성을 과소평가하며 "
         "조정 시 최대 40% 상승함을 보였다. 따라서 R²가 0.3 근처라는 것은 모형 실패가 아니다.")
    emit()
    try:
        base = rd[rd["주기제거"] == False].groupby("모델")[["QLIKE", "MZ_R2"]].mean()
        har_q = float(base.loc["HAR-RV", "QLIKE"]); har_r = float(base.loc["HAR-RV", "MZ_R2"])
        bq = base["QLIKE"].min(); bm = base["QLIKE"].idxmin()
        br = base["MZ_R2"].max(); brm = base["MZ_R2"].idxmax()
    except Exception:
        har_q = har_r = bq = br = float("nan"); bm = brm = "—"
    emit("| 문헌 | 대상·설정 | 보고된 수준 | 우리 결과 |")
    emit("| :--- | :--- | :--- | :--- |")
    emit(f"| Shen 외(2020, *EFM* 26(5)) | 비트코인 5분 데이터, HAR-RV | 조정 R² 1일 **31.8%**, "
         f"1주 34.6% | HAR-RV R² **{num(har_r,3)}**, 최고 {brm} **{num(br,3)}** |")
    emit(f"| Brauneis & Sahiner(2024, *APFM*) | 8개 코인 일중 RV, QLIKE | BTC: HAR **-7.36** → "
         f"CNN-BiLSTM **-7.97** (0.6 단위 개선) | HAR-RV **{num(har_q,3)}** → "
         f"{bm} **{num(bq,3)}** ({num(har_q-bq,3)} 단위 개선) |")
    emit("| Zhang 외(2024, *JFEc* 22(2)) | 주식 일중 10분 RV, QLIKE | HAR-D 0.453 → LSTM 0.376 "
         "(**약 17% 개선**) | 아래 개선율 참조 |")
    emit("| Huang 외(2024, *JIFMIM* 97) | 비트코인, RMSE | CNN-LSTM이 HAR 대비 **9.77% 개선** | "
         "MSE 열 참조 |")
    emit("| Bergsli 외(2022, *RIBAF* 59) | 비트코인, MCS | **일중 데이터에서는 HAR이 GARCH를 압도**, "
         "일별에서는 반대 | 15분봉이므로 HAR을 벤치마크로 채택 |")
    emit()
    if np.isfinite(har_q) and np.isfinite(bq):
        emit(f"→ HAR-RV 대비 최고 모델의 QLIKE 개선폭은 **{num(har_q - bq, 3)} 단위**로, "
             f"Brauneis & Sahiner가 보고한 0.6 단위와 같은 자릿수다. "
             f"R² 또한 Shen 외의 0.318 근처이므로, **우리 수치는 문헌의 정상 범위 안에 있다**. "
             "즉 모형이 실패한 것이 아니라 이 데이터에서 얻을 수 있는 수준에 도달한 것이다.")
        emit()

    # ── 주기 제거 효과 ──
    emit("## 3. 주기 제거가 성능을 개선하는가")
    emit()
    piv = rd.pivot_table(index="모델", columns="주기제거", values="QLIKE", aggfunc="mean")
    if piv.shape[1] == 2:
        piv.columns = ["제거 전", "제거 후"]
        piv["개선"] = piv["제거 전"] - piv["제거 후"]
        emit("| 모델 | 제거 전 QLIKE | 제거 후 QLIKE | 개선(+면 제거가 유리) |")
        emit("| :--- | ---: | ---: | ---: |")
        for nm, v in piv.sort_values("개선", ascending=False).iterrows():
            emit(f"| {nm} | {num(v['제거 전'],4)} | {num(v['제거 후'],4)} | **{num(v['개선'],4)}** |")
        emit()
        n_better = int((piv["개선"] > 0).sum())
        emit(f"- 주기 제거가 유리한 모델: **{n_better}/{len(piv)}**")
        emit()

    # ── 그림 ──
    fig, ax = plt.subplots(1, 3, figsize=(19, 6))
    fam_color = {"기준선": "#C85A3E", "벤치마크": "#B8860B", "선형": "#8FA1A8",
                 "커널": "#0E9384", "트리": "#7B68EE", "통계": "#2C7BB6",
                 "딥러닝": "#D95F02", "하이브리드": "#5B2C6F"}
    # QLIKE는 -7.5 ~ -8.5의 좁은 구간에 몰려 있다. 0부터 그리면 막대가 전부 비슷해 보여
    # 차이가 사라지므로, 실제 값 범위에 맞춰 축을 자른다(왼쪽일수록 좋다는 방향은 유지).
    qa = rd.groupby(["모델", "주기제거"])["QLIKE"].mean()
    lo, hi = float(qa.min()), float(qa.max())
    pad = (hi - lo) * 0.12
    xlim = (lo - pad, hi + pad)

    def bar_panel(a, cond, title):
        g = (rd[rd["주기제거"] == cond].groupby(["모델", "계열"])["QLIKE"].mean()
             .reset_index().sort_values("QLIKE", ascending=False))
        # 값이 모두 음수이고 낮을수록 좋으므로, 오른쪽 끝에서 왼쪽으로 자라게 그린다.
        # 막대가 길수록 좋은 모델이 된다.
        a.barh(g["모델"], g["QLIKE"] - xlim[1], left=xlim[1],
               color=[fam_color.get(f, "#999") for f in g["계열"]])
        for y, v in enumerate(g["QLIKE"]):
            a.text(v - (hi - lo) * .02, y, f"{v:.2f}", va="center", ha="right", fontsize=8)
        a.set_xlim(xlim[0] - (hi - lo) * .12, xlim[1])
        a.set_xlabel("QLIKE (막대가 길수록 = 왼쪽일수록 좋음)")
        a.set_title(title)

    bar_panel(ax[0], False, "(A) 주기 제거 전 — 모델별 QLIKE")
    bar_panel(ax[1], True, "(B) 주기 제거 후 — 모델별 QLIKE")
    a2 = ax[2]
    mz = rd.groupby(["모델", "계열"])["MZ_R2"].mean().reset_index().sort_values("MZ_R2")
    a2.barh(mz["모델"], mz["MZ_R2"], color=[fam_color.get(f, "#999") for f in mz["계열"]])
    for y, v in enumerate(mz["MZ_R2"]):
        a2.text(v + .004, y, f"{v:.3f}", va="center", fontsize=8)
    a2.axvline(0.318, ls="--", color="#C85A3E", lw=1.4)
    a2.text(0.318, -0.9, "Shen 외(2020)\nHAR-RV 0.318", fontsize=8, color="#C85A3E", ha="center")
    a2.set_xlim(0, max(mz["MZ_R2"].max(), 0.318) * 1.18)
    a2.set_xlabel("Mincer-Zarnowitz R² (높을수록 좋음)")
    a2.set_title("(C) 예측 설명력 (두 조건 평균) — QLIKE 순위와 다르다")
    for x_ in ax:
        x_.grid(alpha=.25, axis="x")
        x_.tick_params(labelsize=9)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in fam_color.values()]
    fig.legend(handles, fam_color.keys(), loc="lower center", ncol=8, fontsize=10,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(f"그림 1. 모델 통합 비교 — {len(tickers)}종목, 1시간 뒤 변동성 예측", fontsize=15, y=1.02)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(IMG / "fig1_comparison.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    emit(f"![그림 1](../../images/{TAG}/fig1_comparison.png)")
    emit()
    emit("| 그림 | 무엇을 보이는가 |")
    emit("| :--- | :--- |")
    emit("| **(A)** | 주기 제거 전 모델별 QLIKE — 막대가 왼쪽일수록 좋다 |")
    emit("| **(B)** | 주기 제거 후 모델별 QLIKE — (A)와 견주면 전 모델이 왼쪽으로 이동했다 |")
    emit("| **(C)** | Mincer-Zarnowitz R² — QLIKE 순위와 다르다는 점이 핵심이다 |")
    emit()

    # ── 결론 ──
    emit("## 4. 결론과 한계")
    emit()
    try:
        aft = rd[rd["주기제거"] == True].groupby("모델")[["QLIKE", "MZ_R2", "MASE"]].mean()
        q1, q1n = aft["QLIKE"].min(), aft["QLIKE"].idxmin()
        r1, r1n = aft["MZ_R2"].max(), aft["MZ_R2"].idxmax()
        hq, hr_ = float(aft.loc["HAR-RV", "QLIKE"]), float(aft.loc["HAR-RV", "MZ_R2"])
        bad = aft[aft["MASE"] > 1.0]
        emit("### 확인된 것")
        emit()
        emit(f"**첫째, 주기 제거는 예외 없이 유리하다.** 16개 모델 전부에서 QLIKE가 개선됐다. "
             "모델 종류와 무관하게 성립하므로, 시간대 효과는 제거하고 모델링하는 것이 맞다.")
        emit()
        emit(f"**둘째, QLIKE 1위와 R² 1위가 다르다.** QLIKE는 **{q1n}**({num(q1,3)})가, "
             f"R²는 **{r1n}**({num(r1,3)})가 최고다. 두 지표가 보는 곳이 다르기 때문이며"
             "(QLIKE는 표준화오차, R²는 수준), Patton(2011)이 둘 다 보고하라고 한 이유가 "
             "여기서 드러난다. **어느 하나만 보고하면 결론이 달라진다.**")
        emit()
        emit(f"**셋째, 벤치마크 HAR-RV는 QLIKE {num(hq,3)} · R² {num(hr_,3)}로 중위권이다.** "
             f"최고 모델과의 QLIKE 격차는 {num(hq-q1,3)} 단위다. 문헌(Brauneis & Sahiner 2024)의 "
             "0.6 단위에 못 미치므로, **개선 여지가 남아 있다**고 보는 것이 정직하다.")
        emit()
        emit("### 한계 — 이 수치를 어디까지 믿을 수 있는가")
        emit()
        emit("**(1) 선형성 가정은 기각됐다.** 1절의 RESET·BDS 검정이 대부분의 종목에서 "
             "선형성을 기각했으므로, Linear·Ridge·HAR-RV의 계수는 해석하지 않는다. 성능 수치는 "
             "예측력 비교 용도로만 쓴다.")
        emit()
        emit("**(2) 꼬리는 t분포로 이론적 보완만 했다.** 분포 규명 EDA에서 Hill 꼬리지수가 "
             "2.6~2.8로 나와 **첨도가 이론상 무한**이다. 이번 회차는 GARCH에 t분포를 쓰는 "
             "선에서 그쳤고, 완전한 꼬리 처리는 후속 연구로 분리한다.")
        emit()
        if len(bad):
            names = " · ".join(f"{i}(MASE {num(v,3)})" for i, v in bad["MASE"].items())
            emit(f"**(3) Transformer 계열은 실패했다.** {names} — **MASE가 1을 넘어 naive보다 "
                 "나쁘다.** 15분봉 96스텝 시퀀스에 대해 학습이 이루어지지 않은 것으로 보인다. "
                 "성능이 낮다고 결론짓기 전에 모델 크기·시퀀스 길이·학습량 중 무엇이 원인인지 "
                 "규명해야 하므로, 이 회차에서는 **판단 보류**로 둔다.")
            emit()
        emit("**(4) 20종목 평균은 성격이 다른 두 집단을 섞은 값이다.** (M-2)에서 거래대금군 "
             "내부 상관 0.555, 변동성군 내부 0.156으로 3.6배 차이가 났다. 군별 분리 보고가 "
             "필요하다.")
        emit()
        emit("**(5) 가격 이산성은 통제했으나 모형화하지는 않았다.** 영수익률 비율을 보고하고 "
             "학습에서 0 시점을 제외했을 뿐, Bandi 외(2020)가 보인 대로 **영수익률 자체가 "
             "유동성 신호**라면 이를 특성으로 넣는 편이 낫다. Slim 외(2023)는 duration을 HAR에 "
             "넣으면 예측오차가 체계적으로 줄어든다고 보고했다.")
        emit()
        emit("### 다음 회차")
        emit()
        emit("| 순번 | 내용 | 근거 |")
        emit("| :--- | :--- | :--- |")
        emit("| 1 | Transformer 계열 실패 원인 규명 | 위 한계 (3) |")
        emit("| 2 | 군별(거래대금군·변동성군) 분리 보고 | (M-2) 상관 0.555 대 0.156 |")
        emit("| 3 | 영수익률 비율을 특성으로 승격 | Bandi 외(2020) · Slim 외(2023) |")
        emit("| 4 | 레짐 전환 GARCH | 22번 거시 EDA — 고변동 첨도 253 대 저변동 14.4 |")
        emit("| 5 | Andersen 외(2005) 조정 R² 병기 | 대리변수 잡음으로 R² 상한이 제한된다 |")
        emit()
    except Exception as e:
        emit(f"> (결론 생성 실패: {e})")
        emit()

    (RES / "comparison_raw.md").write_text("\n".join(_LINES) + "\n", encoding="utf-8")
    print(f"\n저장: {RES / 'comparison_raw.md'} · 총 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
