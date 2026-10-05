# %% [markdown]
# # 26번: 시간 기준 평가 재정립과 24번 재평가
#
# 24·25번 점검(2026-10-05)에서 평가 틀 자체의 결함이 확인됐다. 이 회차는 모델을 바꾸지 않고
# **평가 틀만 고쳐** 24번의 모델 16종(naive 포함)을 다시 평가한다. 바뀌는 점은 아래와 같다.
#
# | 24번의 문제 | 이 회차의 처리 |
# | :--- | :--- |
# | 행 기준(4행=1시간). 업비트는 체결이 없으면 캔들을 만들지 않아 저유동 종목은 평가 창의 최대 45%가 60분을 넘었다 | 15분 시간 격자로 복원. 무체결 봉은 가격 불변(수익률 0), 거래소 점검 구간은 제외 |
# | 종목마다 유효 행 70% 지점에서 분할해 평가 날짜가 달랐다 | 전 종목 공통 분할 시각 `SPLIT` |
# | 타깃은 r[i+1..i+4]인데 특성은 r[i-1]까지, naive·GRU/LSTM은 r[i]까지 봤다 | 예측 시점 T의 정보는 T 이전에 끝난 봉 전부. 전 모델 동일 |
# | GARCH는 r[i] 한 봉의 분산을 4배 환산했다 | GARCH 3종 모두 목표 창 전체의 다단계 예측 |
# | MS·TAR 엔진의 h는 분산이 아니라 t 척도 변수였다(분산 = h(ν-2)/ν, 약 1.5배 과대예측) | 분산 = κ·h로 환산하고 다단계 재귀도 E[x²]=κ·h로 |
# | 로그 타깃 모델은 학습 구간 85%로만 적합하고 재적합이 없었다 | 설정 선택 후 학습 구간 전체로 재적합 |
# | 주기 제거 조건은 타깃까지 주기 제거된 값이라 조건끼리 대상이 달랐다 | 타깃은 항상 실제 시장 변동성(원 스케일). 주기 제거는 입력에만 쓰고 예측을 원 스케일로 되돌린다 |
# | 주기 스케일·딥러닝 표준화 통계에 평가 구간이 섞였다 | 학습 구간만으로 추정 |
# | 로그 타깃 모델은 exp 역변환으로 기하평균을 내 체계적으로 낮게 예측했다(QLIKE는 과소예측을 크게 벌함) | 내부검증 구간에서 QLIKE 최적 스케일 상수를 추정해 보정. 보정 전 값도 함께 기록 |
# | 학습 표본을 입력 조건(pas<=0)으로도 걸렀다 | 타깃 RV>0 조건만 |
# | 구간을 사후(실현 RV)로 나눴다(예측자의 딜레마) | 구간은 사전 변수(직전 H시간 RV, 학습 구간 분위수)로만. 실현 RV 구간은 기술 통계로만 |
# | 예측 구간이 1시간 하나였다 | 15분·30분·1시간·4시간·12시간 |
#
# ## 시각 규약
#
# 업비트 캔들의 타임스탬프는 캔들 **시작 시각**(KST)이다(pyupbit `candle_date_time_kst`).
# 라벨 s의 봉은 [s, s+15분)이고 그 수익률 r[s] = log(종가_s / 종가_{s-15분})다. 예측 시점 T(정시)에는
# 라벨이 T보다 작은 봉까지 알 수 있다. H시간 타깃은 라벨 T, T+15분, …, T+H시간-15분인 4H개 봉의
# 실현변동성 sqrt(Σ r²)이고, naive는 직전 4H개 봉의 같은 값이다.
#
# 결측 처리의 근거와 데이터 정의는 `test/research_materials/data_definition.md`를 본다.

# %%
from __future__ import annotations

import argparse
import gc
import importlib.util
import os
import sys
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

warnings.filterwarnings("ignore")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn as nn

matplotlib.rcParams["font.family"] = ["Noto Sans CJK KR", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


def _project_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "engine").is_dir() and (p / "AGENTS.md").exists():
            return p
    return start


ROOT = _project_root(Path(__file__).resolve())
for _p in (ROOT, ROOT / "test" / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

TAG = "26_timebased_reeval_20261005"
STEM = "26_timebased_reeval"
# 시드 반복 실행: RUN26_SEED>0이면 무작위성이 있는 모델만 다시 학습하고 산출물 접두사에 시드를 붙인다.
# 무작위성이 없는 모델은 결정성 확인용으로 GARCH-t·HAR-RV·Linear·Ridge만 함께 돌린다.
SEED = int(os.environ.get("RUN26_SEED", "0"))
RUN_STEM = STEM if SEED == 0 else f"{STEM}_seed{SEED}"
SEED_SKIP = ("MS-GARCH", "TAR-GARCH", "KernelRidge-RBF", "SVR-RBF")
IMG = ROOT / "test" / "images" / TAG
RES = ROOT / "test" / "results" / TAG


def _load_m23():
    path = ROOT / "test" / "models" / "23_volatility_model_comparison_test.py"
    spec = importlib.util.spec_from_file_location("m23_for26", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M23 = _load_m23()

# %% [markdown]
# ## 설정

# %%
BAR = pd.Timedelta("15min")
BPH = 4
HORIZONS_H = (15, 30, 60, 240, 720)          # 예측 구간(분). 15분봉이라 15분 미만은 만들 수 없다
EVAL_HOURS = {15: tuple(range(24)), 30: tuple(range(24)), 60: tuple(range(24)),
              240: tuple(range(0, 24, 4)), 720: (0, 12)}


def hlabel(H: int) -> str:
    return f"{H}분" if H < 60 else f"{H // 60}시간"
SPLIT = pd.Timestamp("2025-08-26 00:00:00")
WARMUP = pd.Timedelta(days=7)
INNER_FRAC = 0.85
HALT_REF = ("KRW-BTC", "KRW-ETH", "KRW-XRP")
TAR_SWITCH_BARS = 4

RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
DL_HIDDEN = 128
DL_MAX_EPOCHS = 60
DL_PATIENCE = 10
DL_BATCH = 2048
DL_LR_GRID = (2e-3, 5e-4, 1e-4)
SEQ_LEN = 96
TREE_MAX_ROUNDS = 4000
TREE_LR = 0.02
TREE_EARLY_STOP = 200
TREE_THREADS = 4
KERNEL_MEM_BUDGET_GB = 1.5
SVR_MAX_N = 8000
NYSTROEM_COMPONENTS = 2048
MS_RESTARTS, MS_MAXITER, MS_FIT_N = 3, 1400, 30000
TAR_RESTARTS, TAR_MAXITER = 2, 600
TAR_TAU_Q = (0.5, 0.65, 0.8, 0.9)
MIN_FREE_RAM_GB = 3.0

# 2026-09-07 가정 진단 결정: 선형성 검정(RESET·BDS)이 기각되어 선형 모델(Linear·Ridge)은 비선형·커널 계열로
# 대체한다(assumption_diagnosis_report_20260907.md 8절 3번). 23~26번 구현이 이 결정을 반영하지 않고 두 모델을
# 계속 적합해 왔으므로, 이미 계산된 값은 기록으로만 두고 순위·동률·검정·결론에서는 뺀다.
DECISION_EXCLUDED = ("Linear", "Ridge")
LOG_TARGET_MODELS = ("HAR-RV", "Linear", "Ridge", "KernelRidge-RBF", "SVR-RBF", "Nystroem+Ridge",
                     "LightGBM", "XGBoost", "HistGBM", "GARCH+LightGBM", "GRU", "LSTM")
VAR_MODELS = ("GARCH-t", "MS-GARCH", "TAR-GARCH")
ALL_MODELS = ("naive",) + VAR_MODELS + LOG_TARGET_MODELS
FAMILY = {"naive": "기준선", "HAR-RV": "벤치마크", "Linear": "선형", "Ridge": "선형",
          "KernelRidge-RBF": "커널", "SVR-RBF": "커널", "Nystroem+Ridge": "커널",
          "LightGBM": "트리", "XGBoost": "트리", "HistGBM": "트리", "GARCH+LightGBM": "하이브리드",
          "GARCH-t": "통계", "MS-GARCH": "통계", "TAR-GARCH": "통계", "GRU": "딥러닝", "LSTM": "딥러닝"}
PROCESS = {"naive": "직전값", "HAR-RV": "지연 특성 회귀(비신경)", "Linear": "특성 기반 회귀(비신경)",
           "Ridge": "특성 기반 회귀(비신경)", "KernelRidge-RBF": "특성 기반 회귀(비신경)",
           "SVR-RBF": "특성 기반 회귀(비신경)", "Nystroem+Ridge": "특성 기반 회귀(비신경)",
           "LightGBM": "특성 기반 트리(비신경)", "XGBoost": "특성 기반 트리(비신경)",
           "HistGBM": "특성 기반 트리(비신경)", "GARCH+LightGBM": "순차·재귀 통계 + 트리 결합",
           "GARCH-t": "순차·재귀(통계)", "MS-GARCH": "순차·재귀(통계)", "TAR-GARCH": "순차·재귀(통계)",
           "GRU": "순차·재귀", "LSTM": "순차·재귀"}

_LINES: list[str] = []


def emit(t: str = "") -> None:
    print(t, flush=True)
    _LINES.append(t)


def free_ram_gb() -> float:
    return psutil.virtual_memory().available / 1024 ** 3


def relieve_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def kernel_max_n(budget_gb: float = KERNEL_MEM_BUDGET_GB) -> int:
    return int(np.sqrt(budget_gb * 1024 ** 3 / 8))


# %% [markdown]
# ## 데이터: 시간 격자 복원
#
# 점검 구간은 유동성 상위 3종목(BTC·ETH·XRP)이 **모두** 캔들이 없는 라벨로 정의한다. 이 세
# 종목은 15분 동안 체결이 없을 수 없으므로, 셋이 동시에 빈 시각은 거래소 전체 중단이다.
# 2025-01-01 공백은 업비트 공지(02:00~08:00 전 마켓 거래 중단)와 일치함을 확인했다.

# %%
_HALTS: pd.DatetimeIndex | None = None


def halt_labels() -> pd.DatetimeIndex:
    global _HALTS
    if _HALTS is not None:
        return _HALTS
    missing = []
    for tk in HALT_REF:
        c = M23.load_close(tk)
        g = pd.date_range(c.index.min(), c.index.max(), freq=BAR)
        missing.append(set(g.difference(c.index)))
    _HALTS = pd.DatetimeIndex(sorted(set.intersection(*missing)))
    return _HALTS


def build_from_close(close: pd.Series, halts: pd.DatetimeIndex, split: pd.Timestamp = SPLIT) -> dict:
    """종가 시계열 → 15분 격자 위의 수익률·주기 스케일. 순수 함수라 자체 시험에 그대로 쓴다."""
    close = close.sort_index()
    grid = pd.date_range(close.index.min(), close.index.max(), freq=BAR)
    if not close.index.isin(grid).all():
        raise ValueError("15분 격자에 맞지 않는 타임스탬프가 있다")
    c = close.reindex(grid)
    traded = c.notna().to_numpy()
    is_halt = grid.isin(halts) & ~traded
    halt_but_traded = int((grid.isin(halts) & traded).sum())
    cf = c.ffill().to_numpy(dtype=float)
    n = len(grid)
    r = np.full(n, np.nan)
    r[1:] = np.log(cf[1:] / cf[:-1])
    bad = np.zeros(n, bool)
    bad[0] = True
    bad |= is_halt
    # 점검이 끝난 뒤 첫 체결 봉까지 제외한다. 그 사이 무체결 봉은 가격을 모르고, 첫 체결 봉의
    # 수익률은 점검 동안 쌓인 변화를 한 봉에 몰아 담는다(재개 직후 첫 봉이 무체결인 저유동 종목).
    pending = False
    for i in range(n):
        if is_halt[i]:
            pending = True
        elif pending:
            bad[i] = True
            if traded[i]:
                pending = False
    r[bad] = np.nan

    # 주기 스케일: 슬롯별 RMS 비율. 원 스케일 복원이 E[r²] ∝ c²를 가정하므로 |r| 평균이 아닌
    # 제곱평균으로 맞춘다(영수익률 비중이 슬롯마다 다른 저유동 종목에서 차이가 난다).
    slot = (grid.dayofweek * 24 + grid.hour).to_numpy()
    tr_bar = (grid < split) & np.isfinite(r)
    r2 = r ** 2
    slot_ms = pd.Series(r2[tr_bar]).groupby(slot[tr_bar]).mean()
    overall = float(np.mean(r2[tr_bar]))
    cfac = np.sqrt((slot_ms.reindex(np.arange(168)).fillna(overall) / overall).to_numpy())[slot]
    d = r / cfac
    return dict(grid=grid, r=r, d=d, cfac=cfac, bad=~np.isfinite(r), traded=traded,
                is_halt=is_halt, no_trade=~traded & ~is_halt, halt_but_traded=halt_but_traded)


def build_data(ticker: str) -> dict:
    D = build_from_close(M23.load_close(ticker), halt_labels())
    D["ticker"] = ticker
    return D


def _ns(idx: pd.DatetimeIndex) -> np.ndarray:
    """해상도(pandas 3은 기본 us)와 무관하게 ns 정수로 저장한다."""
    return np.asarray(idx.values.astype("datetime64[ns]").astype(np.int64))


def _csum(x: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(np.nan_to_num(x, nan=0.0))])


def horizon_data(D: dict, H: int, split: pd.Timestamp = SPLIT) -> dict:
    """한 예측 구간 H(시간)의 예측 시점·타깃·naive·분할을 만든다."""
    grid, r, d, cfac, bad = D["grid"], D["r"], D["d"], D["cfac"], D["bad"]
    n = len(grid)
    m = H // 15
    cs_r2, cs_d2, cs_c2 = _csum(r ** 2), _csum(d ** 2), _csum(cfac ** 2)
    cs_bad = _csum(bad.astype(float))
    j = np.arange(m, n - m + 1)
    clean = (cs_bad[j + m] - cs_bad[j - m]) == 0
    j = j[clean]
    T = grid[j]
    j = j[T >= grid[0] + WARMUP]
    T = grid[j]
    act = np.sqrt(cs_r2[j + m] - cs_r2[j])
    act_d = np.sqrt(cs_d2[j + m] - cs_d2[j])
    nai = np.sqrt(cs_r2[j] - cs_r2[j - m])
    c2m = (cs_c2[j + m] - cs_c2[j]) / m
    end = T + pd.Timedelta(minutes=H)

    t0 = T[0]
    inner = t0 + (split - t0) * INNER_FRAC
    inner = inner.floor("h")
    tr = np.asarray(end <= split)
    tr_in = np.asarray(end <= inner)
    tr_val = np.asarray(T >= inner) & np.asarray(end <= split)
    te = np.asarray(T >= split) & (np.asarray(T.minute) == 0) & np.isin(np.asarray(T.hour), EVAL_HOURS[H])
    fit_ok = act_d > 0
    return dict(H=H, m=m, j=j, T=T, act=act, act_d=act_d, nai=nai, c2m=c2m,
                tr=tr, tr_in=tr_in & fit_ok, tr_val=tr_val & fit_ok,
                va=tr_val & (np.asarray(T.minute) == 0), tr_fit=tr & fit_ok,
                te=te, inner=inner, split=split)


# %% [markdown]
# ## 평가 지표
#
# QLIKE(Patton 2011 식 6)가 주 지표다. 실제값이 0이어도 log(h)로 유한하므로 실제값은 건드리지
# 않는다. 예측값이 0에 붙으면 발산하므로 BPQ(2016)의 insanity filter로 학습 표본의 [최솟값, 최댓값]
# 밖 예측을 학습 평균으로 바꾼다(24번은 0.5~99.5% 분위수를 써서 원 논문보다 공격적이었고, 평가
# 구간 수준이 오른 종목을 불리하게 만들었다). 손실이 유한하지 않으면 조용히 빼지 않고 예외를 낸다.

# %%
def insanity_filter(p: np.ndarray, lo: float, hi: float, mean_: float) -> tuple[np.ndarray, float]:
    p = np.asarray(p, dtype=float)
    badp = ~np.isfinite(p) | (p < lo) | (p > hi)
    return np.where(badp, mean_, p), float(np.mean(badp))


def qlike_vec(act: np.ndarray, pred: np.ndarray) -> np.ndarray:
    h = np.asarray(pred, float) ** 2
    v = np.log(h) + np.asarray(act, float) ** 2 / h
    if not np.all(np.isfinite(v)):
        raise FloatingPointError("QLIKE가 유한하지 않다: 예측값 0 또는 NaN")
    return v


def evaluate(act: np.ndarray, pred: np.ndarray, nai: np.ndarray) -> dict:
    out = {"QLIKE": float(qlike_vec(act, pred).mean()),
           "MSE": float(np.mean((act ** 2 - pred ** 2) ** 2))}
    X = np.column_stack([np.ones(len(pred)), pred])
    beta, *_ = np.linalg.lstsq(X, act, rcond=None)
    sst = float(np.sum((act - act.mean()) ** 2))
    out["MZ_R2"] = float(1 - np.sum((act - X @ beta) ** 2) / sst) if sst > 0 else np.nan
    out["MASE"] = float(np.mean(np.abs(act - pred)) / np.mean(np.abs(act - nai)))
    return out


def qlike_scale(act: np.ndarray, pred: np.ndarray) -> float:
    """QLIKE를 최소화하는 곱셈 상수 c(분산 스케일): d/dc Σ[log(c·p²)+a²/(c·p²)]=0 → c=mean(a²/p²)."""
    return float(np.mean(np.asarray(act, float) ** 2 / np.asarray(pred, float) ** 2))


def scaled_qlike(act: np.ndarray, pred: np.ndarray) -> float:
    """최적 상수 보정 후 QLIKE — 수준 편향을 뺀 예측 '모양'의 질. 하이퍼파라미터 선택 기준."""
    c = qlike_scale(act, pred)
    return float(np.log(c) + np.mean(np.log(np.asarray(pred, float) ** 2)) + 1.0)


# %% [markdown]
# ## GARCH 계열: 목표 창 전체의 다단계 예측
#
# GARCH 3종은 주기 제거 수익률 d를 점검 구간 없이 이어 붙인 수열에 적합한다(점검 직후 봉은 점검
# 시간 동안 쌓인 변화를 한 봉에 몰아 담으므로 수열에서 뺀다). 예측 시점 T의 첫 목표 봉 분산
# h₀는 T 이전 정보로 계산한 1스텝 예측이고, 이후 k스텝은 각 모형의 기대값 재귀로 구한다. 원 스케일
# 분산은 봉마다 주기 스케일 c²를 곱해 더한다(r = d·c이므로 정확하다).
#
# - GARCH(1,1)-t: E[h_k] = ω·Σ_{i<k}φⁱ + φᵏ·h₀, φ=α+β
# - MS-GARCH(Gray 축약): 관측이 없으면 필터 확률 = 예측 확률이므로 ξ_k = Pᵀξ_{k-1},
#   h_{i,k} = ω_i + (α_i+β_i)·h_{k-1}, h_k = Σ ξ_{k,i}·h_{i,k}
# - TAR-GARCH: 국면 변수(직전 1시간 RV)의 미래 값은 모르므로, 아는 봉은 실제 d², 모르는 봉은 예측
#   분산으로 채운 대입값으로 국면을 정한다(지시함수의 기대값이 아니라 대입 근사임을 밝힌다).

# %%
def _compact(D: dict, split: pd.Timestamp = SPLIT) -> tuple[np.ndarray, np.ndarray, int]:
    comp = np.where(np.isfinite(D["d"]))[0]
    dc = D["d"][comp]
    split_c = int(np.searchsorted(D["grid"][comp], split))
    return comp, dc, split_c


def _origin_pos(comp: np.ndarray, j0: np.ndarray, m: int) -> np.ndarray:
    pos = np.searchsorted(comp, j0)
    if not (np.all(comp[pos] == j0) and np.all(comp[pos + m - 1] == j0 + m - 1)):
        raise AssertionError("목표 창이 압축 수열에서 연속이 아니다")
    return pos


def garch_t_fit(dc: np.ndarray, split_c: int) -> dict:
    from arch import arch_model
    from scipy.signal import lfilter
    x = dc * 100.0
    res = arch_model(x[:split_c], mean="Zero", vol="GARCH", p=1, q=1, dist="t").fit(
        disp="off", show_warning=False)
    p = res.params
    om, al, be = float(p["omega"]), float(p["alpha[1]"]), float(p["beta[1]"])
    n = len(x)
    cv = np.full(n, np.nan)
    cv[:split_c] = np.asarray(res.conditional_volatility) ** 2
    drive = om + al * x[split_c - 1:n - 1] ** 2
    cv[split_c:], _ = lfilter([1.0], [1.0, -be], drive, zi=np.array([be * cv[split_c - 1]]))
    return dict(cv=cv, omega=om, alpha=al, beta=be, nu=float(p.get("nu", np.nan)))


def garch_multistep(G: dict, pos: np.ndarray, m: int) -> np.ndarray:
    """(n_origin, m) 봉별 분산 예측(퍼센트² 단위)."""
    phi = G["alpha"] + G["beta"]
    k = np.arange(m)
    geo = k.astype(float) if abs(phi - 1) < 1e-12 else (1 - phi ** k) / (1 - phi)
    h0 = G["cv"][pos][:, None]
    return G["omega"] * geo[None, :] + (phi ** k)[None, :] * h0


def ms_multistep(info: dict, pos: np.ndarray, m: int) -> np.ndarray:
    """봉별 **분산** 예측(퍼센트²). 엔진의 h는 척도 변수라 분산 = κ·h이고, E[x²] = κ·h로 재귀한다."""
    (o1, o2), (a1, a2), (b1, b2) = info["omega"], info["alpha"], info["beta"]
    p11, p22, kap = info["p11"], info["p22"], info["kappa"]
    out = np.empty((len(pos), m))
    h = info["h_pred"][pos].astype(float)
    x1 = info["xi_pred"][pos, 0].astype(float)
    x2 = info["xi_pred"][pos, 1].astype(float)
    out[:, 0] = kap * h
    for k in range(1, m):
        x1, x2 = p11 * x1 + (1 - p22) * x2, (1 - p11) * x1 + p22 * x2
        h1 = np.maximum(o1 + (a1 * kap + b1) * h, 1e-12)
        h2 = np.maximum(o2 + (a2 * kap + b2) * h, 1e-12)
        h = x1 * h1 + x2 * h2
        out[:, k] = kap * h
    return out


def tar_switch(dc: np.ndarray, nb: int = TAR_SWITCH_BARS) -> np.ndarray:
    """switch[t] = sqrt(Σ dc²[t-nb..t-1]) — t 이전 정보만."""
    cs = _csum(dc ** 2)
    t = np.arange(len(dc))
    lo = np.maximum(t - nb, 0)
    return np.sqrt(cs[t] - cs[lo])


def tar_multistep(info: dict, dc: np.ndarray, pos: np.ndarray, m: int, nb: int = TAR_SWITCH_BARS) -> np.ndarray:
    """봉별 **분산** 예측(퍼센트²). 분산 = κ·h, E[x²] = κ·h(엔진 h는 척도 변수)."""
    (o1, o2), (a1, a2), (b1, b2) = info["omega"], info["alpha"], info["beta"]
    tau, kap = info["tau"], info["kappa"]
    hs = np.empty((len(pos), m))
    hs[:, 0] = info["h_pred"][pos]
    known = np.stack([dc[np.maximum(pos - nb + i, 0)] ** 2 for i in range(nb)], axis=1)  # pos-nb..pos-1
    for k in range(1, m):
        parts = []
        for s in range(k - nb, k):
            parts.append(known[:, s + nb] if s < 0 else kap * hs[:, s] / 1e4)
        sw = np.sqrt(np.sum(parts, axis=0))
        hi = sw > tau
        o = np.where(hi, o2, o1); ab = np.where(hi, a2 * kap + b2, a1 * kap + b1)
        hs[:, k] = np.maximum(o + ab * hs[:, k - 1], 1e-12)
    return kap * hs


def var_to_rv(hpath_pct2: np.ndarray, cfac: np.ndarray, j0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """봉별 주기 제거 분산(퍼센트²) → (원 스케일 창 RV, 주기 제거 창 RV)."""
    m = hpath_pct2.shape[1]
    c2 = cfac[j0[:, None] + np.arange(m)[None, :]] ** 2
    raw = np.sqrt(np.sum(hpath_pct2 * c2, axis=1) / 1e4)
    dep = np.sqrt(np.sum(hpath_pct2, axis=1) / 1e4)
    return raw, dep


# %% [markdown]
# ## 특성: 행 = 시간
#
# 격자 위에서는 행 하나가 15분이므로 23번의 특성 함수(`shift(1)` 후 이동창)를 그대로 쓰면 창이
# 곧 시간창이 된다. 행 j의 특성은 r[..j-1]만 쓰고, 예측 시점 T(=행 j)의 타깃은 r[j..]부터이므로
# 정보 시점이 맞는다. 결측(점검)은 이동평균에서 건너뛴다.

# %%
def make_features(D: dict) -> dict:
    d = D["d"]
    return dict(log=M23.feats_log(d), har=M23.feats_har(d), tree=M23.feats_tree(d))


def make_sequences(D: dict, j0: np.ndarray, L: int, split: pd.Timestamp = SPLIT) -> np.ndarray:
    """예측 시점 행 j0마다 (d, |d|)[j0-L : j0]. 표준화 통계는 학습 구간 봉만으로."""
    d = np.nan_to_num(D["d"], nan=0.0)
    ch = np.column_stack([d, np.abs(d)]).astype(np.float32)
    trb = (D["grid"] < split) & np.isfinite(D["d"])
    mu, sd = ch[trb].mean(0), ch[trb].std(0)
    ch = (ch - mu) / np.where(sd < 1e-9, 1.0, sd)
    pad = np.zeros((L, 2), np.float32)
    padded = np.concatenate([pad, ch], axis=0)
    W = np.lib.stride_tricks.sliding_window_view(padded, (L, 2))[:, 0]
    # padded[k]=ch[k-L]이므로 W[j0]=padded[j0:j0+L]=ch[j0-L:j0]
    return np.ascontiguousarray(W[j0])


# %% [markdown]
# ## 한 (종목, 예측 구간)의 점수 기록기

# %%
class Scorer:
    def __init__(self, ticker: str, HD: dict):
        self.tk, self.HD = ticker, HD
        a_tr = HD["act"][HD["tr_fit"]]
        a_in = HD["act"][HD["tr_in"]]
        self.lo, self.hi, self.mean = float(a_tr.min()), float(a_tr.max()), float(a_tr.mean())
        self.ilo, self.ihi, self.imean = float(a_in.min()), float(a_in.max()), float(a_in.mean())
        self.te = HD["te"]
        self.act = HD["act"][self.te]
        self.nai = HD["nai"][self.te]
        self.nai_f, _ = insanity_filter(self.nai, self.lo, self.hi, self.mean)
        self.rows: list[dict] = []
        self.preds: dict[str, np.ndarray] = {}

    def to_raw(self, pred_logdep: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return np.exp(pred_logdep) * np.sqrt(self.HD["c2m"][mask])

    def calib(self, raw_va: np.ndarray) -> float:
        """내부검증 정시 시점(RV=0 포함, 평가와 같은 위상)에서 QLIKE 최적 분산 배율. 필터 → 배율 순서."""
        p, _ = insanity_filter(raw_va, self.ilo, self.ihi, self.imean)
        return qlike_scale(self.HD["act"][self.HD["va"]], p)

    def select_score(self, raw_va: np.ndarray) -> float:
        p, _ = insanity_filter(raw_va, self.ilo, self.ihi, self.imean)
        return scaled_qlike(self.HD["act"][self.HD["va"]], p)

    def score(self, name: str, raw_te: np.ndarray, c: float = 1.0, note: str = ""):
        if len(raw_te) != int(self.te.sum()):
            raise AssertionError(f"{name}: 예측 길이 불일치")
        p_raw, trim = insanity_filter(raw_te, self.lo, self.hi, self.mean)
        p_cal = p_raw * np.sqrt(c)
        row = {"종목": self.tk, "H": self.HD["H"], "모델": name, "계열": FAMILY[name],
               "처리방식": PROCESS[name], "보정계수": c, "트리밍비율": trim, "n": len(p_cal), "비고": note,
               **evaluate(self.act, p_cal, self.nai_f),
               "QLIKE_보정전": float(qlike_vec(self.act, p_raw).mean())}
        self.rows.append(row)
        self.preds[name] = p_cal.astype(np.float32)
        return row


# %% [markdown]
# ## CPU 작업: 종목 하나의 GARCH 3종(1회 적합) + 구간별 비신경 모델

# %%
def run_cpu_job(ticker: str, quick: bool) -> dict:
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.svm import SVR
    from sklearn.kernel_approximation import Nystroem
    from sklearn.pipeline import make_pipeline
    from sklearn.ensemble import HistGradientBoostingRegressor
    import lightgbm as lgb
    import xgboost as xgb
    from engine import regime_garch as rg

    t0 = time.time()
    D = build_data(ticker)
    F = make_features(D)
    comp, dc, split_c = _compact(D)
    fails, timings, meta, results = [], {}, {"종목": ticker}, []

    G = ms_info = tar_info = None
    seed_mode = SEED != 0
    t = time.time()
    try:
        G = garch_t_fit(dc, split_c)
        meta.update(garch_alpha=G["alpha"], garch_beta=G["beta"], garch_nu=G["nu"])
    except Exception as e:
        fails.append(("GARCH-t", "*", type(e).__name__, str(e)[:160]))
    timings["GARCH-t"] = time.time() - t
    t = time.time()
    try:
        if seed_mode:
            raise LookupError("seed-skip")
        _, _, ms_info = rg.fit_ms_garch(dc * 100, split_c, n_restarts=MS_RESTARTS if not quick else 1,
                                        maxiter=MS_MAXITER if not quick else 120,
                                        max_fit_n=MS_FIT_N if not quick else 8000)
        if ms_info is None:
            raise RuntimeError("최종 필터가 발산(info=None)")
        meta.update(ms_p11=ms_info["p11"], ms_p22=ms_info["p22"])
    except Exception as e:
        ms_info = None
        if str(e) != "seed-skip":
            fails.append(("MS-GARCH", "*", type(e).__name__, str(e)[:160]))
    timings["MS-GARCH"] = time.time() - t
    t = time.time()
    try:
        if seed_mode:
            raise LookupError("seed-skip")
        sw = tar_switch(dc)
        taus = np.quantile(sw[TAR_SWITCH_BARS:split_c], list(TAR_TAU_Q))
        _, tau, _, tar_info = rg.fit_tar_garch(dc * 100, sw, split_c, taus,
                                               n_restarts=TAR_RESTARTS if not quick else 1,
                                               maxiter=TAR_MAXITER if not quick else 100)
        if tar_info is None:
            raise RuntimeError("최종 필터가 발산(info=None)")
        meta["tar_tau"] = float(tau)
    except Exception as e:
        tar_info = None
        if str(e) != "seed-skip":
            fails.append(("TAR-GARCH", "*", type(e).__name__, str(e)[:160]))
    timings["TAR-GARCH"] = time.time() - t

    for H in HORIZONS_H:
        HD = horizon_data(D, H)
        S = Scorer(ticker, HD)
        j, te, tr_in, tr_val, va, tf = HD["j"], HD["te"], HD["tr_in"], HD["tr_val"], HD["va"], HD["tr_fit"]
        m = HD["m"]
        S.score("naive", HD["nai"][te])

        pos_all = _origin_pos(comp, j, m)
        g_dep_all = None
        for nm, info, fn in (("GARCH-t", G, "g"), ("MS-GARCH", ms_info, "ms"), ("TAR-GARCH", tar_info, "tar")):
            if info is None:
                continue
            t = time.time()
            try:
                sel = te if fn != "g" else np.ones(len(j), bool)
                pos = pos_all[sel]
                hp = (garch_multistep(info, pos, m) if fn == "g" else
                      ms_multistep(info, pos, m) if fn == "ms" else tar_multistep(info, dc, pos, m))
                raw, dep = var_to_rv(hp, D["cfac"], j[sel])
                if fn == "g":
                    g_dep_all = dep
                    raw = raw[te]
                S.score(nm, raw, note="다단계 예측")
            except Exception as e:
                fails.append((nm, H, type(e).__name__, str(e)[:160]))
            timings[f"{nm}_h{H}"] = time.time() - t

        y = np.log(HD["act_d"].clip(1e-300))
        X = {k: v[j] for k, v in F.items()}

        def fit_log_model(name: str, fit_fn, note: str = ""):
            """fit_fn() → (내부학습 모델의 va 로그 예측, 학습 전체 재적합 모델의 평가 로그 예측).

            설정(정규화 상수·반복 수·에폭)은 내부학습/내부검증으로 고르고, 보정 상수 c는 내부학습
            모델의 va 예측으로 추정한다. 평가 예측은 고른 설정으로 학습 구간 전체(tr_fit)에 다시
            적합한 모델에서 낸다(GARCH 계열이 학습 구간 전체를 쓰는 것과 맞춘다).
            """
            t_ = time.time()
            if seed_mode and name in SEED_SKIP:
                return
            try:
                pv, pt = fit_fn()
                c = S.calib(S.to_raw(pv, va))
                S.score(name, S.to_raw(pt, te), c=c, note=note)
                meta[f"{name}_h{H}_calib"] = c
            except Exception as e:
                fails.append((name, H, type(e).__name__, str(e)[:160]))
                traceback.print_exc()
            timings[f"{name}_h{H}"] = time.time() - t_

        def fit_linear(Xs):
            def f():
                sc = StandardScaler().fit(Xs[tr_in])
                mdl = LinearRegression().fit(sc.transform(Xs[tr_in]), y[tr_in])
                pv = mdl.predict(sc.transform(Xs[va]))
                sc2 = StandardScaler().fit(Xs[tf])
                mdl2 = LinearRegression().fit(sc2.transform(Xs[tf]), y[tf])
                return pv, mdl2.predict(sc2.transform(Xs[te]))
            return f

        fit_log_model("HAR-RV", fit_linear(X["har"]))
        fit_log_model("Linear", fit_linear(X["log"]))

        def fit_ridge():
            sc = StandardScaler().fit(X["log"][tr_in])
            Zi, Zv = sc.transform(X["log"][tr_in]), sc.transform(X["log"][va])
            best = None
            for a in RIDGE_ALPHAS:
                pv = Ridge(alpha=a).fit(Zi, y[tr_in]).predict(Zv)
                q = S.select_score(S.to_raw(pv, va))
                if best is None or q < best[0]:
                    best = (q, a, pv)
            meta[f"ridge_alpha_h{H}"] = best[1]
            sc2 = StandardScaler().fit(X["log"][tf])
            mdl2 = Ridge(alpha=best[1]).fit(sc2.transform(X["log"][tf]), y[tf])
            return best[2], mdl2.predict(sc2.transform(X["log"][te]))
        fit_log_model("Ridge", fit_ridge)

        def std_block(mask):
            """mask 행으로 x·y 표준화기를 맞춘다. 반환: (x 변환기, y 평균, y 표준편차)."""
            sc = StandardScaler().fit(X["log"][mask])
            return sc, float(y[mask].mean()), float(y[mask].std())

        sci, ymi, ysi = std_block(tr_in)
        scf, ymf, ysf = std_block(tf)
        Zi, Zv = sci.transform(X["log"][tr_in]), sci.transform(X["log"][va])
        Zf, Zt = scf.transform(X["log"][tf]), scf.transform(X["log"][te])
        yi_s, yf_s = (y[tr_in] - ymi) / ysi, (y[tf] - ymf) / ysf

        def chunked(mdl, Z, chunk=4000):
            return np.concatenate([mdl.predict(Z[i:i + chunk]) for i in range(0, len(Z), chunk)])

        n_kr = min(kernel_max_n(), len(Zi)) if not quick else min(1500, len(Zi))
        n_svr = min(SVR_MAX_N, len(Zi)) if not quick else min(1200, len(Zi))

        def fit_kr():
            sl = slice(len(Zi) - n_kr, len(Zi))
            g0 = 1.0 / Zi.shape[1]
            best = None
            for a in (0.1, 1.0, 10.0):
                for g in (g0 / 4, g0, g0 * 4):
                    mdl = KernelRidge(kernel="rbf", alpha=a, gamma=g).fit(Zi[sl], yi_s[sl])
                    pv = chunked(mdl, Zv) * ysi + ymi
                    q = S.select_score(S.to_raw(pv, va))
                    if best is None or q < best[0]:
                        best = (q, (a, g), pv)
                    del mdl
            meta[f"kr_h{H}"] = best[1]
            slf = slice(len(Zf) - n_kr, len(Zf))
            mdl2 = KernelRidge(kernel="rbf", alpha=best[1][0], gamma=best[1][1]).fit(Zf[slf], yf_s[slf])
            return best[2], chunked(mdl2, Zt) * ysf + ymf
        fit_log_model("KernelRidge-RBF", fit_kr, f"n={n_kr}")

        def fit_svr():
            sl = slice(len(Zi) - n_svr, len(Zi))
            best = None
            for cc in (1.0, 10.0, 100.0):
                mdl = SVR(kernel="rbf", C=cc, epsilon=0.05, cache_size=1000).fit(Zi[sl], yi_s[sl])
                pv = chunked(mdl, Zv) * ysi + ymi
                q = S.select_score(S.to_raw(pv, va))
                if best is None or q < best[0]:
                    best = (q, cc, pv)
            meta[f"svr_C_h{H}"] = best[1]
            slf = slice(len(Zf) - n_svr, len(Zf))
            mdl2 = SVR(kernel="rbf", C=best[1], epsilon=0.05, cache_size=1000).fit(Zf[slf], yf_s[slf])
            return best[2], chunked(mdl2, Zt) * ysf + ymf
        fit_log_model("SVR-RBF", fit_svr, f"n={n_svr}")

        def fit_nys():
            nc = NYSTROEM_COMPONENTS if not quick else 256
            best = None
            for a in RIDGE_ALPHAS:
                mdl = make_pipeline(Nystroem(kernel="rbf", n_components=nc, random_state=SEED),
                                    Ridge(alpha=a)).fit(Zi, yi_s)
                pv = mdl.predict(Zv) * ysi + ymi
                q = S.select_score(S.to_raw(pv, va))
                if best is None or q < best[0]:
                    best = (q, a, pv)
            meta[f"nys_alpha_h{H}"] = best[1]
            mdl2 = make_pipeline(Nystroem(kernel="rbf", n_components=nc, random_state=SEED),
                                 Ridge(alpha=best[1])).fit(Zf, yf_s)
            return best[2], mdl2.predict(Zt) * ysf + ymf
        fit_log_model("Nystroem+Ridge", fit_nys)

        rounds = TREE_MAX_ROUNDS if not quick else 60
        esr = TREE_EARLY_STOP if not quick else 10

        def lgb_params(n_est):
            return dict(n_estimators=n_est, learning_rate=TREE_LR, max_depth=8, num_leaves=127,
                        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, min_child_samples=40,
                        n_jobs=TREE_THREADS, verbosity=-1, random_state=SEED)

        def lgb_on(Xt_, key):
            """Xt_: 예측 시점 전체 행의 트리 특성(내부용 열, 재적합용 열을 같은 행에 맞춘 배열 쌍)."""
            Xin_, Xfull_ = Xt_
            mdl = lgb.LGBMRegressor(**lgb_params(rounds))
            mdl.fit(Xin_[tr_in], y[tr_in], eval_set=[(Xin_[tr_val], y[tr_val])], eval_metric="l2",
                    callbacks=[lgb.early_stopping(esr, verbose=False)])
            best_n = int(mdl.best_iteration_ or rounds)
            meta[f"{key}_h{H}_rounds"] = best_n
            pv = mdl.predict(Xin_[va])
            mdl2 = lgb.LGBMRegressor(**lgb_params(best_n)).fit(Xfull_[tf], y[tf])
            return pv, mdl2.predict(Xfull_[te])
        fit_log_model("LightGBM", lambda: lgb_on((X["tree"], X["tree"]), "lgbm"))

        def fit_xgb():
            kw = dict(learning_rate=TREE_LR, max_depth=8, subsample=0.8, colsample_bytree=0.8,
                      min_child_weight=10, tree_method="hist", device="cpu", n_jobs=TREE_THREADS,
                      verbosity=0, random_state=SEED)
            Tx = X["tree"]
            mdl = xgb.XGBRegressor(n_estimators=rounds, early_stopping_rounds=esr, **kw)
            mdl.fit(Tx[tr_in], y[tr_in], eval_set=[(Tx[tr_val], y[tr_val])], verbose=False)
            best_n = int(getattr(mdl, "best_iteration", rounds - 1) or 0) + 1
            meta[f"xgb_h{H}_rounds"] = best_n
            pv = mdl.predict(Tx[va], iteration_range=(0, best_n))
            mdl2 = xgb.XGBRegressor(n_estimators=best_n, **kw).fit(Tx[tf], y[tf])
            return pv, mdl2.predict(Tx[te])
        fit_log_model("XGBoost", fit_xgb)

        def fit_hist():
            Tx = X["tree"]
            best_loss, best_it, pat, best_pv = np.inf, 0, 0, None
            step = max(50, rounds // 20)
            kw = dict(learning_rate=TREE_LR, max_depth=8, max_leaf_nodes=127, min_samples_leaf=40,
                      early_stopping=False, random_state=SEED)
            mdl = HistGradientBoostingRegressor(max_iter=step, warm_start=True, **kw)
            for it in range(step, rounds + 1, step):
                mdl.set_params(max_iter=it)
                mdl.fit(Tx[tr_in], y[tr_in])
                loss = float(np.mean((mdl.predict(Tx[tr_val]) - y[tr_val]) ** 2))
                if loss < best_loss - 1e-9:
                    best_loss, best_it, pat, best_pv = loss, it, 0, mdl.predict(Tx[va])
                else:
                    pat += 1
                    if pat >= 2:
                        break
            meta[f"hist_h{H}_rounds"] = best_it
            mdl2 = HistGradientBoostingRegressor(max_iter=best_it, **kw).fit(Tx[tf], y[tf])
            return best_pv, mdl2.predict(Tx[te])
        fit_log_model("HistGBM", fit_hist)

        if g_dep_all is not None:
            # 하이브리드의 GARCH 특성: 내부용은 내부학습 구간까지로 추정한 GARCH, 재적합용은 학습 전체 GARCH
            try:
                inner_c = int(np.searchsorted(D["grid"][comp], HD["inner"]))
                G_in = garch_t_fit(dc, inner_c)
                _, g_dep_in = var_to_rv(garch_multistep(G_in, pos_all, m), D["cfac"], j)
                Xin_ = np.column_stack([X["tree"], np.log(g_dep_in + 1e-12)])
                Xfull_ = np.column_stack([X["tree"], np.log(g_dep_all + 1e-12)])
                fit_log_model("GARCH+LightGBM", lambda: lgb_on((Xin_, Xfull_), "hyb"))
            except Exception as e:
                fails.append(("GARCH+LightGBM", H, type(e).__name__, str(e)[:160]))
        else:
            fails.append(("GARCH+LightGBM", H, "SkippedDependency", "GARCH-t 적합 실패"))

        results.append(dict(H=H, rows=S.rows, preds=S.preds, act=S.act.astype(np.float32),
                            nai=S.nai.astype(np.float32), T=_ns(HD["T"][te]),
                            nai_tr=HD["nai"][HD["tr"]].astype(np.float32),
                            eda=horizon_eda(D, HD)))
    return dict(ticker=ticker, results=results, fails=fails, timings=timings, meta=meta,
                data_eda=data_eda(D), elapsed=time.time() - t0)


def data_eda(D: dict) -> dict:
    g = D["grid"]
    te = g >= SPLIT
    return {"종목": D["ticker"], "시작": str(g[0]), "끝": str(g[-1]), "격자봉수": len(g),
            "점검봉": int(D["is_halt"].sum()), "점검중체결": D["halt_but_traded"],
            "무체결봉비율": float(D["no_trade"].mean()),
            "무체결봉비율_평가구간": float(D["no_trade"][te].mean()),
            "영수익률비율": float(np.mean(D["r"][np.isfinite(D["r"])] == 0))}


def horizon_eda(D: dict, HD: dict) -> dict:
    te = HD["te"]
    a = HD["act"][te]
    a_tr = HD["act"][HD["tr"]]
    return {"평가표본": int(te.sum()), "학습표본": int(HD["tr"].sum()),
            "학습제외_RV0": float(1 - HD["tr_fit"].sum() / max(HD["tr"].sum(), 1)),
            "평가RV0비율": float(np.mean(a == 0)),
            "학습RV중앙값": float(np.median(a_tr)), "평가RV중앙값": float(np.median(a)),
            "평가RV95": float(np.quantile(a, .95)), "평가RV최대": float(a.max())}


# %% [markdown]
# ## GPU 작업: GRU·LSTM(구간별 재학습)

# %%
def train_dl_once(name, Xtr, ytr, Xval, yval, Xva, Xte, dev, max_epochs, patience, batch, lr, tmax=None, seed=0):
    from engine.models import make_model
    torch.manual_seed(seed)
    model = make_model(name, Xtr.shape[1], Xtr.shape[2], DL_HIDDEN).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tmax or max_epochs)
    lossf = nn.MSELoss()
    amp = dev == "cuda"
    xt = torch.as_tensor(Xtr, device=dev); yt = torch.as_tensor(ytr, dtype=torch.float32, device=dev)
    xv = None if Xval is None else torch.as_tensor(Xval, device=dev)
    yv = None if yval is None else torch.as_tensor(yval, dtype=torch.float32, device=dev)

    def infer(x, bs):
        model.eval()
        out = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            for i in range(0, len(x), bs):
                out.append(model(x[i:i + bs]).float())
        return torch.cat(out)

    best_loss, best_state, best_epoch, bad, bs, epoch = np.inf, None, 0, 0, batch, 0
    while epoch < max_epochs:
        model.train()
        snap = ({k: v.detach().clone() for k, v in model.state_dict().items()},
                {k: v for k, v in opt.state_dict().items()})
        perm = torch.randperm(len(xt), device=dev)
        try:
            for i in range(0, len(xt), bs):
                b = perm[i:i + bs]
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                    loss = lossf(model(xt[b]), yt[b])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
        except torch.cuda.OutOfMemoryError:
            if bs <= 128:
                raise
            bs //= 2
            model.load_state_dict(snap[0]); opt.load_state_dict(snap[1])
            relieve_memory()
            continue
        sched.step()
        epoch += 1
        if xv is None:
            continue
        vl = float(lossf(infer(xv, bs), yv).item())
        if vl < best_loss - 1e-6:
            best_loss, best_epoch, bad = vl, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    pv = None if Xva is None else infer(torch.as_tensor(Xva, device=dev), bs).cpu().numpy().ravel()
    pt = infer(torch.as_tensor(Xte, device=dev), bs).cpu().numpy().ravel()
    del xt, yt, xv, yv, model, best_state
    relieve_memory()
    return pv, pt, best_loss, dict(epochs_run=epoch, best_epoch=best_epoch, batch=bs, lr=lr)


def run_gpu_job(ticker: str, H: int, quick: bool, dev: str) -> dict:
    t0 = time.time()
    D = build_data(ticker)
    HD = horizon_data(D, H)
    S = Scorer(ticker, HD)
    j, te, tr_in, tr_val, va, tf = HD["j"], HD["te"], HD["tr_in"], HD["tr_val"], HD["va"], HD["tr_fit"]
    y = np.log(HD["act_d"].clip(1e-300))
    ymu, ysd = float(y[tr_in].mean()), float(y[tr_in].std())
    L = SEQ_LEN if not quick else 16
    Xs = make_sequences(D, j, L)
    fails, meta = [], {"종목": ticker, "H": H}
    for nm in ("GRU", "LSTM"):
        try:
            grid_lr = DL_LR_GRID if not quick else (DL_LR_GRID[0],)
            best = None
            for lr in grid_lr:
                pv, pt, vl, info = train_dl_once(
                    nm, Xs[tr_in], (y[tr_in] - ymu) / ysd, Xs[tr_val], (y[tr_val] - ymu) / ysd, Xs[va], Xs[te],
                    dev, DL_MAX_EPOCHS if not quick else 3, DL_PATIENCE if not quick else 2, DL_BATCH, lr, seed=SEED)
                if best is None or vl < best[2]:
                    best = (pv, pt, vl, info)
            pv, _, _, info = best
            c = S.calib(S.to_raw(pv * ysd + ymu, va))
            # 고른 학습률·에폭으로 학습 구간 전체에 재적합(검증 없이 고정 에폭)
            ep = max(1, info["best_epoch"])
            _, pt, _, _ = train_dl_once(nm, Xs[tf], (y[tf] - ymu) / ysd, None, None, None, Xs[te],
                                        dev, ep, ep + 1, DL_BATCH, info["lr"],
                                        tmax=DL_MAX_EPOCHS if not quick else 3, seed=SEED)
            S.score(nm, S.to_raw(pt * ysd + ymu, te), c=c,
                    note=f"lr {info['lr']:g} · {info['best_epoch']}/{info['epochs_run']}에폭 · 전체 재적합")
            meta.update({f"{nm}_lr": info["lr"], f"{nm}_epoch": info["best_epoch"], f"{nm}_calib": c})
        except Exception as e:
            fails.append((nm, H, type(e).__name__, str(e)[:160]))
            traceback.print_exc()
        relieve_memory()
    return dict(ticker=ticker, H=H, rows=S.rows, preds=S.preds, fails=fails, meta=meta,
                elapsed=time.time() - t0)


# %% [markdown]
# ## 자체 시험
#
# 실제 데이터를 쓰기 전에 합성 데이터로 정렬·정보 시점·다단계 재귀·분할을 검증한다. 하나라도
# 실패하면 본 실행을 시작하지 않는다.

# %%
def selftest() -> None:
    rng = np.random.default_rng(1)
    n = 96 * 40
    idx = pd.date_range("2025-08-01", periods=n, freq=BAR)
    ret = rng.normal(0, 0.003, n)
    px = pd.Series(100 * np.exp(np.cumsum(ret)), index=idx)
    halt = idx[96 * 10 + 8: 96 * 10 + 8 + 16]
    notrade = idx[96 * 20 + 3: 96 * 20 + 6]
    halt2 = idx[96 * 30: 96 * 30 + 8]
    after2 = idx[96 * 30 + 8: 96 * 30 + 10]
    close = px.drop(halt).drop(notrade).drop(halt2).drop(after2)
    split = pd.Timestamp("2025-08-30 00:00")
    D = build_from_close(close, pd.DatetimeIndex(halt).append(pd.DatetimeIndex(halt2)), split=split)
    D["ticker"] = "SYN"
    g = D["grid"]
    assert len(g) == n, "격자 길이"
    assert D["is_halt"].sum() == 24 and D["no_trade"].sum() == 5, "점검·무체결 분류"
    k3 = 96 * 30
    assert np.all(~np.isfinite(D["r"][k3:k3 + 11])) and np.isfinite(D["r"][k3 + 11]), "점검 뒤 첫 체결 봉까지 제외"
    k = 96 * 10 + 8
    assert np.all(~np.isfinite(D["r"][k:k + 17])), "점검 봉과 재개 직후 봉은 제외"
    k2 = 96 * 20 + 3
    assert np.allclose(D["r"][k2:k2 + 3], 0.0), "무체결 봉 수익률 0"
    assert np.isclose(D["r"][k2 + 3], np.log(px.iloc[k2 + 3] / px.iloc[k2 - 1])), "무체결 뒤 첫 체결 수익률"

    D2 = build_from_close(close.where(close.index < split, close * np.exp(rng.normal(0, .5, len(close)))),
                          pd.DatetimeIndex(halt).append(pd.DatetimeIndex(halt2)), split=split)
    assert np.allclose(D["cfac"], D2["cfac"]), "주기 스케일은 학습 구간만으로 추정"

    for H in HORIZONS_H:
        HD = horizon_data(D, H, split=split)
        m = H // 15
        jj = HD["j"][5]
        assert np.isclose(HD["act"][5], np.sqrt(np.nansum(D["r"][jj:jj + m] ** 2))), "타깃 창"
        assert np.isclose(HD["nai"][5], np.sqrt(np.nansum(D["r"][jj - m:jj] ** 2))), "naive 창"
        assert g[jj + m - 1] - g[jj] == pd.Timedelta(minutes=H) - BAR, "타깃 창 = H분"
        assert np.all(HD["T"][HD["tr"]] + pd.Timedelta(minutes=H) <= split), "학습 타깃이 분할을 넘지 않음"
        tt = HD["T"][HD["te"]]
        assert np.all(tt >= split) and np.all(tt.minute == 0) and np.all(np.isin(tt.hour, EVAL_HOURS[H])), "평가 시점"
        assert np.all(np.diff(tt) >= pd.Timedelta(minutes=H)), "평가 창 비겹침"
        assert not np.any(np.isin(HD["j"], np.arange(k - m + 1, k + 17 + m))), "점검을 지나는 창 제외"
        assert np.isin(k - m, HD["j"]) or (g[k - m] < g[0] + WARMUP), "점검 직전까지 끝나는 창은 유지"

    F = make_features(D)
    jj = 96 * 35
    assert g[jj] >= split
    seq = make_sequences(D, np.array([jj]), 16, split=split)
    Dp = dict(D); dp = D["d"].copy(); dp[jj:] = dp[jj:] * 7 + 1; Dp["d"] = dp
    Fp = make_features(Dp)
    seqp = make_sequences(Dp, np.array([jj]), 16, split=split)
    for key in F:
        assert np.allclose(F[key][jj], Fp[key][jj]), f"특성 {key}에 미래 정보"
    assert np.array_equal(seq, seqp), "시퀀스에 미래 정보"
    Dq = dict(D); dq = D["d"].copy(); dq[jj - 1] *= 9; Dq["d"] = dq
    assert not np.allclose(make_features(Dq)["har"][jj], F["har"][jj]), "특성이 직전 봉을 반영"
    assert not np.array_equal(make_sequences(Dq, np.array([jj]), 16, split=split), seq), "시퀀스가 직전 봉을 반영"
    ch = np.column_stack([np.nan_to_num(D["d"]), np.abs(np.nan_to_num(D["d"]))])
    trb = (g < split) & np.isfinite(D["d"])
    man = (ch[jj - 16:jj] - ch[trb].mean(0)) / ch[trb].std(0)
    assert np.allclose(seq[0], man, atol=1e-5), "시퀀스 = d[j0-L:j0]"

    comp, dc, split_c = _compact(D, split=split)
    Gm = garch_t_fit(dc, split_c)
    from arch import arch_model
    x = dc * 100
    am = arch_model(x, mean="Zero", vol="GARCH", p=1, q=1, dist="t")
    res = am.fix([Gm["omega"], Gm["alpha"], Gm["beta"], Gm["nu"]])
    fc = res.forecast(horizon=8, start=split_c + 50, reindex=False).variance.to_numpy()
    pos = np.array([split_c + 51])
    mine = garch_multistep(Gm, pos, 8)
    assert np.allclose(mine[0], fc[0], rtol=1e-5), f"GARCH 다단계 불일치 {mine[0][:3]} vs {fc[0][:3]}"

    from engine import regime_garch as rg
    th = np.array([-2.0, -1.0, 1.0, -1.0, 0.5, 1.5, 2.0, 2.0, 1.0])
    _, info = rg.ms_filter(dc * 100, th, loglik_upto=split_c)
    pos = np.array([split_c + 10])
    msp = ms_multistep(info, pos, 4)
    assert np.isclose(msp[0, 0], info["var_pred"][pos[0]]), "MS 첫 스텝 = 필터 분산 예측(κ·h)"
    # 엔진 t 밀도의 분산이 κ·h인지 수치 적분으로 확인(h=1)
    import math
    from scipy import integrate
    nu = info["nu"]
    cst = math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2) - 0.5 * math.log(math.pi * (nu - 2))
    dens = lambda x_: math.exp(cst - 0.5 * math.log((nu - 2) / nu) - (nu + 1) / 2 * math.log1p(x_ * x_ * nu / (nu - 2) ** 2))
    v = integrate.quad(lambda x_: x_ * x_ * dens(x_), -np.inf, np.inf)[0]
    assert np.isclose(v, info["kappa"], rtol=1e-4), "엔진 밀도의 분산 = κ·h"
    sw = tar_switch(dc)
    assert np.isclose(sw[10], np.sqrt(np.sum(dc[6:10] ** 2))), "TAR 국면 변수는 t 이전 4봉"

    hp = np.full((1, 4), 2.0)
    raw, dep = var_to_rv(hp, D["cfac"], np.array([100]))
    assert np.isclose(raw[0] ** 2, np.sum(2.0 * D["cfac"][100:104] ** 2) / 1e4), "주기 스케일 원복"
    a = rng.uniform(0.5, 2, 1000); p = rng.uniform(0.5, 2, 1000)
    c = qlike_scale(a, p)
    assert qlike_vec(a, p * np.sqrt(c)).mean() <= min(qlike_vec(a, p * np.sqrt(c * s)).mean() for s in (0.9, 1.1)), "보정 상수 최적성"
    print("[selftest] 모든 시험 통과", flush=True)


# %% [markdown]
# ## 보고서

# %%
def regime_tables(store: dict, models: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """사전 구간(직전 H시간 RV, 학습 구간 분위수)별 모델 평균 순위와 구간 크기."""
    rk_rows, size_rows = [], []
    for (tk, H), S in store.items():
        edges = np.quantile(S["nai_tr"], [.2, .4, .6, .8])
        b = np.digitize(S["nai"], edges)
        med_all = float(np.median(S["act"]))
        for q in range(5):
            sel = b == q
            if sel.sum() < 30:
                continue
            ql = {mname: qlike_vec(S["act"][sel], S["preds"][mname][sel]).mean()
                  for mname in models if mname in S["preds"]}
            rk = pd.Series(ql).rank()
            for mname, v in rk.items():
                rk_rows.append({"종목": tk, "H": H, "구간": f"Q{q + 1}", "모델": mname, "순위": v,
                                "dQLIKE_naive": ql[mname] - ql.get("naive", np.nan)})
            size_rows.append({"종목": tk, "H": H, "구간": f"Q{q + 1}", "n": int(sel.sum()),
                              "실현RV중앙값": float(np.median(S["act"][sel])),
                              "중앙값_대비": float(np.median(S["act"][sel]) / med_all)})
    return pd.DataFrame(rk_rows), pd.DataFrame(size_rows)


def quarter_table(store: dict, models: list[str]) -> pd.DataFrame:
    rows = []
    for (tk, H), S in store.items():
        qtr = pd.DatetimeIndex(np.asarray(S["T"]).astype("datetime64[ns]")).to_period("Q").astype(str)
        for qq in np.unique(qtr):
            sel = qtr == qq
            if sel.sum() < 20:
                continue
            ql = {mname: qlike_vec(S["act"][sel], S["preds"][mname][sel]).mean()
                  for mname in models if mname in S["preds"]}
            for mname, v in pd.Series(ql).rank().items():
                rows.append({"종목": tk, "H": H, "분기": qq, "모델": mname, "순위": v})
    return pd.DataFrame(rows)



SHORT = {"GARCH+LightGBM": "G+LGBM", "LightGBM": "LGBM", "XGBoost": "XGB", "HistGBM": "HGB",
         "Nystroem+Ridge": "Nys", "KernelRidge-RBF": "KRR", "SVR-RBF": "SVR", "HAR-RV": "HAR"}
TIE = 0.01      # 동률 폭(QLIKE). GRU 시드 간 표준편차 0.005~0.013(1시간, 4종목 × 5시드)에 근거
TIE2 = 0.03


def cell_losses(rd: pd.DataFrame, store: dict, models: list[str]) -> pd.DataFrame:
    """(종목, H, 사전구간, 기준, 모델)별 평균 QLIKE. 기준: 보정후·보정전·모양(사후 수준 맞춤, 진단용)."""
    C = rd.set_index(["종목", "H", "모델"])["보정계수"]
    out = []
    for (tk, H), S in store.items():
        a = S["act"].astype(float)
        b = np.digitize(S["nai"], np.quantile(S["nai_tr"], [.2, .4, .6, .8]))
        cells = [("전체", np.ones(len(a), bool))] + [(f"Q{q + 1}", b == q) for q in range(5)]
        for nm in models:
            p1 = S["preds"][nm].astype(float)
            p0 = p1 / np.sqrt(float(C[(tk, H, nm)]))
            for cell, sel in cells:
                if sel.sum() < 30:
                    continue
                q1 = qlike_vec(a[sel], p1[sel]).mean()
                q0 = qlike_vec(a[sel], p0[sel]).mean()
                cs = float(np.mean(a[sel] ** 2 / p0[sel] ** 2))
                qs = q0 - (cs - np.log(cs) - 1)
                for ver, v in (("보정후", q1), ("보정전", q0), ("모양", qs)):
                    out.append((tk, H, cell, ver, nm, v))
    return pd.DataFrame(out, columns=["종목", "H", "구간", "기준", "모델", "QLIKE"])


def tier_table(cl: pd.DataFrame) -> pd.DataFrame:
    """칸마다 최선 모델 대비 평균 격차(종목 평균)와, 종목별 최선과 TIE 이내인 종목 수."""
    rows = []
    for (H, cell, ver), g in cl.groupby(["H", "구간", "기준"]):
        pv = g.pivot_table(index="종목", columns="모델", values="QLIKE").dropna(axis=1)
        mean = pv.mean()
        best = mean.idxmin()
        gap = (pv.sub(pv[best], axis=0)).mean()
        near = (pv.sub(pv.min(axis=1), axis=0) <= TIE).sum()
        for nm in pv.columns:
            rows.append({"H": H, "구간": cell, "기준": ver, "모델": nm, "격차": float(gap[nm]),
                         "TIE이내종목": int(near[nm]), "종목수": len(pv),
                         "등급": "A" if gap[nm] <= TIE else ("B" if gap[nm] <= TIE2 else "C")})
    return pd.DataFrame(rows)


def emit_tiers(tt: pd.DataFrame) -> None:
    cells = ["전체", "Q1", "Q2", "Q3", "Q4", "Q5"]
    vers = ["보정후", "보정전", "모양"]
    for H in HORIZONS_H:
        emit(f"### {hlabel(H)}")
        emit()
        emit("| 구간 | " + " | ".join(f"{v} A등급(최선 대비 ≤{TIE})" for v in vers) + " |")
        emit("| :--- | " + " | ".join([":---"] * len(vers)) + " |")
        for cell in cells:
            parts = []
            for v in vers:
                g = tt[(tt.H == H) & (tt["구간"] == cell) & (tt["기준"] == v) & (tt["등급"] == "A")].sort_values("격차")
                parts.append(", ".join(f"{SHORT.get(n, n)}({k}/{t})" for n, k, t in
                                       zip(g["모델"], g["TIE이내종목"], g["종목수"])) or "-")
            emit(f"| {cell} | " + " | ".join(parts) + " |")
        emit()


def family_votes(tt: pd.DataFrame) -> pd.DataFrame:
    a = tt[tt["등급"] == "A"].copy()
    a["계열"] = a["모델"].map(FAMILY)
    return a.groupby(["H", "구간", "기준", "계열"]).size().rename("A등급수").reset_index()


FAM_ORDER = ["통계", "하이브리드", "트리", "딥러닝", "선형", "커널", "벤치마크"]
FAM_COLOR = dict(zip(FAM_ORDER, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]))
FAM_MARK = dict(zip(FAM_ORDER, ["o", "s", "^", "D", "v", "P", "X"]))
FAM_LABEL = {"통계": "통계(GARCH 3종)", "하이브리드": "GARCH+트리", "트리": "트리", "딥러닝": "순환 딥러닝(GRU·LSTM)",
             "선형": "선형(Linear·Ridge)", "커널": "커널", "벤치마크": "HAR-RV"}


def emit_profile(tt: pd.DataFrame) -> None:
    """알고리즘마다 예측 구간별 등급(전체·Q5)과 최선 대비 격차, 그리고 계열 최선 격차 그림."""
    hs = list(HORIZONS_H)
    sub = tt[tt["기준"] == "보정후"]
    shp = tt[tt["기준"] == "모양"]
    emit("셀은 `전체 등급/Q5 등급 (전체 격차)`이다. 등급 A는 최선 대비 ≤0.01, B는 ≤0.03, C는 그 밖이다. 보정후 기준이며, "
         "괄호 안 격차는 그 구간 최선 모델 대비 QLIKE 차이의 종목 평균이다. 마지막 칸은 모양 기준 전체 등급이다.")
    emit()
    emit("| 계열 | 모델 | " + " | ".join(hlabel(H) for H in hs) + " | 모양 등급(" + "·".join(hlabel(H) for H in hs) + ") |")
    emit("| :--- | :--- | " + " | ".join([":---"] * len(hs)) + " | :--- |")
    models = [m_ for m_ in ALL_MODELS if m_ != "naive" and m_ not in DECISION_EXCLUDED]
    for fam in FAM_ORDER:
        for nm in [m_ for m_ in models if FAMILY[m_] == fam]:
            cells, sh = [], []
            for H in hs:
                a = sub[(sub.H == H) & (sub["모델"] == nm) & (sub["구간"] == "전체")]
                q = sub[(sub.H == H) & (sub["모델"] == nm) & (sub["구간"] == "Q5")]
                z = shp[(shp.H == H) & (shp["모델"] == nm) & (shp["구간"] == "전체")]
                if not len(a):
                    cells.append("-"); sh.append("-"); continue
                cells.append(f"{a['등급'].iloc[0]}/{q['등급'].iloc[0] if len(q) else '-'} ({a['격차'].iloc[0]:.3f})")
                sh.append(z["등급"].iloc[0] if len(z) else "-")
            emit(f"| {FAM_LABEL[fam]} | {nm} | " + " | ".join(cells) + " | " + "·".join(sh) + " |")
    emit()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for ax, ver, title in ((axes[0], "보정후", "총 손실(보정 포함, 주 결과)"), (axes[1], "모양", "모양(수준을 사후에 맞춘 진단용)")):
        g = tt[(tt["기준"] == ver) & (tt["구간"] == "전체") & (tt["모델"] != "naive")].copy()
        g["계열"] = g["모델"].map(FAMILY)
        fb = g.groupby(["H", "계열"])["격차"].min().unstack()
        x = np.arange(len(hs))
        ax.axhspan(0, TIE, color="#e8e8e4", zorder=0)
        ax.text(len(hs) - 1 + 0.15, TIE / 2, "동률 폭", fontsize=8, color="#6b6b66", va="center")
        for fam in FAM_ORDER:
            if fam not in fb:
                continue
            yv = fb[fam].reindex(hs).to_numpy()
            ax.plot(x, yv, color=FAM_COLOR[fam], lw=2, marker=FAM_MARK[fam], ms=7,
                    markeredgecolor="white", markeredgewidth=1.5, label=FAM_LABEL[fam], zorder=3)
        ax.set_xticks(x, [hlabel(H) for H in hs])
        ax.set_xlabel("예측 구간")
        ax.set_ylabel("계열 최선 모델의 격차(최선 대비 QLIKE, 종목 평균)")
        ax.set_title(title, fontsize=11, loc="left")
        ax.grid(axis="y", color="#e5e5e0", lw=0.8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.set_ylim(bottom=-0.005)
    axes[0].legend(fontsize=8, frameon=False, loc="upper left")
    fig.suptitle("예측 구간이 길어질 때 계열별로 최선과의 격차가 어떻게 변하는가(0에 가까울수록 최선)", fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    path = IMG / f"{STEM}_fig1_family_gap_by_horizon.png"
    IMG.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    emit(f"![계열별 격차]({os.path.relpath(path, RES)})")
    emit()
    emit("그림 읽는 법: 각 선은 그 계열에서 가장 좋은 모델의 격차다. 회색 띠(0~0.01) 안에 있으면 그 예측 구간의 최선과 "
         "사실상 동률이다. 수치는 위 표와 `tier_table.csv`에 있다.")
    emit()


def emit_zero_split(store: dict, models: list[str]) -> pd.DataFrame:
    """가격 정지(RV=0) 시점과 움직인 시점으로 나눈 손실 비교. 기준은 GARCH-t(수익률 0을 그대로 쓰는 분산 모형)."""
    rows = []
    for (tk, H), S in store.items():
        a = S["act"].astype(float)
        z = a == 0
        base = qlike_vec(a, S["preds"]["GARCH-t"].astype(float))
        for nm in models:
            q = qlike_vec(a, S["preds"][nm].astype(float))
            d = q - base
            rows.append({"종목": tk, "H": H, "모델": nm, "RV0비율": float(z.mean()),
                         "차_전체": float(d.mean()), "차_움직임": float(d[~z].mean()),
                         "차_정지": float(d[z].mean()) if z.any() else np.nan})
    zz = pd.DataFrame(rows)
    zz.to_csv(RES / f"{STEM}_zero_split.csv", index=False)
    hs = list(HORIZONS_H)
    r0 = zz[zz["모델"] == "GARCH-t"].groupby("H")["RV0비율"].median()
    emit("평가 시점을 **가격이 움직인 시점(RV>0)**과 **가격이 한 칸도 안 움직인 시점(RV=0)**으로 나눠, 각 모델의 "
         "손실을 GARCH-t와 비교한다(음수면 그 모델이 GARCH-t보다 낫다). RV=0에서는 QLIKE가 log(예측 분산)이 되어 작게 "
         "예측할수록 유리하다. 로그 타깃 모델은 log(0)을 정의할 수 없어 RV=0 표본을 빼고 학습하므로 이런 시점을 배운 "
         "적이 없다. 반면 GARCH 계열은 수익률 0을 그대로 받아 분산을 추정한다.")
    emit()
    emit("| 예측 구간 | 평가 RV=0 비율(종목 중앙) | 종목별 RV=0 비율과 (LightGBM−GARCH-t) 격차의 상관 |")
    emit("| :--- | ---: | ---: |")
    for H in hs:
        g = zz[(zz["H"] == H) & (zz["모델"] == "LightGBM")]
        cc = np.corrcoef(g["RV0비율"], g["차_전체"])[0, 1] if g["RV0비율"].std() > 0 else np.nan
        emit(f"| {hlabel(H)} | {r0.get(H, np.nan):.1%} | {cc:+.2f} |")
    emit()
    emit("셀은 `움직인 시점 차 / 정지 시점 차`(GARCH-t 대비, 종목 평균)이다.")
    emit()
    emit("| 모델 | " + " | ".join(hlabel(H) for H in hs) + " |")
    emit("| :--- | " + " | ".join([":---"] * len(hs)) + " |")
    for nm in models:
        if nm == "GARCH-t":
            continue
        cells = []
        for H in hs:
            g = zz[(zz["H"] == H) & (zz["모델"] == nm)]
            z_ = g["차_정지"].mean()
            cells.append(f"{g['차_움직임'].mean():+.3f} / " + ("-" if not np.isfinite(z_) else f"{z_:+.3f}"))
        emit(f"| {nm} | " + " | ".join(cells) + " |")
    emit()
    return zz


def write_report(rd, store, fails_df, eda_df, heda_df, tickers, quick, elapsed_h, truncated):
    from report_header import render_standard_header
    rd = rd[~rd["모델"].isin(DECISION_EXCLUDED)].copy()
    models = [m for m in ALL_MODELS if m in set(rd["모델"])]
    rk, size = regime_tables(store, models)
    qt = quarter_table(store, models)
    rk.to_csv(RES / f"{STEM}_exante_regime_ranks.csv", index=False)
    size.to_csv(RES / f"{STEM}_exante_regime_sizes.csv", index=False)
    qt.to_csv(RES / f"{STEM}_quarter_ranks.csv", index=False)

    emit(f"# 26번: 시간 기준 평가 재정립과 24번 재평가{' (quick 배관 확인, 결과 해석 금지)' if quick else ''}")
    emit()
    btc_idx = M23.load_close("KRW-BTC").index
    split_frac = float(np.mean(btc_idx < SPLIT))
    emit(render_standard_header(
        tickers=tickers, train_frac=split_frac, rep_ticker="KRW-BTC", analyzed_tickers=tickers,
        transforms=[("15분 시간 격자 복원", "업비트는 무체결 구간에 캔들을 만들지 않는다. 행 기준 창은 시간이 아니다"),
                    ("점검 구간 제외", "BTC·ETH·XRP가 동시에 빈 시각 = 거래소 전체 중단. 보간하지 않는다"),
                    ("로그수익률", "종가 기반"),
                    ("주기 제거(입력 전용)", "(요일,시) 슬롯 평균 |r|, 학습 구간만으로 추정. 타깃은 원 스케일")]))
    emit()
    emit("## 0. 이 회차가 고친 것")
    emit()
    emit("24·25번 점검에서 확인된 평가 틀의 결함을 고친 재평가다. 결함과 처리는 드라이버 상단 표에 있다. "
         f"분할 시각은 전 종목 공통 **{SPLIT}**(KST)이고, 예측 시점은 정시, 예측 구간은 15분·30분·1시간·4시간·12시간이다. "
         "24번 수치와 직접 빼서 비교하지 않는다(평가 단위·표본·타깃이 모두 다르다).")
    emit()
    emit("**선형 모델(Linear·Ridge) 제외**: 2026-09-07 가정 진단에서 선형성 검정(RESET·BDS)이 기각되어 선형 모델을 "
         "비선형·커널 계열로 대체하기로 결정했다(`assumption_diagnosis_report_20260907.md` 8절 3번). 23~26번 구현은 이 "
         "결정을 반영하지 않고 두 모델을 계속 적합해 왔다. 이미 계산된 값은 결과 CSV에 기록으로만 남기고, 이 보고서의 "
         "순위·동률·검정·결론에서는 모두 뺐다. HAR-RV는 벤치마크로 따로 분류해 왔으므로 일단 남긴다(포함 여부 확인 중).")
    emit()
    emit(f"소요 {elapsed_h:.2f}시간.")
    if truncated:
        emit("**마감으로 잘린 작업**: " + "; ".join(truncated))
    emit()

    emit("## 1. 데이터: 시간 격자 위에서 본 결측과 평가 표본")
    emit()
    emit("| 종목 | 무체결 봉(전체) | 무체결 봉(평가 구간) | 점검 봉 | 영수익률 비율 |")
    emit("| :--- | ---: | ---: | ---: | ---: |")
    for _, x in eda_df.iterrows():
        emit(f"| {x['종목'].replace('KRW-', '')} | {x['무체결봉비율']:.2%} | {x['무체결봉비율_평가구간']:.2%} | "
             f"{int(x['점검봉'])} | {x['영수익률비율']:.1%} |")
    emit()
    emit("예측 구간별 평가 표본과 실현변동성 크기(종목 중앙값):")
    emit()
    emit("| H | 평가 표본(종목 중앙) | 학습 RV 중앙값 | 평가 RV 중앙값 | 평가 RV 95% | 평가 RV=0 비율 | 학습 제외(RV=0) |")
    emit("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for H, g in heda_df.groupby("H"):
        emit(f"| {hlabel(H)} | {int(g['평가표본'].median()):,} | {g['학습RV중앙값'].median():.3%} | "
             f"{g['평가RV중앙값'].median():.3%} | {g['평가RV95'].median():.3%} | "
             f"{g['평가RV0비율'].median():.2%} | {g['학습제외_RV0'].median():.2%} |")
    emit()

    emit("## 2. 적합 완결성")
    emit()
    exp_n = len(tickers) * len(HORIZONS_H) * len(models)   # 결론에 쓰는 모델 기준(선형 2종 제외)
    emit(f"기대 {len(tickers)}종목 × {len(HORIZONS_H)}구간 × {len(models)}모델 = {exp_n}행, 실제 {len(rd)}행.")
    if len(fails_df):
        emit()
        emit("| 종목 | 모델 | H | 예외 | 메시지 |")
        emit("| :--- | :--- | :--- | :--- | :--- |")
        for _, x in fails_df.iterrows():
            emit(f"| {x['종목']} | {x['모델']} | {x['H']} | {x['예외']} | {str(x['메시지'])[:80]} |")
    else:
        emit("적합 실패 0건.")
    emit()

    emit("## 3. 전체 순위(예측 구간별)")
    emit()
    emit("종목마다 QLIKE 순위를 매겨 평균했다(작을수록 좋음). ΔQLIKE는 naive 대비 차이의 종목 평균으로, "
         "스케일에 불변이라 종목을 가로질러 평균할 수 있다(음수일수록 naive보다 좋음).")
    emit()
    emit("**주 결과는 전원 보정이다.** 모든 모델(naive 제외)이 같은 절차로 분산 배율 c를 받는다. 로그 타깃 모델은 "
         "내부학습 모델의 내부검증 정시 예측으로, GARCH 3종은 내부학습 구간까지로 다시 적합한 모형의 같은 시점 "
         "예측으로 c를 추정한다. 이 보정은 로그 역변환 편향을 고치는 동시에 최근 수준에 맞춘 조정 효과도 내므로, "
         "한 계열에만 적용하면 순위가 통째로 바뀐다(초기 실행에서 확인). 그래서 **전원 미보정 순위**를 민감도 "
         "분석으로 함께 싣는다. 두 순위가 어긋나는 모델은 수준 맞춤의 영향을 크게 받는 모델이다.")
    emit()
    rd = rd.copy()
    rd["rank"] = rd.groupby(["종목", "H"])["QLIKE"].rank()
    nq = rd[rd["모델"] == "naive"][["종목", "H", "QLIKE"]].rename(columns={"QLIKE": "QLIKE_naive"})
    rd = rd.merge(nq, on=["종목", "H"], how="left")
    rd["dQLIKE"] = rd["QLIKE"] - rd["QLIKE_naive"]
    rd["rank_raw"] = rd.groupby(["종목", "H"])["QLIKE_보정전"].rank()
    tab = rd.groupby(["H", "모델"]).agg(종목수=("종목", "nunique"), 순위=("rank", "mean"), 순위_미보정=("rank_raw", "mean"),
                                       dQ=("dQLIKE", "mean"),
                                       MASE=("MASE", "median"), 보정=("보정계수", "median"),
                                       보정전=("QLIKE_보정전", "mean"), 보정후=("QLIKE", "mean")).reset_index()
    for H in HORIZONS_H:
        t = tab[tab["H"] == H].sort_values("순위")
        emit(f"### {hlabel(H)}")
        emit()
        emit("| 순위 | 모델 | 처리 방식 | 종목 수 | 평균 순위 | 평균 순위(전원 미보정) | ΔQLIKE(naive 대비) | MASE 중앙 | 보정계수 중앙 |")
        emit("| ---: | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for i, (_, x) in enumerate(t.iterrows(), 1):
            emit(f"| {i} | {x['모델']} | {PROCESS[x['모델']]} | {int(x['종목수'])} | {x['순위']:.1f} | "
                 f"{x['순위_미보정']:.1f} | {x['dQ']:+.4f} | {x['MASE']:.3f} | {x['보정']:.2f} |")
        emit()
    tab.to_csv(RES / f"{STEM}_overall_ranks.csv", index=False)

    emit("### 역변환 보정의 효과")
    emit()
    emit("로그 타깃 모델은 exp 역변환이 조건부 평균이 아닌 기하평균을 내서 체계적으로 낮게 예측한다. "
         "보정계수(분산 배율)가 1보다 크면 과소예측이었다는 뜻이다. GARCH 계열의 보정계수는 역변환 편향이 아니라 "
         "수준 맞춤만 반영한다. 보정 전 QLIKE와 함께 보인다.")
    emit()
    emit("| H | 모델 | 보정계수 중앙 | QLIKE 보정 전(종목 평균) | 보정 후 |")
    emit("| ---: | :--- | ---: | ---: | ---: |")
    for _, x in tab[tab["모델"] != "naive"].iterrows():
        emit(f"| {hlabel(int(x['H']))} | {x['모델']} | {x['보정']:.2f} | {x['보정전']:.4f} | {x['보정후']:.4f} |")
    emit()

    emit("## 4. 사전 구간별 순위(직전 H시간 RV 5분위, 학습 구간 분위수)")
    emit()
    emit("구간 경계는 학습 구간의 직전 RV 분위수로 정해서 예측 시점에 이미 알 수 있다. 24·25번의 사후(실현 RV) "
         "구간과 다르다.")
    emit()
    for H in HORIZONS_H:
        g = rk[rk["H"] == H]
        if not len(g):
            continue
        pv = g.pivot_table(index="모델", columns="구간", values="순위", aggfunc="mean")
        pv = pv.loc[pv.mean(axis=1).sort_values().index]
        sz = size[size["H"] == H].groupby("구간")["중앙값_대비"].median()
        emit(f"### {hlabel(H)}")
        emit()
        emit("| 모델 | " + " | ".join(pv.columns) + " |")
        emit("| :--- | " + " | ".join(["---:"] * len(pv.columns)) + " |")
        emit("| *구간 실현RV(종목 중앙값 대비)* | " + " | ".join(f"*{sz.get(c, np.nan):.2f}배*" for c in pv.columns) + " |")
        colbest = pv.idxmin()
        for mname, x in pv.iterrows():
            emit(f"| {mname} | " + " | ".join((f"**{v:.1f}**" if colbest[c] == mname else f"{v:.1f}")
                                             for c, v in x.items()) + " |")
        emit()
        emit("굵은 글씨는 그 구간(열)의 1위다. 구간별 1위: " + ", ".join(f"{c}={pv[c].idxmin()}" for c in pv.columns))
        emit()

    emit("## 5. 달력 분기별 순위(평가 구간)")
    emit()
    for H in HORIZONS_H:
        g = qt[qt["H"] == H]
        if not len(g):
            continue
        pv = g.pivot_table(index="모델", columns="분기", values="순위", aggfunc="mean")
        pv = pv.loc[pv.mean(axis=1).sort_values().index].head(8)
        emit(f"### {hlabel(H)} (상위 8개 모델)")
        emit()
        emit("| 모델 | " + " | ".join(pv.columns) + " |")
        emit("| :--- | " + " | ".join(["---:"] * len(pv.columns)) + " |")
        for mname, x in pv.iterrows():
            emit(f"| {mname} | " + " | ".join(f"{v:.1f}" for v in x.values) + " |")
        emit()

    emit("## 6. 통합 판정: 구간별로 어느 모델들이 사실상 같이 최선인가")
    emit()
    emit(f"순위는 작은 차이도 한 계단씩 벌려 놓는다. 그래서 칸마다 **최선 모델 대비 QLIKE 격차가 {TIE} 이하인 모델을 "
         f"A등급(사실상 동률)**으로 묶는다. {TIE}는 GRU를 시드 5개로 다시 학습했을 때의 QLIKE 표준편차(0.005~0.013)에 "
         "맞춘 값으로, 이보다 작은 차이는 모델 차이라고 볼 수 없다. 괄호는 그 모델이 종목별 최선과 "
         f"{TIE} 이내였던 종목 수다. 세 기준을 나란히 둔다.")
    emit()
    emit("- **보정후**(주 결과): 전 모델에 같은 절차의 수준 보정")
    emit("- **보정전**(민감도): 아무 보정 없음. 로그 타깃 모델은 옌센 편향을 그대로 안는다")
    emit("- **모양**(진단용, 사후): 칸마다 평가 구간에서 수준을 최적으로 맞춘 뒤의 손실. 수준 정확도를 빼고 오르내림 "
         "추적 실력만 본다. 평가 구간 정보를 쓰므로 실전 성능이 아니다")
    emit()
    cl = cell_losses(rd, store, models)
    tt = tier_table(cl)
    tt.to_csv(RES / f"{STEM}_tier_table.csv", index=False)
    family_votes(tt).to_csv(RES / f"{STEM}_tier_family_votes.csv", index=False)
    emit_tiers(tt)
    emit()
    emit("### 알고리즘별 프로필: 예측 구간이 길어지면 어떻게 되는가")
    emit()
    emit_profile(tt)
    emit("### 짧은 예측 구간과 가격 정지(RV=0): 결과가 데이터 특성에서 오는가")
    emit()
    emit_zero_split(store, models)

    emit("## 7. 24번과 결과가 다른 이유(같은 1시간 예측)")
    emit()
    emit("QLIKE는 **모양**(수준을 완벽히 맞췄을 때의 손실)과 **수준 벌점**(c*−log c*−1, c*=평가 구간의 최적 배율)으로 "
         "정확히 나뉜다. 24번 원래 예측(포스터 근거)과 26번 보정 전 예측에 같은 분해를 적용했다.")
    emit()
    try:
        z = np.load(ROOT / "test" / "results" / "24_maxscale_refit_20260928" / "24_maxscale_refit_validation_predictions.npz")
        r24 = []
        for k in z.files:
            if not k.endswith("|1|실제"):
                continue
            tk = k.split("|")[0]
            a = z[k].astype(float)
            for mk in z.files:
                if mk.startswith(f"{tk}|1|") and not mk.endswith("|실제") and mk.split("|")[2] in models:
                    pp = z[mk].astype(float)
                    q = np.log(pp ** 2) + a ** 2 / pp ** 2
                    ok = np.isfinite(q)
                    cs = float(np.mean(a[ok] ** 2 / pp[ok] ** 2))
                    r24.append((tk, mk.split("|")[2], q[ok].mean(), q[ok].mean() - (cs - np.log(cs) - 1), cs - np.log(cs) - 1))
        d24 = pd.DataFrame(r24, columns=["종목", "모델", "총", "모양", "벌점"])
        d26 = cl[(cl.H == 60) & (cl["구간"] == "전체") & cl["기준"].isin(["보정전", "모양"])].pivot_table(
            index=["종목", "모델"], columns="기준", values="QLIKE").reset_index()
        d26["벌점"] = d26["보정전"] - d26["모양"]
        rk = lambda df, col: df.groupby("종목")[col].rank()
        d24["순위총"], d24["순위모양"] = rk(d24, "총"), rk(d24, "모양")
        d26["순위총"], d26["순위모양"] = rk(d26, "보정전"), rk(d26, "모양")
        t = pd.DataFrame({"24 총": d24.groupby("모델")["순위총"].mean(), "24 모양": d24.groupby("모델")["순위모양"].mean(),
                          "24 벌점": d24.groupby("모델")["벌점"].mean(),
                          "26 총(보정전)": d26.groupby("모델")["순위총"].mean(), "26 모양": d26.groupby("모델")["순위모양"].mean(),
                          "26 벌점": d26.groupby("모델")["벌점"].mean()}).sort_values("24 총")
        emit("| 모델 | 24번 총 순위 | 24번 모양 순위 | 24번 수준 벌점 | 26번 총 순위(보정 전) | 26번 모양 순위 | 26번 수준 벌점 |")
        emit("| :--- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for nm, x in t.iterrows():
            emit(f"| {nm} | {x['24 총']:.1f} | {x['24 모양']:.1f} | {x['24 벌점']:.3f} | {x['26 총(보정전)']:.1f} | "
                 f"{x['26 모양']:.1f} | {x['26 벌점']:.3f} |")
        emit()
        emit("읽는 법: 24번에서도 GRU·LSTM은 모양 1·2위였고, 로그 타깃 모델 공통의 수준 벌점(옌센 편향)에 가려 총 순위가 "
             "5~6위였다. GARCH는 모양 하위권이지만 수준이 정확해서 1위였다. 26번에서는 두 수정이 반대로 작용했다. "
             "정보 시점 통일(24번은 GRU·LSTM만 한 봉 더 최신 정보를 봤다)로 GRU의 모양 우위가 줄고 트리가 올라왔으며, "
             "보정은 로그 모델 전체의 수준 벌점을 지웠다. GRU가 새로 튀어오른 것이 아니라, GARCH와 트리의 자리가 바뀐 것이다.")
    except Exception as e:
        emit(f"(24번 예측 파일을 읽지 못했다: {type(e).__name__})")
    emit()

    emit("## 8. 모형 불안정 사례")
    emit()
    unstable = rd[rd["모델"].isin(VAR_MODELS) & ((rd["보정계수"] < 0.3) | (rd["보정계수"] > 3.0))]
    if len(unstable):
        emit("GARCH 계열의 보정 상수는 내부학습 구간까지로 다시 적합한 모형에서 추정한다. 이 상수가 극단적이면 "
             "추정 구간에 따라 모수가 크게 달라진다는 뜻이다(평가 예측은 학습 구간 전체 모형에서 나온다).")
        emit()
        emit("| 종목 | H | 모델 | 보정계수 | QLIKE 보정 전 | 보정 후 |")
        emit("| :--- | ---: | :--- | ---: | ---: | ---: |")
        for _, x in unstable.iterrows():
            emit(f"| {x['종목']} | {x['H']} | {x['모델']} | {x['보정계수']:.3f} | {x['QLIKE_보정전']:.3f} | {x['QLIKE']:.3f} |")
    else:
        emit("보정 상수가 극단적인(0.3 미만 또는 3 초과) GARCH 계열 사례는 없다.")
    emit()
    emit("## 9. 남은 일")
    emit()
    emit("- 유의성 검정(DM·MCS)을 이 예측값으로 다시 한다: 종목×구간×모델쌍 다중비교 보정, 종목 간 상관 반영, "
         "구간 효과는 조건부 예측 능력 검정.")
    emit("- 연구 종목 선정(`study_universe`)의 변동성 기준이 행 기준 수익률로 계산돼 있어, 무체결 봉이 많은 종목의 "
         "변동성이 부풀었을 수 있다. 종목군은 비교를 위해 그대로 두었다.")
    emit("- TAR-GARCH 다단계 예측의 국면 결정은 대입 근사다.")
    (RES / f"{STEM}_report.md").write_text("\n".join(_LINES), encoding="utf-8")



# %% [markdown]
# ## GARCH 계열에도 같은 보정을 적용한다
#
# 로그 타깃 모델의 보정 상수 c는 내부학습 구간으로 적합한 모델의 내부검증 정시 예측에서 추정한다.
# 이 절차는 역변환 편향뿐 아니라 최근 수준에 맞춘 조정 효과도 내므로, GARCH 계열에만 빠지면
# 비교가 불공정하다. GARCH 3종도 **내부학습 구간까지로 다시 적합**해 같은 시점에서 c를 추정하고,
# 평가 예측(학습 구간 전체 적합)에 곱한다. 순서는 다른 모델과 같다(필터 → 배율).

# %%
def garch_calib_job(ticker: str, quick: bool) -> dict:
    from engine import regime_garch as rg
    t0 = time.time()
    D = build_data(ticker)
    comp, dc, _ = _compact(D)
    HDs = {H: horizon_data(D, H) for H in HORIZONS_H}
    fits: dict = {}
    fails: list = []
    for inner in sorted({HD["inner"] for HD in HDs.values()}):
        ic = int(np.searchsorted(D["grid"][comp], inner))
        f = {}
        try:
            f["GARCH-t"] = garch_t_fit(dc, ic)
        except Exception as e:
            fails.append(("GARCH-t", "calib", type(e).__name__, str(e)[:160]))
        try:
            _, _, info = rg.fit_ms_garch(dc * 100, ic, n_restarts=MS_RESTARTS if not quick else 1,
                                         maxiter=MS_MAXITER if not quick else 120,
                                         max_fit_n=MS_FIT_N if not quick else 8000)
            if info is None:
                raise RuntimeError("최종 필터가 발산(info=None)")
            f["MS-GARCH"] = info
        except Exception as e:
            fails.append(("MS-GARCH", "calib", type(e).__name__, str(e)[:160]))
        try:
            sw = tar_switch(dc)
            taus = np.quantile(sw[TAR_SWITCH_BARS:ic], list(TAR_TAU_Q))
            _, _, _, info = rg.fit_tar_garch(dc * 100, sw, ic, taus,
                                             n_restarts=TAR_RESTARTS if not quick else 1,
                                             maxiter=TAR_MAXITER if not quick else 100)
            if info is None:
                raise RuntimeError("최종 필터가 발산(info=None)")
            f["TAR-GARCH"] = info
        except Exception as e:
            fails.append(("TAR-GARCH", "calib", type(e).__name__, str(e)[:160]))
        fits[inner] = f
    calib = {}
    for H, HD in HDs.items():
        S = Scorer(ticker, HD)
        jv = HD["j"][HD["va"]]
        pos = _origin_pos(comp, jv, HD["m"])
        for nm, info in fits[HD["inner"]].items():
            hp = (garch_multistep(info, pos, HD["m"]) if nm == "GARCH-t" else
                  ms_multistep(info, pos, HD["m"]) if nm == "MS-GARCH" else tar_multistep(info, dc, pos, HD["m"]))
            raw, _ = var_to_rv(hp, D["cfac"], jv)
            calib[(H, nm)] = S.calib(raw)
    return dict(ticker=ticker, calib=calib, fails=fails, elapsed=time.time() - t0)


def apply_calib(rd: pd.DataFrame, store: dict, calib: dict) -> pd.DataFrame:
    """저장된 GARCH 계열 평가 예측(c=1, 필터 적용)에 c를 곱하고 지표를 다시 계산한다."""
    rd = rd.copy()
    for (tk, H, nm), c in calib.items():
        S = store[(tk, H)]
        p_new = S["preds"][nm].astype(float) * np.sqrt(c)
        ix = rd.index[(rd["종목"] == tk) & (rd["H"] == H) & (rd["모델"] == nm)]
        if len(ix) != 1:
            raise AssertionError(f"{tk} H={H} {nm} 행이 {len(ix)}개")
        ev = evaluate(S["act"].astype(float), p_new, S["preds"]["naive"].astype(float))
        for k_, v_ in ev.items():
            rd.loc[ix, k_] = v_
        rd.loc[ix, "보정계수"] = c
        rd.loc[ix, "비고"] = "다단계 예측 · 내부학습 재적합으로 보정"
        S["preds"][nm] = p_new.astype(np.float32)
    return rd


def load_saved() -> tuple[pd.DataFrame, dict]:
    rd = pd.read_csv(RES / f"{STEM}_model_comparison.csv")
    z = np.load(RES / f"{STEM}_test_predictions.npz")
    store: dict = {}
    names = {"실제": "act", "naive_입력": "nai", "시각": "T", "학습naive": "nai_tr"}
    for k in z.files:
        tk, H, nm = k.split("|")
        S = store.setdefault((tk, int(H)), {"preds": {}})
        if nm in names:
            S[names[nm]] = z[k]
        else:
            S["preds"][nm] = z[k]
    return rd, store


def save_preds(store: dict) -> None:
    arrs = {}
    for (tk, H), S in store.items():
        arrs[f"{tk}|{H}|실제"] = S["act"]; arrs[f"{tk}|{H}|naive_입력"] = S["nai"]
        arrs[f"{tk}|{H}|시각"] = S["T"]; arrs[f"{tk}|{H}|학습naive"] = S["nai_tr"]
        for mname, p in S["preds"].items():
            arrs[f"{tk}|{H}|{mname}"] = p
    np.savez_compressed(RES / f"{STEM}_test_predictions.npz", **arrs)


def run_garch_calib(quick: bool, workers: int, elapsed_min: float) -> None:
    rd, store = load_saved()
    tickers = sorted(rd["종목"].unique())
    if "보정계수_GARCH적용" in rd.columns:
        raise RuntimeError("이미 GARCH 보정이 적용된 결과다. 본 실행부터 다시 돌려라")
    calib, fails = {}, []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(garch_calib_job, tk, quick): tk for tk in tickers}
        for k, fu in enumerate(as_completed(futs), 1):
            res = fu.result()
            for (H, nm), c in res["calib"].items():
                calib[(res["ticker"], H, nm)] = c
            fails.extend({"종목": res["ticker"], "모델": f[0], "H": f[1], "예외": f[2], "메시지": f[3]}
                         for f in res["fails"])
            print(f"  [보정 {k}/{len(tickers)}] {res['ticker']} ({res['elapsed']:.0f}s)", flush=True)
    rd = apply_calib(rd, store, calib)
    rd["보정계수_GARCH적용"] = True
    rd.to_csv(RES / f"{STEM}_model_comparison.csv", index=False)
    save_preds(store)
    fdf = pd.read_csv(RES / f"{STEM}_fit_failures.csv")
    fdf = pd.concat([fdf, pd.DataFrame(fails, columns=fdf.columns)], ignore_index=True)
    fdf.to_csv(RES / f"{STEM}_fit_failures.csv", index=False)
    write_report(rd, store, fdf, pd.read_csv(RES / f"{STEM}_data_eda.csv").sort_values("종목"),
                 pd.read_csv(RES / f"{STEM}_horizon_eda.csv"), tickers, quick,
                 (elapsed_min + (time.time() - t0) / 60) / 60, [])
    print(f"[보정 완료] {(time.time() - t0) / 60:.1f}분 · GARCH 보정 {len(calib)}건 · 실패 {len(fails)}", flush=True)


# %% [markdown]
# ## 실행

# %%
def main(argv=None) -> None:
    from report_header import study_universe
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--n-tickers", type=int, default=20)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--deadline-h", type=float, default=10.0)
    ap.add_argument("--garch-calib", action="store_true",
                    help="저장된 결과에 GARCH 계열 보정(내부학습 재적합)만 추가하고 보고서를 다시 쓴다")
    ap.add_argument("--elapsed-min", type=float, default=0.0, help="본 실행 소요(분), 보고서 기록용")
    ap.add_argument("--report-only", action="store_true", help="저장된 결과로 보고서만 다시 쓴다")
    ap.add_argument("--seed", type=int, default=0, help="0보다 크면 시드 반복 실행(무작위성 있는 모델만, 보고서 없음)")
    a = ap.parse_args(argv)

    selftest()
    if a.selftest:
        return
    global SEED, RUN_STEM
    if a.seed:
        os.environ["RUN26_SEED"] = str(a.seed)
        SEED, RUN_STEM = a.seed, f"{STEM}_seed{a.seed}"
    if a.garch_calib:
        run_garch_calib(a.quick, a.workers or 5, a.elapsed_min)
        return
    if a.report_only:
        rd, store = load_saved()
        write_report(rd, store, pd.read_csv(RES / f"{STEM}_fit_failures.csv"),
                     pd.read_csv(RES / f"{STEM}_data_eda.csv").sort_values("종목"),
                     pd.read_csv(RES / f"{STEM}_horizon_eda.csv"), sorted(rd["종목"].unique()), a.quick,
                     a.elapsed_min / 60, [])
        return
    IMG.mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    deadline = t_start + a.deadline_h * 3600
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tickers, _ = study_universe()
    tickers = tickers[:3] if a.quick else tickers[:a.n_tickers]
    halt_labels()
    workers = a.workers or max(1, min(6, int((free_ram_gb() - 8.0) / 3.0), len(tickers)))
    print(f"[시작] 종목 {len(tickers)} × 구간 {HORIZONS_H} · CPU 워커 {workers} · {dev} · "
          f"점검 봉 {len(halt_labels())}", flush=True)

    rows, fails, eda, heda, metas, timings, truncated = [], [], [], [], [], [], []
    store: dict[tuple[str, int], dict] = {}

    def add_fail(tk, f):
        nm, H, et, msg = f
        fails.append({"종목": tk, "모델": nm, "H": H, "예외": et, "메시지": msg})
        print(f"    ! {tk} {nm} H={H} 실패 — {et}: {msg}", flush=True)

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(run_cpu_job, tk, a.quick): tk for tk in tickers}
        gjobs = [(tk, H) for tk in tickers for H in HORIZONS_H]
        for i, (tk, H) in enumerate(gjobs, 1):
            if time.time() > deadline:
                truncated.append(f"GPU {len(gjobs) - i + 1}개 미실행")
                break
            try:
                res = run_gpu_job(tk, H, a.quick, dev)
                rows.extend(res["rows"])
                store.setdefault((tk, H), {"preds": {}})["preds"].update(res["preds"])
                metas.append(res["meta"])
                for f in res["fails"]:
                    add_fail(tk, f)
                print(f"  [GPU {i}/{len(gjobs)}] {tk} H={H} ({res['elapsed']:.0f}s)", flush=True)
            except Exception as e:
                add_fail(tk, ("(GPU)", H, type(e).__name__, str(e)[:200]))
                traceback.print_exc()
        for k, fu in enumerate(as_completed(futs), 1):
            tk = futs[fu]
            try:
                res = fu.result()
            except Exception as e:
                add_fail(tk, ("(CPU)", "*", type(e).__name__, str(e)[:200]))
                traceback.print_exc()
                continue
            eda.append(res["data_eda"])
            metas.append(res["meta"])
            timings.append({"종목": tk, "총소요초": res["elapsed"], **res["timings"]})
            for f in res["fails"]:
                add_fail(tk, f)
            for R in res["results"]:
                rows.extend(R["rows"])
                S = store.setdefault((tk, R["H"]), {"preds": {}})
                S["preds"].update(R["preds"])
                S.update(act=R["act"], nai=R["nai"], T=R["T"], nai_tr=R["nai_tr"])
                heda.append({"종목": tk, "H": R["H"], **R["eda"]})
            print(f"  [CPU {k}/{len(tickers)}] {tk} ({res['elapsed']:.0f}s)", flush=True)
            pd.DataFrame(rows).to_csv(RES / f"{RUN_STEM}_model_comparison_partial.csv", index=False)

    orphan = [k for k, v in store.items() if "act" not in v]
    if orphan:
        print(f"[경고] CPU 작업이 없어 예측 저장에서 빠진 (종목, H): {orphan}", flush=True)
        truncated.append(f"CPU 실패로 예측 저장 누락 {orphan}")
    store = {k: v for k, v in store.items() if "act" in v}
    if len(rows):
        keep = set(store)
        rows = [r_ for r_ in rows if (r_["종목"], r_["H"]) in keep]
    rd = pd.DataFrame(rows)
    for (tk, H), S in store.items():
        for mname, p in S["preds"].items():
            if len(p) != len(S["act"]):
                raise AssertionError(f"{tk} H={H} {mname} 예측 길이 불일치")
    if SEED:
        rd.to_csv(RES / f"{RUN_STEM}_model_comparison.csv", index=False)
        pd.DataFrame(fails, columns=["종목", "모델", "H", "예외", "메시지"]).to_csv(
            RES / f"{RUN_STEM}_fit_failures.csv", index=False)
        pd.DataFrame(metas).to_csv(RES / f"{RUN_STEM}_chosen_hyperparams.csv", index=False)
        arrs = {f"{tk}|{H}|{mname}": p for (tk, H), S_ in store.items() for mname, p in S_["preds"].items()}
        np.savez_compressed(RES / f"{RUN_STEM}_test_predictions.npz", **arrs)
        pp = RES / f"{RUN_STEM}_model_comparison_partial.csv"
        if pp.exists():
            pp.unlink()
        print(f"[시드 {SEED} 완료] {(time.time() - t_start) / 60:.1f}분 · {len(rd)}행 · 실패 {len(fails)}", flush=True)
        return
    rd.to_csv(RES / f"{STEM}_model_comparison.csv", index=False)
    fails_df = pd.DataFrame(fails, columns=["종목", "모델", "H", "예외", "메시지"])
    fails_df.to_csv(RES / f"{STEM}_fit_failures.csv", index=False)
    eda_df, heda_df = pd.DataFrame(eda), pd.DataFrame(heda)
    eda_df.to_csv(RES / f"{STEM}_data_eda.csv", index=False)
    heda_df.to_csv(RES / f"{STEM}_horizon_eda.csv", index=False)
    pd.DataFrame(metas).to_csv(RES / f"{STEM}_chosen_hyperparams.csv", index=False)
    pd.DataFrame(timings).to_csv(RES / f"{STEM}_timings.csv", index=False)
    arrs = {}
    for (tk, H), S in store.items():
        arrs[f"{tk}|{H}|실제"] = S["act"]; arrs[f"{tk}|{H}|naive_입력"] = S["nai"]
        arrs[f"{tk}|{H}|시각"] = S["T"]; arrs[f"{tk}|{H}|학습naive"] = S["nai_tr"]
        for mname, p in S["preds"].items():
            arrs[f"{tk}|{H}|{mname}"] = p
    np.savez_compressed(RES / f"{STEM}_test_predictions.npz", **arrs)
    write_report(rd, store, fails_df, eda_df.sort_values("종목"), heda_df, tickers, a.quick,
                 (time.time() - t_start) / 3600, truncated)
    print(f"[완료] {(time.time() - t_start) / 60:.1f}분 · {len(rd)}행 · 실패 {len(fails_df)}", flush=True)


if __name__ == "__main__":
    main()
