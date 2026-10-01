# %% [markdown]
# # 24번 — 변동성 예측 모델 최대 규모 재적합
#
# 23번(`23_volatility_model_comparison_test.py`)의 후속이다. 23번은 "작은 규모로 일단
# 전부 돌려 본다"가 목적이어서 트리 300그루·딥러닝 8에폭·은닉 48처럼 축소된 설정을 썼고,
# 그 때문에 Transformer 계열이 naive보다 나쁘게(MASE>1) 나왔을 때 **성능이 낮은 것인지
# 학습이 덜 된 것인지 구분할 수 없어 판단을 보류**했다. 이번 회차가 그 보류를 푼다.
#
# 23번과 달라지는 점은 세 가지다.
#
# 1. **레짐 전환 GARCH 2종을 본편으로 편입** — MS-GARCH·TAR-GARCH를 예비검증에서 본편으로
#    올려 나머지와 **같은 실행·같은 분할·같은 필터·같은 지표**로 나란히 비교한다.
#    (`process.md`엔 한때 "예비검증은 종목별 5분위, 본편은 고정 절대구간 풀링이라 계산
#    방식이 달라 비교 불가"라고 적혀 있었는데, 이번에 본편 산출물과 직접 대조해 그 기록이
#    틀렸음을 확인했다 — 본편도 종목별 5분위였다. 그 제약은 애초에 없었지만, 이번 회차는
#    어차피 두 모델을 최대 규모로 다시 돌리므로 정식 통합의 의미는 그대로 유효하다.)
# 2. **최대 규모 하이퍼파라미터 + 표준 최적화** — 조기종료·정규화 격자탐색·학습률 스케줄·
#    혼합정밀도를 붙인다. 규모를 키우면서 조기종료를 함께 넣는 이유는, 규모만 키우면
#    과적합과 과잉계산이 같이 커지기 때문이다.
# 3. **처리 단계마다 EDA를 붙인다**(22번에서 확립한 구조) — 원본 → 로그수익률 → 주기제거
#    → 타깃(실현변동성) → 모델군별 입력까지 각 단계에서 데이터가 어떻게 바뀌는지 수치와
#    그림으로 남긴다. 결과표만 있고 그 수치가 어떤 데이터에서 나왔는지 모르는 상태를 막는다.
#
# **Transformer 계열 2종은 제외한다(`EXCLUDED_MODELS`).** PatchTSTLike는 학습률 격자탐색과
# 패치 위치 임베딩 추가를 모두 시도해도 검증손실이 개선되지 않았다(진단 스크립트로 확인).
# ITransformerLike는 2026-09-28 최초 실행에선 포함해서 돌렸는데 20종목 최종 결과 17개
# 모델 중 16위(naive보다 열세)로 나왔고, 학습곡선을 다시 보니 PatchTSTLike와 같은 패턴
# (검증손실이 개선 추세 없이 요동)을 정도만 덜하게 갖고 있었다 — 계산 낭비를 막기 위해
# 이후 재실행부터는 함께 제외한다. 두 모델의 실제 성능 수치는 `24_maxscale_refit_20260928/`
# 결과에 그대로 남아 있다(이 판단의 근거 자료라 지우지 않는다). 나머지는 예정대로 비교한다.
#
# ## 비교 프로토콜 (23번과 달라진 부분을 먼저 밝힌다)
#
# 학습 구간을 다시 **내부학습 85% / 내부검증 15%**(시간순)로 나눈다. 학습이 필요한 모든
# 모델은 내부학습으로만 적합하고, 반복 횟수가 있는 모델(트리·딥러닝)은 내부검증 손실로
# 조기종료하며, 정규화 상수가 있는 모델(Ridge·커널)은 내부검증 QLIKE로 그 값을 고른다.
# GARCH 계열은 조정할 하이퍼파라미터가 없는 최대우도 추정이라 학습 구간 전체를 쓴다.
# 23번은 전 모델이 학습 구간 전체를 썼으므로 **23번 수치와 이 회차 수치를 직접 빼서
# 비교하지 않는다** — 같은 프로토콜 안에서 모델끼리 비교하는 용도다.
#
# ## 과거에 났던 사고를 코드로 막는 장치
#
# | 과거 사고 | 이번 회차의 방지 장치 |
# | :--- | :--- |
# | CUDA OOM으로 학습 중단(4·6·13번) | 배치 절반으로 줄여 재시도, 모델마다 캐시 비움 |
# | `num_workers>0` 공유메모리 붕괴(13번) | DataLoader 없이 GPU 상주 텐서로 직접 배치 |
# | 커널 모델 n×n 행렬로 메모리 폭발 | 표본 상한을 **가용 메모리에서 역산**, 예측은 분할 |
# | 끝나지 않는 과잉 계산 | 전체 마감시각 + 조기종료 + 작업별 소요시간 기록 |
# | `except: pass`가 실패를 삼켜 2종목 평균이 1위로 찍힘(23번) | 실패 전량 수집 후 적합 완결성 점검표를 성능표보다 앞에 |
# | 8종목만 돌고 20종목이라 표기(포스터) | 결과 행 수를 기대값과 대조, 어긋나면 보고서에 명시 |

# %%
from __future__ import annotations

import argparse
import gc
import os
import sys
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

warnings.filterwarnings("ignore")

# BLAS 스레드 상한 — numpy를 불러오기 전에 정해야 적용된다. 워커 6개가 저마다 32스레드를
# 잡으면 코어 경합으로 오히려 느려지고 메모리도 같이 뛴다.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import scipy.stats as st
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "test" / "scripts"))

TAG = "24_maxscale_refit_20260928"
# 파일 이름에도 출처를 박는다(AGENTS.md 2.3b) — 그림·표가 메일 첨부나 보고서 삽입으로
# 디렉터리 밖에 나가도 어느 실험 산출물인지 잃지 않게 한다.
STEM = "24_maxscale_refit"
IMG = ROOT / "test" / "images" / TAG
RES = ROOT / "test" / "results" / TAG
IMG.mkdir(parents=True, exist_ok=True)
RES.mkdir(parents=True, exist_ok=True)

ACCENT, ACCENT2, MUTED = "#0E9384", "#C85A3E", "#8FA1A8"

_LINES: list[str] = []


def emit(t: str = "") -> None:
    print(t, flush=True)
    _LINES.append(t)


# %% [markdown]
# ## 설정 — 최대 규모 하이퍼파라미터와 자원 상한
#
# "최대 규모"는 무한대가 아니라 **이 서버에서 안전하게 끝낼 수 있는 최대**다. 아래 값들은
# 전부 상한이며, 조기종료가 걸리면 그보다 일찍 멈춘다.

# %%
INNER_FRAC = 0.85          # 학습 구간 중 내부학습 비율(나머지는 조기종료·격자선택용 검증)
RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)

DL_HIDDEN = 128            # AGENTS.md 4.3 "Shallow but Wide" 권장 폭(64~128)의 상단
DL_MAX_EPOCHS = 60
DL_PATIENCE = 10
DL_BATCH = 2048            # OOM 시 절반씩 줄여 재시도
# 23번은 학습률 하나(2e-3)만 썼고 Transformer 2종이 naive보다 나빠 판단을 보류했다.
# 예비 실행에서 그 두 모델이 **1~2에폭에서 조기종료**되는 것을 확인했다 — 규모가 모자란
# 게 아니라 학습이 시작되지 않는 것이다. 그래서 학습률을 내부검증으로 고른다.
DL_LR_GRID = (2e-3, 5e-4, 1e-4)
SEQ_LEN_MAX = {"rnn": 96, "tfm": 192}

TREE_MAX_ROUNDS = 4000
TREE_LR = 0.02
TREE_EARLY_STOP = 200
TREE_THREADS = 4           # 워커마다 4스레드 × 워커 수 ≤ 물리 코어

KERNEL_MEM_BUDGET_GB = 1.5   # n×n 커널 행렬 하나에 허용할 메모리
SVR_MAX_N = 8000             # libsvm은 O(n²~n³)이라 커널릿지보다 낮게 잡는다
NYSTROEM_COMPONENTS = 2048

MS_RESTARTS, MS_MAXITER, MS_FIT_N = 3, 1400, 30000
TAR_RESTARTS, TAR_MAXITER = 2, 600
TAR_TAU_Q = (0.5, 0.65, 0.8, 0.9)

# 이번 회차에서 제외하는 모델 — 조용히 빼지 않고 사유를 남긴다(AGENTS.md 2.9j).
# PatchTSTLike: 학습률 3단 격자탐색(2e-3/5e-4/1e-4)과 패치 위치 임베딩 추가를 모두
# 시도했다(진단 스크립트로 확인: 학습손실은 15에폭에 걸쳐 0.985→0.839로 꾸준히 감소하는데
# 검증손실은 0.88~1.03 사이에서 개선 추세 없이 요동만 친다 — 그래디언트 소실이나 학습률
# 문제가 아니라 이 데이터·이 모델 조합에서 일반화가 안 되는 구조적 문제로 판단). 같은
# 은닉폭(128)의 GRU·LSTM은 정상적으로 여러 에폭에 걸쳐 검증손실이 개선되므로 "모델이
# 크다"만으로 설명되지 않는다.
# ITransformerLike: 처음엔 학습률 격자탐색으로 "고쳐졌다"고 판단했으나(2026-09-28 중간
# 진단에서 MASE가 naive 밑으로 내려간 사례가 있었다), 실제 20종목 전체 실행 결과 17개
# 모델 중 16위(naive보다 아래, MASE 1.16~1.17)로 확정됐다. 20에폭 학습곡선을 다시 그려
# 보니 PatchTSTLike와 사실상 같은 패턴(학습손실은 내려가는데 검증손실은 개선 추세 없이
# 요동)을 정도만 덜하게 보였을 뿐이었다 — 중간 판단이 우연히 뽑힌 좋은 지점 하나에
# 근거한 과장이었다. 두 Transformer 계열 모두 원 채널이 (r, |r|) 2개뿐이라 GARCH·트리·
# 커널이 받는 다중 lag 특성 없이 셀프어텐션만으로 시간구조를 학습해야 하는데, 그 경로가
# 이 데이터에서 일반화로 이어지지 않는 것으로 보인다(GRU·LSTM은 같은 2채널 입력으로도
# 정상 학습 — 순환 구조의 재귀적 귀납편향이 이 데이터엔 더 맞는다는 뜻).
# 더 깊은 아키텍처 규명(채널 확장, 다른 정규화 등)은 이번 연구 범위를 넘어 후속 과제로
# 남긴다. 계산량 낭비를 막기 위해 두 모델 다 다음 회차부터 제외한다(EXPERIMENT_LOG.md 기록).
EXCLUDED_MODELS = {
    "PatchTSTLike": "학습률 격자탐색+위치임베딩 후에도 검증손실 미개선(구조적 일반화 실패, 후속 과제)",
    "ITransformerLike": "20종목 전체 결과 16/17위(naive보다 열세) 확정, 학습곡선상 PatchTSTLike와 "
                        "동일한 일반화 실패 패턴(정도만 덜함) 재확인 — 계산 낭비 방지 위해 제외",
}

MIN_FREE_RAM_GB = 3.0       # 이 아래로 떨어지면 정리 후 진행


def kernel_max_n(budget_gb: float = KERNEL_MEM_BUDGET_GB) -> int:
    """n×n float64 커널 행렬이 예산 안에 들어오는 최대 n.

    커널 릿지는 학습 표본 수의 제곱에 비례하는 행렬을 만들고 그것을 분해한다. 표본을
    무작정 늘리면 메모리가 아니라 **행렬 분해 시간**(O(n³))이 먼저 터지므로, 메모리
    예산에서 역산한 상한을 그대로 계산량 상한으로도 쓴다.
    """
    return int(np.sqrt(budget_gb * 1024 ** 3 / 8))


def free_ram_gb() -> float:
    return psutil.virtual_memory().available / 1024 ** 3


def relieve_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# %% [markdown]
# ## 23번에서 승계하는 계산 정의
#
# 데이터 적재·주기 제거·타깃 정의·insanity filter·특성 생성·평가지표는 23번 드라이버의
# 함수를 **그대로 불러 쓴다**. 같은 정의를 두 벌 관리하면 두 회차 수치가 왜 다른지
# 추적할 수 없게 되므로, 정의는 한 곳에만 둔다.

# %%
def _load_m23():
    import importlib.util
    path = ROOT / "test" / "models" / "23_volatility_model_comparison_test.py"
    spec = importlib.util.spec_from_file_location("m23", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M23 = _load_m23()
BPD = M23.BPD
HORIZON = M23.HORIZON
TRAIN_FRAC = M23.TRAIN_FRAC


def seqs_fast(r: np.ndarray, seq_len: int, revin: bool) -> np.ndarray:
    """시퀀스 입력 — 23번 `seqs()`와 같은 배열을 만들되 파이썬 루프를 걷어냈다.

    23번은 `for i in range(n)`으로 10만 번 슬라이싱했다. 같은 결과를 앞쪽에 0을 덧댄 뒤
    `sliding_window_view`로 한 번에 얻을 수 있다(복사 1회). 이번 회차는 시퀀스 길이를
    최대 192까지 늘리므로 이 차이가 그대로 실행시간이 된다.
    """
    ch = np.column_stack([r, np.abs(r)]).astype(np.float32)
    n, c = ch.shape
    pad = np.zeros((seq_len - 1, c), np.float32)
    padded = np.concatenate([pad, ch], axis=0)
    X = np.lib.stride_tricks.sliding_window_view(padded, (seq_len, c))[:, 0]
    X = np.ascontiguousarray(X)
    assert X.shape == (n, seq_len, c)
    if revin:
        mu = X.mean(axis=1, keepdims=True)
        sd = X.std(axis=1, keepdims=True)
        X = (X - mu) / np.where(sd < 1e-6, 1.0, sd)
    else:
        mu = X.reshape(-1, c).mean(0)
        sd = X.reshape(-1, c).std(0)
        X = (X - mu) / np.where(sd < 1e-6, 1.0, sd)
    return X.astype(np.float32)


# %% [markdown]
# ## 분할·마스크 준비 — 모든 모델이 같은 행을 본다
#
# 23번 `run_one()`의 분할 로직을 그대로 쓰되, 여기에 내부학습/내부검증 경계를 하나 더
# 얹는다. 이 함수는 (종목, 주기제거)만으로 결정되므로 CPU 패스와 GPU 패스에서 각각
# 호출해도 **반드시 같은 값**이 나온다(두 패스의 결과를 나중에 합칠 수 있는 근거).

# %%
def prepare(ticker: str, deper: bool) -> dict | None:
    close = M23.load_close(ticker)
    r_s = np.log(close).diff().dropna()
    if deper:
        r_s = M23.remove_periodicity(r_s)
    r = r_s.to_numpy()
    n = len(r)
    if n < 20000:
        return None

    raw_r = np.log(close).diff().dropna().to_numpy()
    fut, pas = M23.targets(r, HORIZON)
    valid = np.isfinite(fut) & np.isfinite(pas)
    valid[:7 * BPD + 10] = False
    valid_tr = valid & (fut > 0) & (pas > 0)

    idxv = np.where(valid)[0]
    split_idx = int(idxv[int(len(idxv) * TRAIN_FRAC)])
    tr_mask = valid_tr.copy(); tr_mask[split_idx:] = False
    te_mask = valid.copy(); te_mask[:split_idx] = False
    fit_mask = tr_mask | te_mask
    sp = int(tr_mask.sum())
    sp_in = int(sp * INNER_FRAC)

    y = np.log(fut[fit_mask] + 1e-14)
    rv_act_all = fut[fit_mask]
    rv_nai_all = pas[fit_mask]
    ii = np.arange(0, len(y) - sp, HORIZON)      # 겹치지 않는 검증 표본

    rv_tr = rv_act_all[:sp]
    f_lo, f_hi = np.percentile(rv_tr, [0.5, 99.5])

    # 내부검증용 필터 범위는 내부학습 구간만으로 다시 잡는다(검증 정보 유입 차단)
    rv_in = rv_act_all[:sp_in]
    i_lo, i_hi = np.percentile(rv_in, [0.5, 99.5])

    return dict(
        ticker=ticker, deper=deper, r=r, n=n, close=close,
        fut=fut, pas=pas, valid=valid, fit_mask=fit_mask, tr_mask=tr_mask,
        te_mask=te_mask, split_idx=split_idx, sp=sp, sp_in=sp_in,
        y=y, rv_act_all=rv_act_all, rv_nai_all=rv_nai_all, ii=ii,
        rv_act=rv_act_all[sp:][ii], rv_nai=rv_nai_all[sp:][ii],
        rv_inner_act=rv_act_all[sp_in:sp], rv_inner_nai=rv_nai_all[sp_in:sp],
        f_lo=float(f_lo), f_hi=float(f_hi), f_mean=float(rv_tr.mean()),
        i_lo=float(i_lo), i_hi=float(i_hi), i_mean=float(rv_in.mean()),
        zero_ret=float((raw_r == 0).mean()),
        zero_rv=float(np.mean(fut[valid] <= 0)),
        tick_ratio=M23.tick_ratio(close),
        drop_share=float(1 - sp / max(int((valid & (np.arange(n) < split_idx)).sum()), 1)),
    )


def inner_qlike(P: dict, pred_inner: np.ndarray) -> float:
    """내부검증 QLIKE — 정규화 상수를 고를 때 쓰는 선택 기준."""
    p, _ = M23.insanity_filter(pred_inner, P["i_lo"], P["i_hi"], P["i_mean"])
    return M23.qlike(P["rv_inner_act"] ** 2, p ** 2)


def scorer(P: dict):
    """모델 하나의 검증 예측을 성능 행으로 바꾼다(예측값도 함께 보관)."""
    preds: dict[str, np.ndarray] = {}
    rows: list[dict] = []

    def score(name: str, family: str, rv_pred: np.ndarray, note: str = "", extra: dict | None = None):
        p = rv_pred[P["ii"]] if len(rv_pred) == len(P["y"]) - P["sp"] else rv_pred
        p, trim = M23.insanity_filter(p, P["f_lo"], P["f_hi"], P["f_mean"])
        preds[name] = p.astype(np.float32)
        row = {"종목": P["ticker"], "주기제거": P["deper"], "모델": name, "계열": family,
               "비고": note, "트리밍비율": trim,
               **M23.evaluate(P["rv_act"], p, P["rv_nai"])}
        row.update(extra or {})
        rows.append(row)
        return row

    return score, rows, preds


# %% [markdown]
# ## CPU 패스 — 통계·선형·커널·트리 (종목별 병렬)
#
# GARCH 계열의 최대우도 추정은 순수 파이썬 스칼라 루프라 한 코어만 쓴다. 40개 작업을
# 순차로 돌리면 코어 31개가 노는 동안 7시간이 걸리므로 프로세스 풀로 나눈다. 워커 수는
# **가용 메모리 ÷ 워커당 예상 사용량**으로 정한다 — 코어 수로 정하면 메모리가 먼저 터진다.

# %%
def run_cpu_job(ticker: str, deper: bool, quick: bool) -> dict:
    """한 (종목, 주기제거)에서 CPU 모델 전부를 적합한다. 예외는 삼키지 않고 모아서 반환."""
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
    P = prepare(ticker, deper)
    if P is None:
        return dict(ticker=ticker, deper=deper, rows=[], preds={}, fails=[],
                    skipped="표본 부족", timings={}, meta={})

    score, rows, preds = scorer(P)
    fails: list[tuple[str, str, str]] = []
    timings: dict[str, float] = {}
    meta: dict[str, object] = {}
    sp, sp_in = P["sp"], P["sp_in"]
    r, valid = P["r"], P["fit_mask"]
    y = P["y"]
    y_in, y_val = y[:sp_in], y[sp_in:sp]

    def stamp(name, t):
        timings[name] = time.time() - t

    score("naive", "기준선", P["rv_nai"])

    # ── 선형성 근거(대표 종목만 그림) ──
    Xl = M23.feats_log(r)[valid]
    Xh = M23.feats_har(r)[valid]
    Xt = M23.feats_tree(r)[valid]
    plot_path = IMG / f"{STEM}_figL_linearity_{'deper' if deper else 'raw'}.png" if ticker == "KRW-BTC" else None
    try:
        lin_ev = M23.linearity_evidence(Xl[:sp], y[:sp], plot_path,
                                        title=f"{ticker} · 주기제거 {'후' if deper else '전'}")
    except Exception as e:
        lin_ev = {}
        fails.append(("linearity_evidence", type(e).__name__, str(e)[:160]))
    note_lin = ("선형성 기각(RESET p="
                f"{lin_ev.get('RESET_p', float('nan')):.4g}, BDS p={lin_ev.get('BDS_p', float('nan')):.4g})")

    # ── 선형·HAR ──
    for nm, fam, Xsrc in (("HAR-RV", "벤치마크", Xh), ("Linear", "선형", Xl)):
        t = time.time()
        try:
            sc = StandardScaler().fit(Xsrc[:sp_in])
            mdl = LinearRegression().fit(sc.transform(Xsrc[:sp_in]), y_in)
            score(nm, fam, np.exp(mdl.predict(sc.transform(Xsrc[sp:]))),
                  "" if fam == "벤치마크" else note_lin)
        except Exception as e:
            fails.append((nm, type(e).__name__, str(e)[:160]))
        stamp(nm, t)

    # ── Ridge: 정규화 상수를 내부검증 QLIKE로 고른다 ──
    t = time.time()
    try:
        sc = StandardScaler().fit(Xl[:sp_in])
        Zin, Zval, Zte = sc.transform(Xl[:sp_in]), sc.transform(Xl[sp_in:sp]), sc.transform(Xl[sp:])
        best = None
        for a in RIDGE_ALPHAS:
            mdl = Ridge(alpha=a).fit(Zin, y_in)
            q = inner_qlike(P, np.exp(mdl.predict(Zval)))
            if best is None or q < best[0]:
                best = (q, a, mdl)
        meta["ridge_alpha"] = best[1]
        score("Ridge", "선형", np.exp(best[2].predict(Zte)), f"{note_lin} · alpha={best[1]}")
    except Exception as e:
        fails.append(("Ridge", type(e).__name__, str(e)[:160]))
    stamp("Ridge", t)

    # ── 커널 3종 ──
    scx = StandardScaler().fit(Xl[:sp_in])
    scy = StandardScaler().fit(y_in.reshape(-1, 1))
    Zin, Zval, Zte = scx.transform(Xl[:sp_in]), scx.transform(Xl[sp_in:sp]), scx.transform(Xl[sp:])
    y_in_s = scy.transform(y_in.reshape(-1, 1)).ravel()

    def unscale(v):
        return scy.inverse_transform(np.asarray(v).reshape(-1, 1)).ravel()

    def chunked_predict(mdl, Z, chunk=4000):
        """커널 예측 행렬은 n_test × n_train이라 한 번에 만들면 메모리가 터진다."""
        return np.concatenate([mdl.predict(Z[i:i + chunk]) for i in range(0, len(Z), chunk)])

    kmax = kernel_max_n()
    n_kr = min(kmax, sp_in) if not quick else min(1500, sp_in)
    n_svr = min(SVR_MAX_N, sp_in) if not quick else min(1200, sp_in)
    meta["kernel_n"] = n_kr
    meta["svr_n"] = n_svr

    t = time.time()
    try:
        sl = slice(sp_in - n_kr, sp_in)
        gamma0 = 1.0 / Zin.shape[1]
        best = None
        for a in (0.1, 1.0, 10.0):
            for g in (gamma0 / 4, gamma0, gamma0 * 4):
                mdl = KernelRidge(kernel="rbf", alpha=a, gamma=g).fit(Zin[sl], y_in_s[sl])
                q = inner_qlike(P, np.exp(unscale(chunked_predict(mdl, Zval))))
                if best is None or q < best[0]:
                    best = (q, a, g, mdl)
        meta["kr_alpha"], meta["kr_gamma"] = best[1], best[2]
        score("KernelRidge-RBF", "커널", np.exp(unscale(chunked_predict(best[3], Zte))),
              f"선형성 불필요 · n={n_kr} · alpha={best[1]} · gamma={best[2]:.4g}")
    except Exception as e:
        fails.append(("KernelRidge-RBF", type(e).__name__, str(e)[:160]))
    stamp("KernelRidge-RBF", t)

    t = time.time()
    try:
        sl = slice(sp_in - n_svr, sp_in)
        best = None
        for c in (1.0, 10.0, 100.0):
            mdl = SVR(kernel="rbf", C=c, epsilon=0.05, cache_size=1000).fit(Zin[sl], y_in_s[sl])
            q = inner_qlike(P, np.exp(unscale(chunked_predict(mdl, Zval))))
            if best is None or q < best[0]:
                best = (q, c, mdl)
        meta["svr_C"] = best[1]
        score("SVR-RBF", "커널", np.exp(unscale(chunked_predict(best[2], Zte))),
              f"선형성 불필요 · n={n_svr} · C={best[1]}")
    except Exception as e:
        fails.append(("SVR-RBF", type(e).__name__, str(e)[:160]))
    stamp("SVR-RBF", t)

    t = time.time()
    try:
        nc = NYSTROEM_COMPONENTS if not quick else 256
        best = None
        for a in RIDGE_ALPHAS:
            mdl = make_pipeline(Nystroem(kernel="rbf", n_components=nc, random_state=0),
                                Ridge(alpha=a)).fit(Zin, y_in_s)
            q = inner_qlike(P, np.exp(unscale(mdl.predict(Zval))))
            if best is None or q < best[0]:
                best = (q, a, mdl)
        meta["nystroem_alpha"] = best[1]
        score("Nystroem+Ridge", "커널", np.exp(unscale(best[2].predict(Zte))),
              f"선형성 불필요 · 성분 {nc}개 · alpha={best[1]}")
    except Exception as e:
        fails.append(("Nystroem+Ridge", type(e).__name__, str(e)[:160]))
    stamp("Nystroem+Ridge", t)

    # ── 트리 3종: 내부검증 조기종료 ──
    rounds = TREE_MAX_ROUNDS if not quick else 60
    esr = TREE_EARLY_STOP if not quick else 10
    Ti, Tv, Tt = Xt[:sp_in], Xt[sp_in:sp], Xt[sp:]

    t = time.time()
    try:
        mdl = lgb.LGBMRegressor(n_estimators=rounds, learning_rate=TREE_LR, max_depth=8,
                                num_leaves=127, subsample=0.8, subsample_freq=1,
                                colsample_bytree=0.8, min_child_samples=40,
                                n_jobs=TREE_THREADS, verbosity=-1, random_state=0)
        mdl.fit(Ti, y_in, eval_set=[(Tv, y_val)], eval_metric="l2",
                callbacks=[lgb.early_stopping(esr, verbose=False)])
        meta["lgbm_rounds"] = int(mdl.best_iteration_ or rounds)
        score("LightGBM", "트리", np.exp(mdl.predict(Tt)),
              f"가정 없음(비모수) · 조기종료 {meta['lgbm_rounds']}그루")
    except Exception as e:
        fails.append(("LightGBM", type(e).__name__, str(e)[:160]))
    stamp("LightGBM", t)

    t = time.time()
    try:
        mdl = xgb.XGBRegressor(n_estimators=rounds, learning_rate=TREE_LR, max_depth=8,
                               subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
                               tree_method="hist", device="cpu", n_jobs=TREE_THREADS,
                               early_stopping_rounds=esr, verbosity=0, random_state=0)
        mdl.fit(Ti, y_in, eval_set=[(Tv, y_val)], verbose=False)
        meta["xgb_rounds"] = int(getattr(mdl, "best_iteration", rounds) or rounds)
        score("XGBoost", "트리", np.exp(mdl.predict(Tt)),
              f"가정 없음(비모수) · 조기종료 {meta['xgb_rounds']}그루")
    except Exception as e:
        fails.append(("XGBoost", type(e).__name__, str(e)[:160]))
    stamp("XGBoost", t)

    t = time.time()
    try:
        # sklearn HistGBM의 내장 조기종료는 검증 분할을 **무작위로** 뽑아 시계열에서
        # 낙관적으로 멈춘다. 대신 warm_start로 반복을 이어 붙이며 내부검증 손실을 직접
        # 감시하고, 손실이 가장 낮았던 시점의 예측을 그대로 보관한다(재적합 없음 —
        # 단계마다 처음부터 다시 학습하면 같은 결과에 10배 넘는 계산이 든다).
        best_iter, best_loss, patience = 0, np.inf, 0
        best_pred = None
        step = max(50, rounds // 20)
        mdl = HistGradientBoostingRegressor(max_iter=step, learning_rate=TREE_LR, max_depth=8,
                                            max_leaf_nodes=127, min_samples_leaf=40,
                                            early_stopping=False, warm_start=True, random_state=0)
        for it in range(step, rounds + 1, step):
            mdl.set_params(max_iter=it)
            mdl.fit(Ti, y_in)
            loss = float(np.mean((mdl.predict(Tv) - y_val) ** 2))
            if loss < best_loss - 1e-9:
                best_loss, best_iter, patience = loss, it, 0
                best_pred = mdl.predict(Tt)
            else:
                patience += 1
                if patience >= 2:
                    break
        meta["hist_rounds"] = best_iter
        score("HistGBM", "트리", np.exp(best_pred),
              f"가정 없음(비모수) · 조기종료 {best_iter}회(격자 {step}회 단위)")
    except Exception as e:
        fails.append(("HistGBM", type(e).__name__, str(e)[:160]))
    stamp("HistGBM", t)

    # ── GARCH 계열 3종 (원 수익률, 학습 구간 전체 사용) ──
    gv_full = None
    t = time.time()
    try:
        gv_full = M23.garch_cond_vol(r, P["split_idx"], dist="t")
        score("GARCH-t", "통계", gv_full[P["te_mask"]] * np.sqrt(HORIZON), "t분포로 두꺼운 꼬리 보완")
    except Exception as e:
        fails.append(("GARCH-t", type(e).__name__, str(e)[:160]))
    stamp("GARCH-t", t)

    r_pct = r * 100.0
    t = time.time()
    try:
        restarts = MS_RESTARTS if not quick else 1
        it = MS_MAXITER if not quick else 120
        fitn = MS_FIT_N if not quick else 8000
        _, _, info = rg.fit_ms_garch(r_pct, P["split_idx"], n_restarts=restarts,
                                     maxiter=it, max_fit_n=fitn)
        vol = np.sqrt(np.clip(info["h_pred"], 0, None)) / 100.0
        meta.update(ms_p11=info["p11"], ms_p22=info["p22"], ms_nu=info["nu"])
        score("MS-GARCH", "통계", vol[P["te_mask"]] * np.sqrt(HORIZON),
              f"마르코프 2국면 전환 · 잔류확률 {info['p11']:.3f}/{info['p22']:.3f} · nu={info['nu']:.2f}")
    except Exception as e:
        fails.append(("MS-GARCH", type(e).__name__, str(e)[:160]))
    stamp("MS-GARCH", t)

    t = time.time()
    try:
        pas = P["pas"]
        switch_lagged = np.empty(P["n"])
        switch_lagged[0] = np.nan_to_num(pas[0], nan=0.0)
        switch_lagged[1:] = np.nan_to_num(pas[:-1], nan=0.0)
        taus = np.quantile(switch_lagged[P["tr_mask"]], list(TAR_TAU_Q))
        restarts = TAR_RESTARTS if not quick else 1
        it = TAR_MAXITER if not quick else 100
        _, tau, _, info = rg.fit_tar_garch(r_pct, switch_lagged, P["split_idx"], taus,
                                           n_restarts=restarts, maxiter=it)
        vol = np.sqrt(np.clip(info["h_pred"], 0, None)) / 100.0
        meta["tar_tau"] = float(tau)
        score("TAR-GARCH", "통계", vol[P["te_mask"]] * np.sqrt(HORIZON),
              f"임계값 전환(Hansen 프로파일 우도) · tau={tau:.5f}")
    except Exception as e:
        fails.append(("TAR-GARCH", type(e).__name__, str(e)[:160]))
    stamp("TAR-GARCH", t)

    # ── 하이브리드: GARCH 조건부분산을 트리 특성으로 결합 ──
    t = time.time()
    if gv_full is not None:
        try:
            gcol = gv_full[P["fit_mask"]] * np.sqrt(HORIZON)
            Xhyb = np.column_stack([Xt, np.log(gcol + 1e-12)])
            mdl = lgb.LGBMRegressor(n_estimators=rounds, learning_rate=TREE_LR, max_depth=8,
                                    num_leaves=127, subsample=0.8, subsample_freq=1,
                                    colsample_bytree=0.8, min_child_samples=40,
                                    n_jobs=TREE_THREADS, verbosity=-1, random_state=0)
            mdl.fit(Xhyb[:sp_in], y_in, eval_set=[(Xhyb[sp_in:sp], y_val)], eval_metric="l2",
                    callbacks=[lgb.early_stopping(esr, verbose=False)])
            score("GARCH+LightGBM", "하이브리드", np.exp(mdl.predict(Xhyb[sp:])),
                  f"통계 구조 + 비모수 학습 · 조기종료 {mdl.best_iteration_ or rounds}그루")
        except Exception as e:
            fails.append(("GARCH+LightGBM", type(e).__name__, str(e)[:160]))
    else:
        fails.append(("GARCH+LightGBM", "SkippedDependency", "GARCH-t 적합 실패로 입력 특성 없음"))
    stamp("GARCH+LightGBM", t)

    for rw in rows:
        rw["영수익률비율"] = P["zero_ret"]
        rw["RV0비율"] = P["zero_rv"]
        rw["학습제외비율"] = P["drop_share"]
        rw["호가가격비"] = P["tick_ratio"]

    lin_ev = {**lin_ev, "종목": ticker, "주기제거": deper}
    return dict(ticker=ticker, deper=deper, rows=rows, preds=preds, fails=fails,
                lin_ev=lin_ev, timings=timings, meta=meta, skipped=None,
                rv_act=P["rv_act"].astype(np.float32), elapsed=time.time() - t0)


# %% [markdown]
# ## GPU 패스 — 딥러닝 4종
#
# 배치를 GPU에 통째로 올려 두고 인덱싱으로 미니배치를 만든다. DataLoader를 쓰지 않는
# 이유는 13번에서 `num_workers>0`이 공유메모리를 터뜨린 전례가 있어서다. OOM이 나면
# 배치를 절반으로 줄여 다시 시도하고, 모델마다 캐시를 비운다.

# %%
def train_dl_once(name: str, Xtr, ytr, Xval, yval, Xte, dev: str,
                  max_epochs: int, patience: int, batch: int, lr: float
                  ) -> tuple[np.ndarray, float, dict]:
    from engine.models import make_model

    torch.manual_seed(0)
    model = make_model(name, Xtr.shape[1], Xtr.shape[2], DL_HIDDEN).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)
    lossf = nn.MSELoss()
    use_amp = dev == "cuda"

    xt = torch.as_tensor(Xtr, device=dev)
    yt = torch.as_tensor(ytr, dtype=torch.float32, device=dev)
    xv = torch.as_tensor(Xval, device=dev)
    yv = torch.as_tensor(yval, dtype=torch.float32, device=dev)

    def infer(x, bs):
        model.eval()
        out = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            for i in range(0, len(x), bs):
                out.append(model(x[i:i + bs]).float())
        return torch.cat(out)

    best_loss, best_state, best_epoch, bad = np.inf, None, 0, 0
    bs = batch
    epoch = 0
    while epoch < max_epochs:
        model.train()
        perm = torch.randperm(len(xt), device=dev)
        try:
            for i in range(0, len(xt), bs):
                b = perm[i:i + bs]
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    loss = lossf(model(xt[b]), yt[b])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
        except torch.cuda.OutOfMemoryError:
            if bs <= 128:
                raise
            bs //= 2
            relieve_memory()
            print(f"    [oom-retry] {name} batch_size -> {bs}", flush=True)
            continue
        sched.step()
        epoch += 1
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
    xe = torch.as_tensor(Xte, device=dev)
    pred = infer(xe, bs).cpu().numpy().ravel()
    info = dict(epochs_run=epoch, best_epoch=best_epoch, batch=bs, lr=lr, val_mse=best_loss)

    del xt, yt, xv, yv, xe, model, best_state
    relieve_memory()
    return pred, best_loss, info


def train_dl_maxscale(name: str, Xtr, ytr, Xval, yval, Xte, dev: str,
                      max_epochs: int, patience: int, batch: int, lr_grid=DL_LR_GRID
                      ) -> tuple[np.ndarray, dict]:
    """학습률을 내부검증 손실로 고른다 — 나머지는 조기종료가 알아서 멈춘다."""
    best = None
    tried = []
    for lr in lr_grid:
        pred, vloss, info = train_dl_once(name, Xtr, ytr, Xval, yval, Xte, dev,
                                          max_epochs, patience, batch, lr)
        tried.append((lr, info["best_epoch"], vloss))
        if best is None or vloss < best[1]:
            best = (pred, vloss, info)
    best[2]["lr_tried"] = tried
    return best[0], best[2]


def run_gpu_job(ticker: str, deper: bool, quick: bool, dev: str) -> dict:
    from sklearn.preprocessing import StandardScaler

    t0 = time.time()
    P = prepare(ticker, deper)
    if P is None:
        return dict(ticker=ticker, deper=deper, rows=[], preds={}, fails=[],
                    skipped="표본 부족", timings={}, meta={})

    score, rows, preds = scorer(P)
    fails: list[tuple[str, str, str]] = []
    timings: dict[str, float] = {}
    meta: dict[str, object] = {}
    sp, sp_in = P["sp"], P["sp_in"]
    y = P["y"]
    y_in = y[:sp_in]
    scy = StandardScaler().fit(y_in.reshape(-1, 1))
    y_in_s = scy.transform(y_in.reshape(-1, 1)).ravel()
    y_val_s = scy.transform(y[sp_in:sp].reshape(-1, 1)).ravel()

    dl_list = [(nm, grp, revin) for nm, grp, revin in
               [("GRU", "rnn", False), ("LSTM", "rnn", False),
                ("PatchTSTLike", "tfm", True), ("ITransformerLike", "tfm", True)]
               if nm not in EXCLUDED_MODELS]
    max_ep = DL_MAX_EPOCHS if not quick else 3
    pat = DL_PATIENCE if not quick else 2

    seq_cache: dict[tuple[str, bool], np.ndarray] = {}
    for nm, grp, revin in dl_list:
        t = time.time()
        try:
            if free_ram_gb() < MIN_FREE_RAM_GB:
                seq_cache.clear()
                relieve_memory()
            key = (grp, revin)
            if key not in seq_cache:
                sl = SEQ_LEN_MAX[grp] if not quick else 16
                seq_cache[key] = seqs_fast(P["r"], sl, revin)[P["fit_mask"]]
            Xs = seq_cache[key]
            grid = DL_LR_GRID if not quick else (DL_LR_GRID[0],)
            pred_s, info = train_dl_maxscale(
                nm, Xs[:sp_in], y_in_s, Xs[sp_in:sp], y_val_s, Xs[sp:], dev,
                max_ep, pat, DL_BATCH, grid)
            meta[f"{nm}_epochs"] = info["best_epoch"]
            meta[f"{nm}_batch"] = info["batch"]
            meta[f"{nm}_lr"] = info["lr"]
            unscaled = scy.inverse_transform(pred_s.reshape(-1, 1)).ravel()
            score(nm, "딥러닝", np.exp(unscaled),
                  f"{'RevIN(시퀀스별 정규화)' if revin else '전역 표준화'} · 학습률 {info['lr']:g} · "
                  f"{info['best_epoch']}/{info['epochs_run']}에폭에서 조기종료 · 배치 {info['batch']}")
        except Exception as e:
            fails.append((nm, type(e).__name__, str(e)[:160]))
            traceback.print_exc()
        timings[nm] = time.time() - t
        relieve_memory()

    seq_cache.clear()
    relieve_memory()
    for rw in rows:
        rw["영수익률비율"] = P["zero_ret"]
        rw["RV0비율"] = P["zero_rv"]
        rw["학습제외비율"] = P["drop_share"]
        rw["호가가격비"] = P["tick_ratio"]
    return dict(ticker=ticker, deper=deper, rows=rows, preds=preds, fails=fails,
                timings=timings, meta=meta, skipped=None,
                rv_act=P["rv_act"].astype(np.float32), elapsed=time.time() - t0)


# %% [markdown]
# ## 처리 단계별 EDA — 결과와 같은 회차에서 함께 낸다
#
# 22번에서 확립한 구조다. 모델 성능표만 내면 그 수치가 어떤 데이터에서 나왔는지 알 수
# 없고, 실제로 그 상태에서 방향이 어긋난 적이 있다(2026-09-07 지적). 이번 회차가 실제로
# 밟는 처리 순서를 그대로 따라가며 각 단계의 기초통계·검정·그림을 남긴다.

# %%
def diagnose(x: np.ndarray, label: str) -> dict:
    from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
    from statsmodels.tsa.stattools import adfuller, kpss

    v = np.asarray(x, dtype=float)
    v = v[np.isfinite(v)]
    s = v[-40000:]                                    # 검정은 최근 구간으로(계산량 통제)
    out = {"단계": label, "n": len(v), "평균": float(v.mean()), "표준편차": float(v.std()),
           "왜도": float(st.skew(v)), "초과첨도": float(st.kurtosis(v))}
    for key, fn in (("ADF p", lambda: adfuller(s, autolag="AIC")[1]),
                    ("KPSS p", lambda: kpss(s, regression="c", nlags="auto")[1]),
                    ("ARCH p", lambda: het_arch(s, nlags=12)[1]),
                    ("LB(|x|) p", lambda: float(acorr_ljungbox(np.abs(s), lags=[24],
                                                               return_df=True)["lb_pvalue"].iloc[0]))):
        try:
            out[key] = float(fn())
        except Exception:
            out[key] = np.nan
    return out


def acf_at(x: np.ndarray, lag: int) -> float:
    v = np.asarray(x, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) <= lag:
        return np.nan
    a, b = v[:-lag], v[lag:]
    return float(np.corrcoef(a, b)[0, 1])


def stage_eda(tickers: list[str], rep: str) -> dict:
    """원본 → 로그수익률 → 주기제거 → 타깃 → 모델 입력, 매 단계 수치와 그림."""
    close = M23.load_close(rep)
    r0 = np.log(close).diff().dropna()
    r1 = M23.remove_periodicity(r0)
    fut0, _ = M23.targets(r0.to_numpy(), HORIZON)
    fut1, _ = M23.targets(r1.to_numpy(), HORIZON)

    stages = {"S0 로그가격(레벨)": np.log(close).to_numpy(),
              "S1 로그수익률(1차 차분)": r0.to_numpy(),
              "S2 +주기 제거": r1.to_numpy(),
              "S3 타깃 = 1시간 실현변동성": fut1[np.isfinite(fut1)],
              "S4 타깃의 로그": np.log(fut1[np.isfinite(fut1) & (fut1 > 0)])}
    dd = pd.DataFrame([diagnose(v, k) for k, v in stages.items()])

    emit("## 1. 처리 단계별 EDA — 이 회차가 실제로 밟는 순서 그대로")
    emit()
    emit(f"대표 종목 **{rep}** 기준이다. 성능표에 앞서, 모델이 보게 될 데이터가 각 처리에서 "
         "어떻게 바뀌는지를 먼저 수치로 고정한다.")
    emit()
    emit("| 단계 | n | 표준편차 | 왜도 | 초과첨도 | ADF p | KPSS p | ARCH p | LB(&#124;x&#124;) p |")
    emit("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for _, x in dd.iterrows():
        emit(f"| {x['단계']} | {int(x['n']):,} | {M23.num(x['표준편차'], 6)} | {M23.num(x['왜도'], 3)} | "
             f"**{M23.num(x['초과첨도'], 1)}** | {M23.num(x['ADF p'], 4)} | {M23.num(x['KPSS p'], 4)} | "
             f"{M23.num(x['ARCH p'], 4)} | {M23.num(x['LB(|x|) p'], 4)} |")
    emit()
    emit("**읽는 법** — ADF p<0.05면 정상, KPSS p>0.05면 정상이다. 두 검정은 귀무가설이 서로 "
         "반대라서 함께 봐야 한다. ARCH p<0.05는 조건부 이분산이 있다는 뜻이고, 이것이 변동성 "
         "모델을 쓸 근거다. LB(|x|) p<0.05는 변동성 자기상관(군집)이 있다는 뜻이며, 이것이 "
         "우리가 예측하려는 대상 그 자체다.")
    emit()
    emit("S3에서 S4로 로그를 취하는 이유는 실현변동성이 강하게 우편향이기 때문이다. 다만 "
         "**GARCH 계열은 이 로그 타깃을 쓰지 않고 원 수익률을 직접 쓴다** — 23번에서 저가 "
         "코인의 실현변동성 0 구간이 로그 타깃에서 -32.2가 되어 로그축 모델만 선택적으로 "
         "무너졌던 사고가 여기서 갈린다.")
    emit()

    # ── 그림 1: 단계별 시계열·분포·자기상관 ──
    idx_map = {0: close.index, 1: r0.index, 2: r0.index}
    fig, ax = plt.subplots(5, 3, figsize=(18, 19))
    for i, (name, v) in enumerate(stages.items()):
        v = np.asarray(v, float)
        v = v[np.isfinite(v)]
        xi = idx_map.get(i)
        a = ax[i, 0]
        if xi is not None:
            a.plot(xi[-len(v):], v, lw=0.35, color=ACCENT2 if i else MUTED)
        else:
            a.plot(np.arange(len(v)), v, lw=0.35, color=ACCENT2)
        a.set_title(f"(S{i}-1) {name} — 시계열", fontsize=11)
        a.grid(alpha=.2)

        a = ax[i, 1]
        lim = np.quantile(np.abs(v), 0.999)
        sel = v[np.abs(v) < lim] if i in (1, 2) else v[v < np.quantile(v, 0.999)]
        a.hist(sel, bins=150, density=True, color=ACCENT, alpha=.65)
        if i in (1, 2):
            xs = np.linspace(-lim, lim, 400)
            a.plot(xs, st.norm.pdf(xs, v.mean(), v.std()), "r--", lw=1.5, label="정규분포")
            a.legend(fontsize=8)
        a.set_title(f"(S{i}-2) 분포 — 초과첨도 {st.kurtosis(v):.1f}", fontsize=11)

        a = ax[i, 2]
        lags = list(range(1, 97))
        a.plot(lags, [acf_at(np.abs(v), L) for L in lags], color=ACCENT2, lw=1.6, label="|x| 자기상관")
        a.plot(lags, [acf_at(v, L) for L in lags], color=MUTED, lw=1.1, label="x 자기상관")
        a.axhline(0, color="k", lw=.7)
        a.set_title(f"(S{i}-3) 자기상관 (시차 1~96봉)", fontsize=11)
        a.legend(fontsize=8); a.grid(alpha=.2)
    fig.suptitle(f"그림 1. 처리 단계별 EDA — {rep}, 원본에서 모델 입력까지", fontsize=16, y=0.999)
    fig.tight_layout()
    f1 = IMG / f"{STEM}_fig1_stage_eda.png"
    fig.savefig(f1, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    emit(f"![그림 1](../../images/{TAG}/{f1.name})")
    emit()

    # ── 그림 2: 주기 제거 효과 + 20종목 타깃 분포 ──
    slot_h = r0.index.hour
    prof0 = pd.Series(r0.abs().to_numpy()).groupby(slot_h).mean()
    prof1 = pd.Series(np.abs(r1.to_numpy())).groupby(slot_h).mean()
    ratio0 = float(prof0.max() / prof0.min())
    ratio1 = float(prof1.max() / prof1.min())

    tgt_stats = []
    for tk in tickers:
        c = M23.load_close(tk)
        rr = np.log(c).diff().dropna()
        rr = M23.remove_periodicity(rr)
        f_, _ = M23.targets(rr.to_numpy(), HORIZON)
        f_ = f_[np.isfinite(f_)]
        tgt_stats.append(dict(종목=tk.replace("KRW-", ""), 중앙값=float(np.median(f_)),
                              분위95=float(np.quantile(f_, 0.95)), 최대=float(f_.max()),
                              RV0비율=float(np.mean(f_ <= 0))))
    ts = pd.DataFrame(tgt_stats).sort_values("중앙값")

    fig, ax = plt.subplots(1, 3, figsize=(19, 5.6))
    ax[0].plot(prof0.index, prof0.values, "o-", color=MUTED, label=f"제거 전 (최대/최소 {ratio0:.2f}배)")
    ax[0].plot(prof1.index, prof1.values, "o-", color=ACCENT2, label=f"제거 후 ({ratio1:.2f}배)")
    ax[0].set_xlabel("시각 (KST)"); ax[0].set_ylabel("평균 |로그수익률|")
    ax[0].set_title("(E-1) 주기 제거 효과 — 시간대 차이가 사라지는가", fontsize=12)
    ax[0].legend(fontsize=9); ax[0].grid(alpha=.25)

    ax[1].barh(ts["종목"], ts["중앙값"] * 100, color=ACCENT, alpha=.85)
    ax[1].set_xlabel("1시간 실현변동성 중앙값 (%)"); ax[1].set_xscale("log")
    ax[1].xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax[1].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax[1].set_title("(E-2) 종목별 타깃 수준 — 로그축으로 봐야 할 만큼 격차가 크다", fontsize=12)
    ax[1].grid(alpha=.25, axis="x")

    ax[2].barh(ts.sort_values("RV0비율")["종목"], ts.sort_values("RV0비율")["RV0비율"] * 100,
               color=ACCENT2, alpha=.85)
    ax[2].set_xlabel("실현변동성이 정확히 0인 구간 비율 (%)")
    ax[2].set_title("(E-3) 가격 이산성 — 로그 타깃이 무너지는 원인", fontsize=12)
    ax[2].grid(alpha=.25, axis="x")
    fig.suptitle("그림 2. 이번 회차 입력의 성질 — 주기·타깃 수준·가격 이산성", fontsize=15)
    fig.tight_layout()
    f2 = IMG / f"{STEM}_fig2_target_eda.png"
    fig.savefig(f2, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    emit("### 주기 제거와 타깃의 성질")
    emit()
    emit(f"![그림 2](../../images/{TAG}/{f2.name})")
    emit()
    emit(f"- **(E-1)** {rep}의 시간대별 평균 |수익률| 최대/최소 비가 제거 전 **{ratio0:.2f}배**에서 "
         f"제거 후 **{ratio1:.2f}배**로 줄었다. 시계만 보고도 알 수 있는 부분을 덜어낸 것이며, "
         "이것을 남겨 두면 모델이 그 쉬운 성분만 학습해 성능이 부풀고 GARCH 지속성이 왜곡된다.")
    emit(f"- **(E-2)** 종목별 타깃 중앙값이 {ts['중앙값'].min():.5f}에서 {ts['중앙값'].max():.5f}까지 "
         f"**{ts['중앙값'].max() / max(ts['중앙값'].min(), 1e-12):.1f}배** 차이 난다. 종목을 뭉쳐 "
         "하나의 절대 오차로 비교하면 변동성이 큰 종목이 표를 지배하므로, 척도에 자유로운 "
         "QLIKE와 MASE를 주 지표로 쓴다.")
    emit(f"- **(E-3)** 실현변동성이 정확히 0인 구간이 최대 **{ts['RV0비율'].max() * 100:.1f}%**"
         f"({ts.sort_values('RV0비율')['종목'].iloc[-1]})에 이른다. 이 구간은 관측 실패가 아니라 "
         "호가 단위 때문에 실제로 가격이 한 칸도 안 움직인 것이며, 학습에서는 결측으로 빼고 "
         "검증에서는 그대로 둔다.")
    emit()
    ts.to_csv(RES / f"{STEM}_stage_eda_targets.csv", index=False)
    dd.to_csv(RES / f"{STEM}_stage_eda_diagnostics.csv", index=False)
    return dict(ratio0=ratio0, ratio1=ratio1, targets=ts, diag=dd)


# %% [markdown]
# ## 구간별 성능 분해 — 종목별 5분위 (본편 계산 방식 재확인 후 확정)
#
# `process.md`에는 본편(포스터 그림 2)이 **고정 절대구간 풀링**을 썼고 예비검증(종목별
# 5분위)과 계산 방식이 달라 나란히 비교할 수 없다고 적혀 있었다. 이번에 두 방식을 모두
# 계산해 본편 산출물(`poster/regime.csv`)과 대조한 결과 **그 기술이 틀렸다**. 본편도
# 종목별 5분위였다 — naive 모델의 KRW-BTC 포착률이 본편 1.877 / 1.335 / 1.155 / 0.952 /
# 0.710 인데, 종목별 5분위로 재계산하면 1.88 / 1.34 / 1.15 / 0.95 / 0.71 로 일치하고,
# 풀링 절대경계로 계산하면 1.57 / 1.12 / 0.88 / 0.74 / 0.54 로 전혀 맞지 않는다.
#
# 방법 자체로도 종목별 5분위가 맞다. 풀링 절대경계를 쓰면 가장 낮은 구간에 **실현변동성이
# 정확히 0인 검열 관측**이 몰려 평균이 0에 붙고, 포착률(예측평균÷실제평균)이 발산한다
# (실측: DOGE 주기제거 후 Q1에서 1,892배). 종목 간 변동성 수준이 최대 수십 배 차이 나는
# 표본에서 절대 경계를 공유하면 저변동 종목은 Q5가 거의 비고 고변동 종목은 Q1이 거의 비어
# 종목 간 평균 자체가 성립하지 않는다.
#
# 그래서 **종목별 5분위를 주 분해로 쓰고**, 풀링 절대경계는 "같은 절대 변동성 수준에서
# 어느 모델이 나은가"라는 다른 질문에 답하는 보조 표로 QLIKE만 싣는다(비율 지표는 위
# 검열 문제 때문에 정의되지 않는다).

# %%
def regime_decomposition(preds_by_key: dict, acts_by_key: dict, pooled_edges: bool = False
                         ) -> tuple[pd.DataFrame, dict]:
    rows, edges_by_cond = [], {}
    for deper in (False, True):
        keys = sorted(k for k in acts_by_key if k[1] == deper)
        if not keys:
            continue
        if pooled_edges:
            shared = np.quantile(np.concatenate([acts_by_key[k] for k in keys]), [0.2, 0.4, 0.6, 0.8])
        per_ticker_edges = []
        for k in keys:
            act = acts_by_key[k]
            edges = shared if pooled_edges else np.quantile(act, [0.2, 0.4, 0.6, 0.8])
            per_ticker_edges.append(edges)
            bucket = np.digitize(act, edges)
            for nm, p in preds_by_key.get(k, {}).items():
                for q in range(5):
                    sel = bucket == q
                    if sel.sum() <= 5:
                        continue
                    a, pp = act[sel], p[sel]
                    am = float(a.mean())
                    rows.append({"종목": k[0], "주기제거": deper, "모델": nm, "구간": f"Q{q + 1}",
                                 "n": int(sel.sum()),
                                 "포착률": float(pp.mean() / am) if am > 1e-9 else np.nan,
                                 "상대오차": float(np.mean(np.abs(pp - a)) / am) if am > 1e-9 else np.nan,
                                 "QLIKE": M23.qlike(a ** 2, pp ** 2)})
        # 라벨용 대표 경계 — 종목별 경계의 중앙값(종목마다 경계가 다르므로 대표값만 보인다)
        edges_by_cond[deper] = shared if pooled_edges else np.median(np.array(per_ticker_edges), axis=0)
    return pd.DataFrame(rows), edges_by_cond


# %% [markdown]
# ## 구간별 데이터 특성 실측 — 설명의 근거를 추측이 아니라 수치로 만든다
#
# "Q5에서 레짐 전환 모델이 좋다"까지는 성능표가 말해 주지만, **왜** 그런지는 성능표에 없다.
# 그래서 구간마다 데이터 자체가 어떻게 다른지를 따로 재 둔다. 아래 항목들은 모두 검증
# 구간에서 직접 계산한 값이며, 이것이 §6 설명의 근거가 된다.
#
#   변동계수      — 구간 내부가 얼마나 퍼져 있는가(작으면 구간 중심값만 맞춰도 오차가 작다)
#   구간내첨도    — 구간 내부의 꼬리 두께(음수면 평탄, 크면 같은 구간 안에서도 극단값이 섞임)
#   RV0비율       — 실현변동성이 정확히 0인 검열 관측 비율
#   과거RV상관    — 직전 h구간 실현변동성이 현재를 얼마나 설명하는가(구간 내 예측가능성)
#   naive상대오차 — 과거를 그대로 베끼는 전략의 오차(구간의 내재적 난이도)
#   국면 전이     — 이 구간 관측이 직전에 어느 구간에 있었는가(행 합 = 1)

# %%
def regime_data_eda(tickers: list[str], conds: list[bool]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, trans = [], []
    for tk in tickers:
        for deper in conds:
            P = prepare(tk, deper)
            if P is None:
                continue
            act, nai = P["rv_act"], P["rv_nai"]
            edges = np.quantile(act, [.2, .4, .6, .8])
            bucket = np.digitize(act, edges)
            prev_bucket = np.digitize(nai, edges)     # 직전 상태를 같은 경계로 분류
            for q in range(5):
                sel = bucket == q
                if sel.sum() <= 5:
                    continue
                aa, nn = act[sel], nai[sel]
                am = float(aa.mean())
                rows.append(dict(
                    종목=tk, 주기제거=deper, 구간=f"Q{q + 1}", n=int(sel.sum()),
                    실제평균=am, 변동계수=float(aa.std() / am) if am > 0 else np.nan,
                    구간내첨도=float(st.kurtosis(aa)), 구간내왜도=float(st.skew(aa)),
                    RV0비율=float((aa <= 0).mean()),
                    과거RV상관=float(np.corrcoef(nn, aa)[0, 1]) if nn.std() > 0 else np.nan,
                    naive상대오차=float(np.mean(np.abs(nn - aa)) / am) if am > 0 else np.nan))
                for pq in range(5):
                    trans.append(dict(종목=tk, 주기제거=deper, 현재구간=f"Q{q + 1}",
                                      직전구간=f"Q{pq + 1}",
                                      비율=float((prev_bucket[sel] == pq).mean())))
    return pd.DataFrame(rows), pd.DataFrame(trans)


def plot_regime_eda(eda: pd.DataFrame, trans: pd.DataFrame, deper: bool, path: Path) -> None:
    """(R-1)~(R-4) — §6 설명이 기대는 네 가지 실측을 한 장에 모은다."""
    labs = [f"Q{i}" for i in range(1, 6)]
    s = eda[eda["주기제거"] == deper].groupby("구간").mean(numeric_only=True).reindex(labs)
    tt = (trans[trans["주기제거"] == deper]
          .pivot_table(index="현재구간", columns="직전구간", values="비율", aggfunc="mean")
          .reindex(index=labs, columns=labs))

    fig, ax = plt.subplots(1, 4, figsize=(22, 5.2))
    a = ax[0]
    a.bar(labs, s["변동계수"], color=ACCENT, alpha=.85)
    a.set_title("(R-1) 구간 내부가 얼마나 퍼져 있는가\n변동계수 = 표준편차 ÷ 평균", fontsize=12)
    a.set_ylabel("변동계수"); a.grid(alpha=.25, axis="y")
    for i, v in enumerate(s["변동계수"]):
        a.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=10)

    a = ax[1]
    a.bar(labs, s["구간내첨도"], color=ACCENT2, alpha=.85)
    a.axhline(0, color="k", lw=.8)
    a.set_title("(R-2) 구간 내부의 꼬리 두께\n초과첨도(정규분포=0, 균등분포≈-1.2)", fontsize=12)
    a.set_ylabel("구간 내 초과첨도"); a.set_yscale("symlog"); a.grid(alpha=.25, axis="y")
    for i, v in enumerate(s["구간내첨도"]):
        a.text(i, v, f"{v:.1f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=10)

    a = ax[2]
    a.bar(np.arange(5) - .2, s["과거RV상관"], width=.4, color=ACCENT, alpha=.85, label="과거RV상관")
    a.bar(np.arange(5) + .2, s["RV0비율"], width=.4, color="#7B68EE", alpha=.85, label="RV=0 비율")
    a.set_xticks(range(5)); a.set_xticklabels(labs)
    a.axhline(0, color="k", lw=.8)
    a.set_title("(R-3) 구간 내 예측가능성과 검열\n상관이 0이면 과거로 알 수 없다", fontsize=12)
    a.legend(fontsize=9); a.grid(alpha=.25, axis="y")

    a = ax[3]
    im = a.imshow(tt.values, cmap="YlOrRd", vmin=0, vmax=float(np.nanmax(tt.values)))
    a.set_xticks(range(5)); a.set_xticklabels(labs)
    a.set_yticks(range(5)); a.set_yticklabels(labs)
    a.set_xlabel("직전 구간"); a.set_ylabel("현재 구간")
    for i in range(5):
        for j in range(5):
            v = tt.values[i, j]
            a.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                   color="white" if v > 0.3 else "black")
    a.set_title("(R-4) 국면 전이 — 대각선이 지속성\n무작위라면 전부 0.20", fontsize=12)
    fig.colorbar(im, ax=a, fraction=.046)

    fig.suptitle(f"그림 4. 구간별 데이터 특성 실측 — 주기제거 {'후' if deper else '전'}, 20종목 평균",
                 fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_regime(rd: pd.DataFrame, edges: np.ndarray, deper: bool, path: Path,
                order: list[str]) -> None:
    labs = [f"Q{i}" for i in range(1, 6)]
    e = edges * 100
    xt = [f"하위 20%\n(종목별 중앙 ±{e[0]:.2f}% 이내)", f"20~40%\n(±{e[0]:.2f}~{e[1]:.2f}%)",
          f"40~60%\n(±{e[1]:.2f}~{e[2]:.2f}%)", f"60~80%\n(±{e[2]:.2f}~{e[3]:.2f}%)",
          f"상위 20%\n(±{e[3]:.2f}% 이상, 급등락)"]
    sub = rd[rd["주기제거"] == deper]
    pv = sub.pivot_table(index="모델", columns="구간", values="포착률", aggfunc="mean").reindex(columns=labs)
    pe = sub.pivot_table(index="모델", columns="구간", values="상대오차", aggfunc="mean").reindex(columns=labs)
    order = [m for m in order if m in pv.index]
    cmap = plt.get_cmap("tab20")
    col = {m: cmap(i % 20) for i, m in enumerate(order)}

    fig, ax = plt.subplots(1, 2, figsize=(19, 7.2))
    a = ax[0]
    a.axhline(1.0, color="#333", lw=2, ls="--", zorder=1)
    a.text(4.05, 1.03, "완벽한 예측 = 1.0", fontsize=10, color="#333", ha="right")
    for m in order:
        a.plot(range(5), pv.loc[m], marker="o", ms=6, lw=2.0, color=col[m], label=m, alpha=.85)
    a.set_xticks(range(5)); a.set_xticklabels(xt, fontsize=10)
    a.set_ylabel("포착률 = 예측 평균 ÷ 실제 평균", fontsize=12)
    a.set_xlabel("1시간 뒤 실제 변동성 크기 구간 (종목별 5분위)", fontsize=12)
    a.set_title("모든 모델이 평균으로 수축하는가\n작을 땐 부풀리고 클 땐 덜 잡는 경향", fontsize=13)
    a.legend(fontsize=8, ncol=3, frameon=False)
    a.grid(alpha=.25)

    a = ax[1]
    w = 0.9 / max(len(order), 1)
    for i, m in enumerate(order):
        a.bar(np.arange(5) + (i - (len(order) - 1) / 2) * w, pe.loc[m], width=w,
              color=col[m], label=m, alpha=.9)
    for j, c in enumerate(labs):
        if c in pe.columns and pe[c].notna().any():
            b = pe[c].idxmin()
            a.text(j, pe[c].max() * 1.02, f"최우수\n{b}", ha="center", fontsize=9, weight="bold")
    a.set_xticks(range(5)); a.set_xticklabels(xt, fontsize=10)
    a.set_ylabel("상대오차 (낮을수록 좋음)", fontsize=12)
    a.set_xlabel("1시간 뒤 실제 변동성 크기 구간", fontsize=12)
    a.set_title("구간마다 최우수 모델이 다른가\n단일 승자 존재 여부", fontsize=13)
    a.legend(fontsize=8, ncol=3, frameon=False)
    a.grid(alpha=.25, axis="y")
    fig.suptitle(f"그림 3. 변동성 구간별 성능 분해 — 주기제거 {'후' if deper else '전'}, "
                 f"{len(order)}개 모델 · {rd['종목'].nunique()}종목", fontsize=15, y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# %% [markdown]
# ## 구간별 차이의 원인 설명
#
# 서술 원칙(사용자 지시 2026-09-28): **"다른 논문도 이랬습니다"로 시작하지 않는다.**
# 순서를 반드시 `우리 EDA 실측 → 그 특성이 이 알고리즘의 어느 성질과 맞아떨어지는가 →
# 그래서 이 결과가 나왔다 → 이 해석을 뒷받침하는 문헌`으로 둔다. 문헌은 우리 관찰을
# 설명하는 보조 근거이고, 주장의 1차 근거는 항상 우리 데이터다.

# %%
def emit_regime_explanation(rd: pd.DataFrame, reg: pd.DataFrame, tickers: list[str],
                            conds: list[bool], md: pd.DataFrame) -> None:
    from report_header import num

    labs = [f"Q{i}" for i in range(1, 6)]
    print("[설명] 구간별 데이터 특성 실측 중...", flush=True)
    eda, trans = regime_data_eda(tickers, conds)
    if not len(eda):
        emit("## 6. 왜 구간마다 최우수 모델이 다른가")
        emit()
        emit("> 구간별 데이터 특성 실측에 실패해 이 절을 비워 둔다.")
        emit()
        return
    eda.to_csv(RES / f"{STEM}_regime_data_eda.csv", index=False)
    trans.to_csv(RES / f"{STEM}_regime_transition.csv", index=False)

    REF = "주기제거 후"                     # 본문 수치는 이 조건을 기준으로 서술한다
    ref_deper = True if True in conds else conds[0]
    s = (eda[eda["주기제거"] == ref_deper].groupby("구간")
         .mean(numeric_only=True).reindex(labs))
    tt = (trans[trans["주기제거"] == ref_deper]
          .pivot_table(index="현재구간", columns="직전구간", values="비율", aggfunc="mean")
          .reindex(index=labs, columns=labs))
    rsub = reg[reg["주기제거"] == ref_deper]
    cap = rsub.pivot_table(index="모델", columns="구간", values="포착률", aggfunc="mean").reindex(columns=labs)
    rel = rsub.pivot_table(index="모델", columns="구간", values="상대오차", aggfunc="mean").reindex(columns=labs)
    ql = rsub.pivot_table(index="모델", columns="구간", values="QLIKE", aggfunc="mean").reindex(columns=labs)

    def g(col, q):
        return float(s.loc[q, col])

    def mv(tbl, model, q):
        try:
            return float(tbl.loc[model, q])
        except Exception:
            return float("nan")

    emit("## 6. 왜 구간마다 최우수 모델이 다른가 — 우리 데이터 실측에 근거한 설명")
    emit()
    emit("§5는 \"구간마다 승자가 다르다\"는 **사실**을 보였다. 이 절은 그 **이유**를 댄다. "
         "설명의 출발점은 남의 논문이 아니라 우리 데이터다 — 구간마다 데이터 자체가 어떻게 "
         "다른지를 먼저 실측하고, 그 특성이 각 알고리즘의 어느 성질과 맞아떨어지는지를 "
         "연결한 다음, 마지막에 그 해석을 뒷받침하는 문헌을 붙인다.")
    emit()
    emit(f"아래 수치는 모두 **{REF}** 조건 · 20종목 검증 구간에서 직접 계산한 값이다"
         "(조건별 전량은 `regime_data_eda.csv`·`regime_transition.csv`).")
    emit()

    # ── 6-A. 구간별 데이터 특성 실측표 ──
    emit("### 6-A. 먼저 구간마다 데이터가 어떻게 다른가 (실측)")
    emit()
    emit("| 구간 | 실제 평균 RV | 변동계수 | 구간내 초과첨도 | RV=0 비율 | 과거RV 상관 | naive 상대오차 |")
    emit("| :--- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for q in labs:
        emit(f"| {q} | {g('실제평균', q) * 100:.3f}% | {g('변동계수', q):.3f} | "
             f"{g('구간내첨도', q):.2f} | {g('RV0비율', q) * 100:.2f}% | "
             f"{g('과거RV상관', q):.3f} | {g('naive상대오차', q):.3f} |")
    emit()
    emit("**국면 전이 — 이 구간 관측은 직전에 어디 있었나** (행 합 = 1, 무작위라면 전부 0.20)")
    emit()
    emit("| 현재 \\ 직전 | " + " | ".join(labs) + " |")
    emit("| :--- | " + " | ".join(["---:"] * 5) + " |")
    for q in labs:
        emit(f"| **{q}** | " + " | ".join(f"{tt.loc[q, p]:.3f}" for p in labs) + " |")
    emit()
    fe = IMG / f"{STEM}_fig4_regime_eda_{'deper' if ref_deper else 'raw'}.png"
    try:
        plot_regime_eda(eda, trans, ref_deper, fe)
        emit(f"![그림 4](../../images/{TAG}/{fe.name})")
        emit()
    except Exception as e:
        emit(f"> (그림 4 생성 실패: {e})")
        emit()
    emit("이 표에서 **세 가지 사실**이 읽힌다. 아래 설명은 전부 이 세 가지로 환원된다.")
    emit()
    emit(f"1. **Q1은 정보가 없는 구간이다.** 실현변동성이 정확히 0인 검열 관측이 "
         f"**{g('RV0비율', 'Q1') * 100:.2f}%** 몰려 있는데 다른 구간은 0%다. 그리고 과거 RV와 "
         f"현재 RV의 상관이 **{g('과거RV상관', 'Q1'):.3f}** — 사실상 0이다. naive 상대오차가 "
         f"**{g('naive상대오차', 'Q1'):.3f}**로 5개 구간 중 유일하게 1을 넘는다(과거를 베끼는 "
         "전략이 이 구간에서만 실패한다).")
    mid_cv = [g("변동계수", q) for q in ("Q2", "Q3", "Q4")]
    mid_ku = [g("구간내첨도", q) for q in ("Q2", "Q3", "Q4")]
    emit(f"2. **Q2~Q4는 좁고 평탄한 구간이다.** 변동계수가 "
         f"{min(mid_cv):.3f}~{max(mid_cv):.3f}로 Q1({g('변동계수', 'Q1'):.3f})·"
         f"Q5({g('변동계수', 'Q5'):.3f})보다 4~6배 작고, 구간내 초과첨도가 "
         f"{min(mid_ku):.2f}~{max(mid_ku):.2f}로 **음수**다"
         "(정규분포 0, 균등분포 약 −1.2) — 구간 내부가 거의 고르게 꽉 차 있다는 뜻이다.")
    emit(f"3. **Q5는 다른 세계다.** 구간내 초과첨도가 **{g('구간내첨도', 'Q5'):.1f}**로 "
         f"Q2~Q4의 음수와 질적으로 다르고, 변동계수도 {g('변동계수', 'Q5'):.3f}로 "
         f"Q3({g('변동계수', 'Q3'):.3f})의 {g('변동계수', 'Q5') / g('변동계수', 'Q3'):.1f}배다. "
         f"동시에 과거RV 상관이 **{g('과거RV상관', 'Q5'):.3f}**로 5개 구간 중 가장 높다 — "
         "**같은 구간 안에서도 극단값이 섞여 있지만, 그 구간 자체는 과거로부터 예측 가능하다.**")
    emit(f"   전이 행렬이 이를 확정한다: 대각 성분이 Q1 {tt.loc['Q1', 'Q1']:.3f} → "
         f"Q3 {tt.loc['Q3', 'Q3']:.3f} → Q5 {tt.loc['Q5', 'Q5']:.3f}로 **U자형**이다. "
         "저변동·고변동 양 끝만 끈적하고 중간은 지나가는 구간이다 — 즉 이 데이터는 "
         "**두 개의 머무는 상태와 그 사이 전이**라는 구조를 실제로 갖고 있다.")
    emit()

    # ── 6-B. 구간별 승자 설명 ──
    emit("### 6-B. 그래서 구간마다 누가 이기는가")
    emit()
    emit("아래 각 항목은 `실측 → 알고리즘 성질 → 결과` 순서로 읽는다. 괄호 안 수치는 전부 "
         f"{REF} 조건의 실측값이다.")
    emit()

    emit("#### Q1·Q2 (가장 잔잔) — GRU·LSTM")
    emit()
    emit(f"- **실측**: 위 사실 1. 과거RV 상관 {g('과거RV상관', 'Q1'):.3f}, 검열 "
         f"{g('RV0비율', 'Q1') * 100:.2f}%. 그리고 이 구간에서 **모든 모델이 과대예측한다** — "
         f"포착률(예측평균÷실제평균)이 GARCH-t {mv(cap, 'GARCH-t', 'Q1'):.2f}배, "
         f"MS-GARCH {mv(cap, 'MS-GARCH', 'Q1'):.2f}배, TAR-GARCH {mv(cap, 'TAR-GARCH', 'Q1'):.2f}배인데 "
         f"GRU는 {mv(cap, 'GRU', 'Q1'):.2f}배로 **과대예측이 가장 작다**.")
    emit("- **알고리즘 성질**: GARCH 계열의 조건부분산 재귀식은 예측이 길어지면 "
         "무조건부 분산(전체 표본의 평균 변동성 수준)으로 회귀한다. 실제가 가장 작은 이 "
         "구간에서는 그 회귀가 곧 과대예측이 된다. 주 지표 QLIKE = log(h) + σ²/h는 "
         "**과대예측을 log(h) 항으로 직접 벌점**하므로 이 구간에서 GARCH 계열이 크게 불리하다. "
         "반면 GRU·LSTM은 로그 실현변동성의 제곱오차를 직접 최소화하도록 학습되어 낮은 "
         "수준대에서 눈금이 맞는다.")
    emit(f"- **결과**: Q1 상대오차 GRU {mv(rel, 'GRU', 'Q1'):.3f} < Linear "
         f"{mv(rel, 'Linear', 'Q1'):.3f} < LightGBM {mv(rel, 'LightGBM', 'Q1'):.3f} ≪ "
         f"GARCH-t {mv(rel, 'GARCH-t', 'Q1'):.3f}. 다만 "
         "**GRU가 이 구간을 잘 맞힌다는 뜻은 아니다** — 상대오차가 1을 넘어 어떤 모델도 "
         "실질적 예측력이 없고, 순위는 \"누가 덜 과대예측하는가\"로 결정된다. 이 구간의 "
         "1위는 성능이 아니라 **눈금 보정(calibration)의 승리**다.")
    emit()

    emit("#### Q3 (보통) — KernelRidge-RBF")
    emit()
    emit(f"- **실측**: 위 사실 2. Q3의 변동계수가 {g('변동계수', 'Q3'):.3f}로 5개 구간 중 "
         f"가장 작고, 구간내 초과첨도 {g('구간내첨도', 'Q3'):.1f}로 균등분포에 가깝다. "
         "즉 구간 내부 값들이 좁은 띠 안에 고르게 모여 있다.")
    emit(f"- **알고리즘 성질**: RBF 커널 릿지는 본질적으로 \"입력이 가까운 학습 사례들의 "
         f"가중 평균\"이다. 값이 좁고 조밀하게 모여 있으면 이 국소 평균이 최적예측에 "
         f"가까워진다. 실제로 내부검증이 고른 대역폭이 gamma 중앙값 "
         f"{md['kr_gamma'].median() if 'kr_gamma' in md.columns else float('nan'):.4f}로 "
         "넓은 쪽이다 — 데이터가 강한 스무딩을 요구했다는 뜻이다.")
    emit(f"- **결과**: Q3 상대오차 KernelRidge-RBF {mv(rel, 'KernelRidge-RBF', 'Q3'):.3f} < "
         f"LightGBM {mv(rel, 'LightGBM', 'Q3'):.3f} < HAR-RV {mv(rel, 'HAR-RV', 'Q3'):.3f}. "
         "트리 계열이 근소한 2위인 것도 같은 이유다 — 트리도 구간을 잘라 그 안의 평균을 "
         "내는 국소 평균 기계다.")
    emit()

    emit("#### Q4 (크지만 극단은 아님) — GARCH-t")
    emit()
    emit(f"- **실측**: Q4는 구간내 첨도가 여전히 음수({g('구간내첨도', 'Q4'):.1f})로 평탄하면서, "
         f"과거RV 상관이 {g('과거RV상관', 'Q4'):.3f}로 중간 구간들보다 올라오기 시작한다. "
         f"전이 행렬에서도 Q4의 자기지속성이 {tt.loc['Q4', 'Q4']:.3f}로 Q3({tt.loc['Q3', 'Q3']:.3f})보다 "
         "높아진다 — 변동성 군집이 본격적으로 작동하는 영역이다.")
    emit("- **알고리즘 성질**: GARCH의 재귀식 σ²ₜ = ω + α·rₜ₋₁² + β·σ²ₜ₋₁은 \"직전 충격과 "
         "직전 변동성 수준을 지수가중으로 누적\"하는 구조다. 이것이 정확히 변동성 군집의 "
         "형태이고, 우리 EDA가 ARCH-LM p=0.000으로 그 존재를 이미 확인했다. 파라미터가 "
         "3~4개뿐이라 이 구간의 표본으로도 안정적으로 추정된다.")
    emit(f"- **결과**: Q4 QLIKE에서 GARCH-t {num(mv(ql, 'GARCH-t', 'Q4'), 3)}가 1위다. "
         f"다만 상대오차 기준으로는 XGBoost {mv(rel, 'XGBoost', 'Q4'):.3f}가 GARCH-t "
         f"{mv(rel, 'GARCH-t', 'Q4'):.3f}보다 낫다 — **두 지표가 다른 승자를 지목하는 구간**이다. "
         "QLIKE는 분산 축에서 비대칭 벌점을 주고 상대오차는 표준편차 축에서 대칭 벌점을 "
         "주므로, 이 구간에서는 어느 축으로 재느냐가 순위를 바꾼다는 사실 자체를 결과로 "
         "보고한다(주 지표는 QLIKE로 고정하되 이 불일치를 숨기지 않는다).")
    emit()

    emit("#### Q5 (급등락) — TAR-GARCH·MS-GARCH")
    emit()
    emit("이 구간의 설명이 이번 회차에서 가장 근거가 두텁다. 우리 실측 네 가지가 모두 "
         "같은 방향을 가리킨다.")
    emit()
    emit(f"- **실측 ①** 구간내 초과첨도 **{g('구간내첨도', 'Q5'):.1f}** — Q2~Q4의 음수와 "
         "질적으로 다르다. 같은 Q5 안에서도 평범한 급변과 극단적 급변이 섞여 있다.")
    emit(f"- **실측 ②** 과거RV 상관 **{g('과거RV상관', 'Q5'):.3f}**로 5개 구간 중 최고 — "
         "이 구간만 과거로부터 예측 가능한 지속 구조를 갖는다.")
    emit(f"- **실측 ③** 전이 행렬의 U자형(Q1 {tt.loc['Q1', 'Q1']:.3f} · "
         f"Q3 {tt.loc['Q3', 'Q3']:.3f} · Q5 {tt.loc['Q5', 'Q5']:.3f}, 무작위 0.20) — "
         "데이터가 **두 개의 머무는 상태**를 실제로 갖는다.")
    if {"ms_p11", "ms_p22"}.issubset(md.columns):
        h = md[md["주기제거"] == ref_deper]
        p11, p22 = float(h["ms_p11"].median()), float(h["ms_p22"].median())
        emit(f"- **실측 ④** MS-GARCH의 최대우도 추정이 실측 ③을 **독립적으로** 확인한다. "
             f"추정된 잔류확률이 p₁₁={p11:.4f}, p₂₂={p22:.4f}로, 기대 체류기간이 "
             f"저변동 국면 {1 / (1 - p11):.0f}봉({1 / (1 - p11) / BPD:.1f}일) · "
             f"고변동 국면 {1 / (1 - p22):.0f}봉({1 / (1 - p22) / BPD:.1f}일)이다. "
             "전이 행렬을 보지 않고 우도만으로 추정했는데도 \"두 국면 모두 매우 끈적하다\"는 "
             "같은 결론에 도달했다.")
    emit("- **실측 ⑤** 별도 회차(22번 거시 EDA)에서 고변동 국면 초과첨도 253 대 저변동 국면 "
         "14.4 — 두 국면의 통계적 성질이 17배 차이 난다.")
    emit()
    emit("- **알고리즘 성질**: 단일국면 GARCH-t는 하나의 (ω, α, β, ν) 세트로 이 두 이질적 "
         "상태를 **모두** 설명해야 하므로, 추정치가 두 국면의 타협점에 놓인다. 고변동 "
         "국면에서는 그 타협이 과소예측으로 나타난다. 레짐 전환 모델은 국면별 파라미터를 "
         "따로 두므로 고변동 국면에 그 국면 전용 동학을 쓸 수 있다.")
    emit(f"- **결과**: Q5 포착률이 GARCH-t {mv(cap, 'GARCH-t', 'Q5'):.3f}(실제 변동의 "
         f"{mv(cap, 'GARCH-t', 'Q5') * 100:.0f}%만 잡음)인데 "
         f"TAR-GARCH {mv(cap, 'TAR-GARCH', 'Q5'):.3f} · MS-GARCH {mv(cap, 'MS-GARCH', 'Q5'):.3f}로 "
         "1.0에 훨씬 가깝다. 상대오차도 TAR-GARCH "
         f"{mv(rel, 'TAR-GARCH', 'Q5'):.3f} · MS-GARCH {mv(rel, 'MS-GARCH', 'Q5'):.3f} < "
         f"GARCH-t {mv(rel, 'GARCH-t', 'Q5'):.3f}다. "
         f"반대급부로 Q1에서는 같은 이유로 레짐 모델이 더 크게 과대예측한다"
         f"(포착률 {mv(cap, 'MS-GARCH', 'Q1'):.2f}·{mv(cap, 'TAR-GARCH', 'Q1'):.2f}배 대 "
         f"GARCH-t {mv(cap, 'GARCH-t', 'Q1'):.2f}배) — **전체평균 1위가 GARCH-t인 것과 Q5 "
         "1위가 레짐 모델인 것은 모순이 아니라 같은 구조의 양면**이다.")
    emit()

    # ── 6-C. 왜 어떤 모델은 전 구간에서 나쁜가 ──
    emit("### 6-C. 전 구간에서 하위권인 모델들 — 나쁜 것도 이유가 있다")
    emit()
    emit("- **HAR-RV·Linear·Ridge (13~15위)**: 우리 EDA의 RESET·BDS 검정이 선형성을 "
         "p=0.000으로 기각했다(§분석조건 5항). 이 셋은 전부 선형 결합 구조이므로 "
         "기각된 가정 위에 서 있다. 검정이 예고한 결과가 순위로 확인된 것이며, 실제로 "
         f"Q5 포착률이 HAR-RV {mv(cap, 'HAR-RV', 'Q5'):.3f}로 주요 모델 중 가장 낮다 — "
         "선형 구조가 극단 구간의 크기를 가장 못 따라간다.")
    emit(f"- **ITransformerLike (16위, naive보다 열세)**: 입력이 원 시퀀스 (r, |r|) 2채널뿐인데 "
         "셀프어텐션은 \"어느 시점을 볼지\"를 데이터로부터 배워야 한다. 우리 EDA가 확인한 "
         "예측 가능한 구조는 **|수익률|의 느린 자기상관**(하루 96봉 뒤에도 0.12)인데, 이건 "
         "다중 시간대 집계로 드러나는 구조다 — GARCH는 재귀식으로, 트리·커널은 사람이 만든 "
         "다중 lag 특성으로 이것을 **직접** 받는다. GRU·LSTM은 같은 2채널만 받지만 순환 "
         "구조의 상태 누적이 지수가중이동평균과 수학적으로 닮아 있어 이 구조에 맞는 "
         "귀납편향을 갖는다. 어텐션에는 그 편향이 없고, 학습 6.2만 표본은 그 자유도를 "
         "지지하지 못했다(학습손실은 0.985→0.812로 내려가는데 검증손실은 0.88~1.07에서 "
         "요동 — §제외 모델 절 참조).")
    emit("- **naive (17위)**: 기준선이다. 다만 위 실측에서 보인 대로 **Q1에서는 naive의 "
         f"포착률 {mv(cap, 'naive', 'Q1'):.3f}이 GARCH 계열보다 오히려 1에 가깝다** — "
         "기준선이 모든 구간에서 최하위인 것은 아니라는 점을 함께 적는다.")
    emit()

    # ── 6-D. 문헌 근거 ──
    emit("### 6-D. 위 해석을 뒷받침하는 문헌")
    emit()
    emit("아래 문헌은 **우리 관찰을 설명하는 보조 근거**다. 주장의 1차 근거는 위 6-A/6-B의 "
         "실측값이며, 문헌은 \"이 메커니즘이 우리 데이터에서만 보이는 우연이 아니다\"를 "
         "보이는 역할만 한다. arxiv 수록 논문은 초록을 대조했고, 고전 문헌은 arxiv에 없어 "
         "**초록 대조 불가**로 표시한다(`AGENTS.md` 5절).")
    emit()
    for line in LITERATURE_NOTES:
        emit(line)
    emit()


# 문헌 목록 — arxiv MCP(`test/scripts/mcp_client.py` 경유)로 **초록까지 대조**한 것만
# 인용하고, arxiv 미수록 고전은 "초록 대조 불가"로 구분해 적는다(AGENTS.md 5절).
# 2026-09-30 조사: 16개 검색 쿼리 · 34편 초록 대조.
LITERATURE_NOTES: list[str] = [
    "| 우리 관찰(§6-A·6-B) | 뒷받침 문헌 (arXiv ID) | 지지 강도 · 주의점 |",
    "| :--- | :--- | :--- |",
    "| **구간마다 최우수 모델이 다르다** — Q1·Q2=GRU, Q3=KernelRidge, Q4=GARCH-t, Q5=레짐전환 |"
    " Zhong(2026) `2604.10402` | **직접지지.** 초록이 \"the strongest forecaster is "
    "**regime-dependent** rather than stable across all states\"로 우리 관찰과 문장 단위로 "
    "일치한다. 단 ETF 6종·일간·2구간(calm/stressed)이라 표본과 구간 해상도가 우리보다 거칠다. |",
    "| **전체평균 순위만으로 모델을 고르면 안 된다**(§5에서 구간별 병기한 이유) |"
    " Chagas 외(2026) `2608.01599` | **직접지지.** \"aggregate accuracy metrics … can **hide "
    "conditional failures**\", \"models with competitive aggregate accuracy can still exhibit "
    "substantial **regime-specific bias and severe tail underprediction**\". **암호화폐를 포함**한 "
    "연구라 자산이 가깝다. 단 국면을 잠재 클러스터링으로 정의(우리는 관측 RV 5분위). |",
    "| **Q5에서 레짐 전환 모델이 단일국면 GARCH를 이긴다** |"
    " Chaudhary(2026) `2606.06190` · Blake 외(2025) `2510.03236` |"
    " **직접지지.** 전자는 \"single-regime models **inadequate**\"와 \"statistically distinct "
    "Calm, Turbulent, **Crisis** regimes\" + 예측 개선을 명시. 후자는 \"especially during periods "
    "of **heightened uncertainty and structural change**\". 단 전자는 EUR/USD 단일 통화쌍이고 "
    "**국면별 분해 수치는 초록에 없어** 전체평균 개선만 주장한다. 후자는 MS-GARCH가 아니라 "
    "soft-clustering+XGBoost 조합이라 \"레짐 인지 모델링\"의 근거로만 쓴다. |",
    "| (같은 항목의 **반증**) |"
    " Koch 외(2024) `2401.03393` · Balcerek 외(2026) `2609.25965` |"
    " **미지지 — 숨기지 않고 함께 싣는다.** 전자는 **비트코인**에서 \"in certain situations, "
    "persistent simple GARCH models **may even outperform** Markov-Switching GARCH\"로 "
    "**자산이 우리와 가장 가까운데 결론이 반대**다(단 일간·조건부분산 대상, 우리는 15분봉·실현변동성). "
    "후자는 MSGARCH의 1-step 정확도가 **동등**이고 가치는 해석가능성에 있다고 본다. |",
    "| **GARCH 계열은 잔잔한 구간에서 과대예측, 학습형 모델은 급변 구간에서 과소예측** "
    "(포착률 실측: Q1에서 GARCH계열 2.9~3.6배 · Q5에서 GRU 0.59) |"
    " Chung(2024) `2405.19849` | **직접지지 — 이번 조사에서 가장 유용한 해석 근거.** \"**Machine "
    "learning models tend to underpredict, while GARCH models tend to overpredict**\"가 우리 "
    "포착률 방향과 그대로 일치한다. QLIKE가 과소예측을 비대칭적으로 크게 벌하는 성질과 겹쳐 "
    "구간별 순위가 갈리는 메커니즘을 설명한다. 단 에너지 원자재·일간이라 자산이 다르다. |",
    "| **두꺼운 꼬리 때문에 정규분포가 아닌 조건부 분포가 필요하다**(우리 EDA: 초과첨도 107, "
    "Hill 꼬리지수 2.6~2.8) |"
    " Cerqueti 외(2020) `2004.11674` · Gyamerah(2019) `1909.04903` |"
    " **직접지지.** 전자는 암호화폐에서 \"**relaxing the normality assumption**\"이 예측을 "
    "개선함을 실증. 후자는 \"time series of most cryptocurrency are **leptokurtic**\". "
    "**주의**: 전자가 최우수로 꼽은 것은 **왜도까지 허용한** Skewed GED이고 우리는 **대칭 "
    "Student-t**만 썼다 — 비대칭 분포는 아직 시도하지 않은 후속 후보다. |",
    "| **암호화폐에서 GARCH 계열이 강하다**(우리: GARCH-t 1위) |"
    " Chi & Hao(2020) `2010.07402` | **부분지지.** \"GARCH and EGARCH models perform **much "
    "better** than other models\". 단 비교군에 기계학습이 아예 없다(HIST/EMA/ARCH 계열만). |",
    "| **선형 모델이 하위권**(HAR-RV 15위, Linear·Ridge 13·14위) — RESET·BDS가 p=0.000으로 "
    "선형성 기각 |"
    " Zeng 외(2022) `2205.13504` | **주의 필요 — 부분적으로만 인용 가능.** 이 논문은 \"단순 선형이 "
    "Transformer를 이긴다\"는 쪽이라 **우리 결과와 방향이 반대**다. 따라서 아래 Transformer "
    "항목에서 \"어텐션이 실패할 수 있다\"까지만 인용하고 \"따라서 선형이 낫다\"로 확장하지 않는다. |",
    "| **Transformer 계열이 naive보다 못하다**(ITransformerLike 16위, PatchTSTLike 제외) |"
    " Zeng 외(2022) `2205.13504` · Andreoletti(2026) `2604.00064` · Ekambaram 외(2023) "
    "`2306.09364` · Bilokon & Qiu(2023) `2309.11400` |"
    " **직접지지(복수).** `2205.13504`는 \"permutation-invariant self-attention … "
    "**temporal information loss**\". `2604.00064`는 우리가 실측한 현상(학습손실은 내려가는데 "
    "검증손실은 요동)의 **메커니즘 설명**을 준다 — \"increased model expressivity does not improve "
    "predictive accuracy but instead introduces **spurious trajectory fluctuations** … from the "
    "**reuse of noise** … increased prediction variance without any reduction in bias\". "
    "`2309.11400`은 금융 시계열에서 \"**LSTM is still a dominant architecture**\". "
    "**중요한 한계**: `2604.00064`의 메커니즘은 \"조건부 구조가 약한\" 시계열 + 제곱손실을 "
    "전제하는데, 우리 표적은 |수익률| 자기상관이 96봉 뒤에도 0.12인 **조건부 구조가 강한** "
    "실현변동성이다 — 전제가 달라 1:1로 이전되지 않는다. |",
    "| (같은 항목의 **반증**) | Nie 외(2022) `2211.14730` |"
    " **미지지 — 함께 싣는다.** PatchTST 원논문은 \"can improve the long-term forecasting "
    "accuracy **significantly**\"다. 그 성능은 **패칭 + 채널 독립** 두 설계에 귀속되므로, "
    "우리 구현 실패를 \"Transformer 일반의 실패\"로 일반화하지 않는다. 특히 "
    "ITransformerLike는 **채널 혼합** 계열이라 이 논문의 설계 전제와 반대다. |",
    "| **GRU·LSTM은 같은 2채널 입력으로도 학습된다** — 순환 구조의 귀납편향이 "
    "변동성 군집(느린 저주파 신호)과 맞는다 |"
    " Ishii 외(2023) `2305.09178` · Chen & Chen(2025) `2509.20789` |"
    " **부분지지 — 이 연결은 문헌의 결론이 아니라 본 보고서의 해석이다.** 전자는 \"LSTM and GRU "
    "have an inductive bias towards **lower-frequency patterns**\"(단 시퀀스 **분류** 과제, "
    "합성 데이터, 어텐션과 비교 안 함). 후자는 \"귀납편향이 과제 구조와 어긋나면 표본 효율이 "
    "나빠진다\"는 일반 원리(단 State Space Model 대상). **\"RNN의 재귀 편향이 어텐션보다 "
    "유리하다\"를 정면으로 검증한 문헌은 arxiv에서 찾지 못했다.** 반대 사례도 있다"
    "(Hollis 외 2018 `1812.07699`: LSTM+attention이 순수 LSTM보다 우수). |",
    "| **Q3에서 RBF 커널이 1위, 극단 구간에서는 열위**(Q3 상대오차 0.207 1위 / Q5 0.466 하위) — "
    "구간 내부가 좁고 조밀(변동계수 0.088)할수록 국소 평균이 유리 |"
    " Ge & Braun(2024) `2406.10308` · Yoshida(2019) `1903.03242` |"
    " **직접지지.** 전자는 \"local polynomial regression … performs **poorly in regions of "
    "sparse data** … especially at the **boundaries**\". 후자는 \"investigation of the tail "
    "quantile is difficult because of **data sparsity**\"이며 **중간 영역은 통상 추정량, 극단은 "
    "별도 외삽**으로 나눈다 — 우리 Q3 우위/Q5 열위 구조와 대응한다. "
    "**주의**: 둘 다 국소다항·분위 회귀이고 KernelRidge는 RKHS 정칙화 회귀라 엄밀히 다른 "
    "추정량이다 — \"커널 평활 계열의 공통 성질\" 수준으로만 쓴다. |",
    "",
    "#### 문헌과 어긋나는 지점 — 숨기지 않고 자기점검 항목으로 남긴다",
    "",
    "초록 대조 과정에서 **우리 결과와 충돌하는 문헌**이 나왔다. 유리한 것만 고르면 인용이 "
    "아니라 장식이 되므로, 충돌 지점과 그에 대한 우리 쪽 확인 결과를 함께 적는다.",
    "",
    "1. **HAR-RV가 15위인 것은 문헌과 충돌한다.** Wang & Lu(2024) `2409.08356`은 일간 실현변동성에서 "
    "**HAR이 QLIKE 최우수**라고 보고한다. 다만 같은 논문이 \"when the data is replaced with "
    "**hourly high-frequency** realized volatility, the **deep learning models outperform** the "
    "GARCH model\"이라고도 한다 — 즉 **빈도가 조건 변수**이고, 15분봉인 우리 설정은 후자에 가깝다. "
    "우리 HAR-RV 구현은 과거 1봉/1일(96봉)/1주(672봉) 로그 실현변동성이라는 Corsi(2009) 표준 "
    "정의를 그대로 따랐음을 코드에서 확인했다(`feats_har`).",
    "2. **Linear·Ridge가 13·14위인 것도 문헌과 충돌한다.** Dudek 외(2025) `2508.15922`는 암호화폐 "
    "실현분산에서 \"**log-transformed** realized volatility 위의 **선형** 기저모형이 정교한 "
    "대안보다 낫다\"고 본다. 우리도 타깃을 로그변환해 쓰고 있음을 확인했다"
    "(`y = log(fut + 1e-14)`) — 따라서 \"로그변환을 안 해서\"로는 설명되지 않는다. "
    "그 논문이 **분위(확률)예측** 프레임이고 **BTC 단일 종목**이라는 점이 차이로 남는다.",
    "3. **Transformer 실패를 아키텍처 탓으로만 돌리기 전에 룩백 탐색이 필요하다.** "
    "Huang 외(2026) `2606.27282`는 \"**optimal lookback is strongly series-specific** and often "
    "non-monotonic\"이라며 전처리·룩백 튜닝이 아키텍처 확장보다 중요하다고 본다. 이번 회차는 "
    "20종목 전부에 **동일 룩백(192봉)·동일 은닉폭(128)**을 적용했으므로, 이 한계를 명시한다. "
    "(학습률은 3단 격자로 탐색했으나 룩백은 고정했다.)",
    "4. **TAR-GARCH의 우위를 \"레버리지 효과\"로 설명하지 않는다.** Chi & Hao(2020) `2010.07402`는 "
    "비트코인에서 \"the EGARCH model's asymmetric term is positive and **insignificant** … "
    "Bitcoin prices **lack the asymmetric volatility response** to past returns\"라 하고, "
    "Kakinaka & Umeno(2021) `2102.02865`는 종목에 따라 비대칭의 **부호가 뒤집힌다**고 보고한다. "
    "우리 TAR-GARCH의 임계변수는 수익률 부호가 아니라 **직전 실현변동성 수준**임을 코드에서 "
    "확인했으므로(`switch_lagged = pas`를 1칸 민 값), 레버리지가 아니라 **변동성 수준 분할**로 "
    "해석하는 것이 맞고 위 두 문헌과도 충돌하지 않는다.",
    "5. **실현변동성 프록시 정의가 순위를 흔들 수 있다.** Wang 외(2021) `2110.01189`는 \"the "
    "empirical loss comparison between two volatility predictors **hinges on the deviation of "
    "the volatility proxy from the true volatility**\"를 **비트코인 데이터에서** 보인다. 우리는 "
    "1시간 구간 제곱수익률 합의 제곱근을 프록시로 쓰며, 이 선택이 순위에 주는 영향은 "
    "측정하지 않았다 — 후속 과제로 남긴다.",
    "6. **QLIKE 직접 최적화의 불안정성은 이번 회차에 해당하지 않는다.** Reisenhofer 외(2022) "
    "`2205.07719`는 QLIKE를 손실로 쓸 때 최적화가 불안정해진다고 보고하는데, 우리 딥러닝 "
    "모델은 **로그 타깃의 평균제곱오차**로 학습하고 QLIKE는 평가에만 썼음을 확인했다"
    "(`nn.MSELoss`). 따라서 Transformer의 검증손실 요동을 이 요인으로 설명할 수는 없다.",
    "",
    "#### arxiv 미수록 고전 — 초록 대조 불가",
    "",
    "아래는 이 분야의 표준 문헌이지만 arxiv에 없어 **초록을 대조하지 못했다.** 개념적 출처로만 "
    "적고, 수치를 인용하지 않는다(AGENTS.md 5절).",
    "",
    "- **Patton(2011)**, *J. Econometrics* — QLIKE·MSE가 불완전 프록시 아래 순위를 보존하는 "
    "유일한 손실이라는 증명. 우리가 QLIKE를 주 지표로 쓰는 근거 그 자체다. "
    "(arxiv 대체 근거: `2110.01189`, `2109.02432` — 둘 다 초록 대조 완료, 암호화폐 적용 포함)",
    "- **Corsi(2009)**, *J. Financial Econometrics* — HAR-RV 원논문. (arxiv 확장: `2205.07719`)",
    "- **Bollerslev(1986)** — GARCH 원논문.",
    "- **Hamilton & Susmel(1994)** · **Gray(1996)** · **Klaassen(2002)** — 레짐 전환 "
    "ARCH/GARCH 계보. 우리 `engine/regime_garch.py`의 collapsing 근사가 Gray·Klaassen을 따른다.",
    "- **Ardia, Bluteau & Rüede(2019)**, *Finance Research Letters*, \"Regime changes in Bitcoin "
    "GARCH volatility dynamics\" — 암호화폐 MS-GARCH의 표준 인용인데 arxiv 미수록.",
    "- **Zakoian(1994)** · **Glosten, Jagannathan & Runkle(1993)** · **Tong(1990)** — "
    "임계값 전환 GARCH·SETAR 계보.",
    "- **Fan & Gijbels(1996)** — 커널 평활의 경계 편향 표준 레퍼런스.",
]


# %% [markdown]
# ## 실행

# %%
def main(argv=None) -> None:
    from report_header import render_standard_header, study_universe, num

    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="3종목·축소 설정으로 배관만 확인")
    ap.add_argument("--probe", action="store_true", help="1종목×2조건을 최대 규모로 돌려 시간 측정")
    ap.add_argument("--n-tickers", type=int, default=20)
    ap.add_argument("--workers", type=int, default=0, help="0이면 가용 메모리에서 자동 산정")
    ap.add_argument("--deadline-h", type=float, default=8.0, help="전체 마감(시간). 넘으면 정리 후 중단")
    ap.add_argument("--report-only", action="store_true",
                    help="적합을 건너뛰고 직전 실행이 저장한 산출물로 보고서만 재생성")
    a = ap.parse_args(argv)

    t_start = time.time()
    deadline = t_start + a.deadline_h * 3600
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    tickers, sel = study_universe()
    if a.quick:
        tickers = tickers[:3]
    elif a.probe:
        tickers = tickers[:1]
    else:
        tickers = tickers[:a.n_tickers]
    conds = [False, True]
    jobs = [(tk, d) for tk in tickers for d in conds]

    if a.workers > 0:
        workers = a.workers
    else:
        # 워커당 약 2.5GB(커널 행렬 1.5GB + 특성·부산물)로 잡고, 메인 프로세스 몫 6GB를 남긴다
        workers = max(1, min(6, int((free_ram_gb() - 6.0) / 2.5), len(jobs)))

    print(f"[시작] 종목 {len(tickers)} × 조건 {len(conds)} = 작업 {len(jobs)}개 · "
          f"CPU 워커 {workers} · GPU {dev} · 마감 {a.deadline_h}시간", flush=True)
    print(f"[자원] 가용 메모리 {free_ram_gb():.1f}GB · 커널 표본 상한 {kernel_max_n():,}개", flush=True)

    all_rows: list[dict] = []
    all_fails: list[dict] = []
    lin_evs: list[dict] = []
    timings: list[dict] = []
    metas: list[dict] = []
    preds_by_key: dict[tuple, dict] = {}
    acts_by_key: dict[tuple, np.ndarray] = {}
    truncated: list[str] = []

    def absorb(res: dict, pass_name: str):
        if res.get("skipped"):
            all_fails.append({"종목": res["ticker"], "주기제거": res["deper"], "모델": "(전체)",
                              "예외": "Skipped", "메시지": res["skipped"]})
            return
        all_rows.extend(res["rows"])
        key = (res["ticker"], res["deper"])
        preds_by_key.setdefault(key, {}).update(res["preds"])
        if "rv_act" in res:
            acts_by_key[key] = res["rv_act"]
        for nm, et, msg in res["fails"]:
            all_fails.append({"종목": res["ticker"], "주기제거": res["deper"], "모델": nm,
                              "예외": et, "메시지": msg})
            print(f"    ! {res['ticker']}(주기제거={res['deper']}) {nm} 실패 — {et}: {msg}", flush=True)
        if res.get("lin_ev"):
            lin_evs.append(res["lin_ev"])
        timings.append({"종목": res["ticker"], "주기제거": res["deper"], "패스": pass_name,
                        "총소요초": res.get("elapsed", np.nan), **res["timings"]})
        metas.append({"종목": res["ticker"], "주기제거": res["deper"], **res["meta"]})
        pd.DataFrame(all_rows).to_csv(RES / f"{STEM}_model_comparison_partial.csv", index=False)

    if a.report_only:
        # 적합을 건너뛰고 직전 실행이 저장한 산출물로 보고서만 다시 만든다. 설명 절을
        # 보강하려고 75분짜리 재적합을 다시 돌리는 건 계산 낭비다(사용자 지적 2026-09-28).
        print("[report-only] 저장된 산출물로 보고서만 재생성한다(적합 건너뜀)", flush=True)
        rd = pd.read_csv(RES / f"{STEM}_model_comparison.csv")
        all_rows = rd.to_dict("records")
        for p, sink in ((RES / f"{STEM}_fit_failures.csv", all_fails),
                        (RES / f"{STEM}_timings.csv", timings),
                        (RES / f"{STEM}_chosen_hyperparams.csv", metas),
                        (RES / f"{STEM}_linearity_evidence.csv", lin_evs)):
            if p.exists() and p.stat().st_size > 2:
                sink.extend(pd.read_csv(p).to_dict("records"))
        z = np.load(RES / f"{STEM}_validation_predictions.npz")
        for k in z.files:
            tk, dp, nm = k.split("|")
            key = (tk, bool(int(dp)))
            if nm == "실제":
                acts_by_key[key] = z[k]
            else:
                preds_by_key.setdefault(key, {})[nm] = z[k]
        tickers = sorted(rd["종목"].unique())
        print(f"[report-only] {len(rd)}행 · 종목 {len(tickers)} · "
              f"모델 {rd['모델'].nunique()} 적재", flush=True)
    else:
        # CPU 패스를 백그라운드로 던져 두고, 그동안 메인 프로세스는 GPU 패스를 돈다.
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(run_cpu_job, tk, d, a.quick): (tk, d) for tk, d in jobs}

            for i, (tk, d) in enumerate(jobs, 1):
                if time.time() > deadline:
                    truncated.append(f"GPU 패스: {len(jobs) - i + 1}개 작업 미실행(마감 초과)")
                    break
                try:
                    res = run_gpu_job(tk, d, a.quick, dev)
                    absorb(res, "GPU")
                    print(f"  [GPU {i}/{len(jobs)}] {tk} 주기제거={d} — {len(res['rows'])}개 모델 "
                          f"({res.get('elapsed', 0):.0f}s, 남은 메모리 {free_ram_gb():.1f}GB)", flush=True)
                except Exception as e:
                    all_fails.append({"종목": tk, "주기제거": d, "모델": "(GPU 패스 전체)",
                                      "예외": type(e).__name__, "메시지": str(e)[:200]})
                    traceback.print_exc()
                relieve_memory()

            done = 0
            for fut in as_completed(futures):
                tk, d = futures[fut]
                done += 1
                if fut.cancelled():
                    continue
                try:
                    res = fut.result()
                    absorb(res, "CPU")
                    print(f"  [CPU {done}/{len(jobs)}] {tk} 주기제거={d} — {len(res['rows'])}개 모델 "
                          f"({res.get('elapsed', 0):.0f}s)", flush=True)
                except Exception as e:
                    all_fails.append({"종목": tk, "주기제거": d, "모델": "(CPU 패스 전체)",
                                      "예외": type(e).__name__, "메시지": str(e)[:200]})
                    traceback.print_exc()
                if time.time() > deadline:
                    pending = [f for f in futures if not f.done()]
                    n_cancel = sum(1 for f in pending if f.cancel())
                    if n_cancel:
                        truncated.append(f"CPU 패스: 대기 중이던 {n_cancel}개 작업 취소(마감 초과)")
                        print(f"  [마감] CPU 대기 작업 {n_cancel}개 취소", flush=True)

        rd = pd.DataFrame(all_rows)

    if not a.report_only:
        rd.to_csv(RES / f"{STEM}_model_comparison.csv", index=False)
        pd.DataFrame(all_fails).to_csv(RES / f"{STEM}_fit_failures.csv", index=False)
        pd.DataFrame(timings).to_csv(RES / f"{STEM}_timings.csv", index=False)
        pd.DataFrame(metas).to_csv(RES / f"{STEM}_chosen_hyperparams.csv", index=False)
        if lin_evs:
            pd.DataFrame(lin_evs).to_csv(RES / f"{STEM}_linearity_evidence.csv", index=False)
        np.savez_compressed(
            RES / f"{STEM}_validation_predictions.npz",
            **{f"{k[0]}|{int(k[1])}|{nm}": v
               for k, dv in preds_by_key.items() for nm, v in dv.items()},
            **{f"{k[0]}|{int(k[1])}|실제": v for k, v in acts_by_key.items()})

    # ── 보고서 ──
    n_models = rd["모델"].nunique()
    emit(f"# 24번 — 변동성 예측 모델 최대 규모 재적합 ({n_models}개 모델)")
    emit()
    emit(f"작성일 2026-09-28 · **{rd['종목'].nunique()}종목** · 15분봉 · 예측 대상 **1시간 뒤 실현변동성** · "
         f"총 실행 {(time.time() - t_start) / 60:.0f}분")
    emit()
    emit(f"> 23번의 축소 설정(트리 300그루·딥러닝 8에폭·은닉 48)을 최대 규모로 키우고, 예비검증에 "
         f"머물러 있던 레짐 전환 GARCH 2종을 본편과 같은 분할·같은 지표로 끌어올려 **{n_models}개 모델**을 "
         "한 표에서 비교한다. 규모를 키우는 만큼 조기종료와 정규화 격자탐색을 함께 넣어 "
         "과적합과 과잉 계산을 같이 막는다.")
    emit()
    if EXCLUDED_MODELS:
        emit("### 이번 회차에서 제외한 모델")
        emit()
        for nm, reason in EXCLUDED_MODELS.items():
            emit(f"- **{nm}** — {reason}")
        emit()
        emit("조용히 빼지 않고 여기 남긴다(AGENTS.md 2.9j). 원인 규명 절차: 학습률 3단 "
             "격자탐색(2e-3/5e-4/1e-4)과 패치 위치 임베딩 추가(`engine/models.py`)를 순서대로 "
             "시도했다. 진단 결과 학습손실은 15에폭에 걸쳐 0.985→0.839로 꾸준히 감소하는데 "
             "검증손실은 0.88~1.03 사이에서 개선 추세 없이 요동만 쳤다 — 그래디언트 소실이나 "
             "학습률 부족이 아니라 일반화 실패다. 같은 은닉폭(128)의 GRU·LSTM은 정상적으로 "
             "여러 에폭에 걸쳐 검증손실이 개선되므로 \"모델이 커서\"로는 설명되지 않는다. "
             "위치 임베딩은 원 PatchTST 논문과 맞추는 정당한 개선이라 코드에는 남겨 두었다.")
        emit()
    print("[header] 표준 헤더 생성 중...", flush=True)
    emit(render_standard_header(
        tickers=tickers, train_frac=TRAIN_FRAC, rep_ticker="KRW-BTC",
        analyzed_tickers=sorted(rd["종목"].unique()), selection_reason=sel,
        transforms=[("모델군별 가정 맞춤 전처리",
                     "GARCH 계열=원 수익률+t분포 / HAR·선형·커널=로그축 / 트리=원 스케일+시간구조 / "
                     "딥러닝=시퀀스(RevIN 또는 전역 표준화)"),
                    ("학습 구간 재분할",
                     f"학습 구간을 내부학습 {INNER_FRAC:.0%} / 내부검증 {1 - INNER_FRAC:.0%}로 시간순 분할해 "
                     "조기종료와 정규화 상수 선택에 쓴다(GARCH 계열은 조정 대상이 없어 학습 구간 전체 사용)"),
                    ("주기 제거 전·후", "시간대 주기가 GARCH 지속성을 왜곡하고 모델이 쉬운 성분만 "
                     "학습할 수 있어 두 조건을 모두 돌린다")]))
    emit("---")
    emit()

    if truncated:
        emit("> ⚠ **이 회차는 마감시각에 걸려 일부 작업이 실행되지 않았다.** " + " / ".join(truncated))
        emit()

    stage_eda(tickers, "KRW-BTC")

    # ── 적합 완결성(성능표보다 먼저) ──
    emit("## 2. 적합 완결성 점검 — 모든 모델이 모든 종목에서 돌았는가")
    emit()
    emit("모델별 평균을 비교하려면 **같은 종목 집합에서 계산된 평균**이어야 한다. 23번 이전 "
         "실행에서 `except: pass`가 예외를 삼켜 20종목 중 2종목에서만 적합된 GARCH가 1위로 "
         "찍힌 적이 있다. 그래서 성능표를 읽기 전에 이 점검을 먼저 싣는다.")
    emit()
    n_all = rd["종목"].nunique()
    n_cell = rd.groupby("모델").size()
    cov = rd.groupby("모델")["종목"].nunique().sort_values()
    expect_cell = n_all * len(conds)
    emit(f"| 모델 | 적합 성공 종목 수 (전체 {n_all}) | 결과 행 수 (기대 {expect_cell}) | 판정 |")
    emit("| :--- | ---: | ---: | :--- |")
    for nm, v in cov.items():
        ok = "정상" if (v == n_all and n_cell[nm] == expect_cell) else \
             f"**⚠ 누락 {n_all - v}종목 / {expect_cell - n_cell[nm]}행 — 평균 비교 불가**"
        emit(f"| {nm} | {v} | {n_cell[nm]} | {ok} |")
    emit()
    total_expect = expect_cell * rd["모델"].nunique()
    emit(f"- 결과 행 수 **{len(rd):,}** (기대 종목 {n_all} × 조건 {len(conds)} × 모델 "
         f"{rd['모델'].nunique()} = {total_expect:,})")
    if all_fails:
        emit(f"- 적합 실패 **{len(all_fails)}건** — 전량 `fit_failures.csv`에 기록했다.")
        fd = pd.DataFrame(all_fails)
        emit()
        emit("| 종목 | 주기제거 | 모델 | 예외 | 메시지 |")
        emit("| :--- | :--- | :--- | :--- | :--- |")
        for _, x in fd.head(30).iterrows():
            emit(f"| {x['종목']} | {x['주기제거']} | {x['모델']} | {x['예외']} | {str(x['메시지'])[:90]} |")
        if len(fd) > 30:
            emit(f"| … | | | | (총 {len(fd)}건, 전량은 CSV 참조) |")
    else:
        emit("- 적합 실패 **0건**. 전 모델이 전 종목·전 조건에서 적합되었다.")
    emit()

    # ── 규모·조기종료 실측 ──
    md = pd.DataFrame(metas)
    emit("## 3. 실제로 쓰인 규모 — 상한과 조기종료 지점")
    emit()
    emit("\"최대 규모\"는 상한을 걸어 둔 것이지 그 값을 다 쓴다는 뜻이 아니다. 조기종료가 "
         "어디서 걸렸는지를 함께 보아야 **규모가 부족했는지 충분했는지** 판단할 수 있다.")
    emit()
    emit("| 항목 | 상한 | 실제 사용(중앙값) | 상한에 닿은 셀 |")
    emit("| :--- | ---: | ---: | ---: |")
    scale_rows = [("LightGBM 그루 수", TREE_MAX_ROUNDS, "lgbm_rounds"),
                  ("XGBoost 그루 수", TREE_MAX_ROUNDS, "xgb_rounds"),
                  ("HistGBM 반복 수", TREE_MAX_ROUNDS, "hist_rounds"),
                  ("GRU 에폭", DL_MAX_EPOCHS, "GRU_epochs"),
                  ("LSTM 에폭", DL_MAX_EPOCHS, "LSTM_epochs"),
                  ("PatchTSTLike 에폭", DL_MAX_EPOCHS, "PatchTSTLike_epochs"),
                  ("ITransformerLike 에폭", DL_MAX_EPOCHS, "ITransformerLike_epochs")]
    for label, cap, col in scale_rows:
        if col in md.columns and md[col].notna().any():
            v = md[col].dropna()
            emit(f"| {label} | {cap:,} | {v.median():,.0f} | {int((v >= cap * 0.98).sum())}/{len(v)} |")
    emit()
    emit(f"- 딥러닝 은닉 폭 **{DL_HIDDEN}**(AGENTS.md 4.3 \"Shallow but Wide\" 권장 64~128의 상단), "
         f"시퀀스 길이 RNN **{SEQ_LEN_MAX['rnn']}봉** · Transformer **{SEQ_LEN_MAX['tfm']}봉**, "
         f"배치 **{DL_BATCH}**(OOM 시 절반으로 자동 축소), AdamW + 코사인 스케줄 + bf16 혼합정밀도, "
         f"학습률은 {{{', '.join(f'{v:g}' for v in DL_LR_GRID)}}} 중 내부검증 손실로 선택.")
    lr_cols = [c for c in md.columns if c.endswith("_lr")]
    if lr_cols:
        emit()
        emit("| 딥러닝 모델 | 선택된 학습률(최빈값) | 조기종료 에폭(중앙값) |")
        emit("| :--- | ---: | ---: |")
        for c in sorted(lr_cols):
            nm = c[:-3]
            v = md[c].dropna()
            ep = md.get(f"{nm}_epochs", pd.Series(dtype=float)).dropna()
            if not len(v):
                continue
            emit(f"| {nm} | {v.mode().iloc[0]:g} | {ep.median():.0f} |" if len(ep)
                 else f"| {nm} | {v.mode().iloc[0]:g} | n/a |")
    emit(f"- 커널 릿지 학습 표본 상한 **{kernel_max_n():,}개** — 가용 메모리 예산 "
         f"{KERNEL_MEM_BUDGET_GB}GB에서 n×n 행렬 크기로 역산한 값이다. SVR은 libsvm의 "
         f"계산 복잡도 때문에 **{SVR_MAX_N:,}개**로 따로 낮췄다.")
    emit(f"- MS-GARCH 재시작 {MS_RESTARTS}회 × 반복 {MS_MAXITER:,} · TAR-GARCH 재시작 "
         f"{TAR_RESTARTS}회 × 반복 {TAR_MAXITER} × 문턱값 후보 {len(TAR_TAU_Q)}개.")
    emit()
    if "ridge_alpha" in md.columns:
        emit("정규화 상수는 내부검증 QLIKE로 셀마다 따로 골랐다. 선택된 값의 분포는 "
             "`chosen_hyperparams.csv`에 있다.")
        emit()

    # ── 성능표 ──
    emit(f"## 4. 전체 성능 — {n_models}개 모델")
    emit()
    emit("QLIKE가 주 지표다. Patton(2011, *J. Econometrics*)이 변동성 대리변수에 잡음이 있을 "
         "때 순위를 보존하는 손실함수가 QLIKE와 MSE뿐임을 증명했고, 그중 QLIKE가 종목 간 "
         "변동성 수준 격차에 덜 민감하다. 이 증명은 arxiv 미수록 고전이라 초록 대조는 못 했지만, "
         "더 최신(2021) arxiv 논문 두 편이 같은 결과를 확장해 **암호화폐 데이터로 직접 검증**했다 "
         "— Wang, An & Zhu(arXiv:2110.01189)는 \"비교 결과가 변동성 프록시의 참값 이탈 정도에 "
         "좌우된다\"를 **비트코인 데이터**로 실증했고, Holzmann & Klar(arXiv:2109.02432)는 "
         "Patton의 robust loss 결과를 일반 적률비로 확장해 **여러 암호화폐의 고빈도 로그수익률**로 "
         "수치 예시를 보였다. MASE는 1보다 작아야 naive(직전 h구간 실현변동성)보다 낫다는 뜻이다.")
    emit()
    for deper in conds:
        s = rd[rd["주기제거"] == deper]
        if not len(s):
            continue
        g = s.groupby(["모델", "계열"]).agg(QLIKE=("QLIKE", "mean"), MSE=("MSE", "mean"),
                                          MZ_R2=("MZ_R2", "mean"), MASE=("MASE", "mean"),
                                          트리밍=("트리밍비율", "mean")).reset_index()
        g = g.sort_values("QLIKE")
        emit(f"### 주기 제거 {'후' if deper else '전'} ({s['종목'].nunique()}종목 평균)")
        emit()
        emit("| 순위 | 계열 | 모델 | QLIKE | MSE | MZ-R² | MASE | 트리밍 |")
        emit("| ---: | :--- | :--- | ---: | ---: | ---: | ---: | ---: |")
        for i, (_, x) in enumerate(g.iterrows(), 1):
            emit(f"| {i} | {x['계열']} | **{x['모델']}** | {num(x['QLIKE'], 4)} | "
                 f"{num(x['MSE'], 10)} | {num(x['MZ_R2'], 4)} | {num(x['MASE'], 3)} | "
                 f"{num(x['트리밍'], 4)} |")
        emit()

    # ── 구간별 분해 ──
    emit("## 5. 구간별 성능 분해 — 단일 승자가 있는가")
    emit()
    emit("전체 평균 한 줄로는 \"어떤 상황에서 잘 맞는가\"를 알 수 없다. 검증 표본을 실제 "
         "변동성 크기로 5등분해 구간마다 따로 본다.")
    emit()
    emit("### 구간 경계를 어떻게 잡았는가 — 기존 기록의 정정")
    emit()
    emit("`process.md`에는 본편(포스터 그림 2)이 **고정 절대구간 풀링**을 썼고 레짐 전환 "
         "예비검증은 **종목별 5분위**를 써서 두 결과를 나란히 놓을 수 없다고 적혀 있었다. "
         "이번에 두 방식을 모두 계산해 본편 산출물(`23_model_comparison_20260908/poster/"
         "regime.csv`)과 직접 대조한 결과 **그 기술이 사실과 다르다**. 본편도 종목별 "
         "5분위였다.")
    emit()
    emit("| naive 모델 · KRW-BTC · 주기제거 후 | Q1 | Q2 | Q3 | Q4 | Q5 |")
    emit("| :--- | ---: | ---: | ---: | ---: | ---: |")
    emit("| 본편 산출물(`poster/regime.csv`) | 1.877 | 1.335 | 1.155 | 0.952 | 0.710 |")
    emit("| **종목별 5분위로 재계산** | **1.88** | **1.34** | **1.15** | **0.95** | **0.71** |")
    emit("| 풀링 절대경계로 재계산 | 1.57 | 1.12 | 0.88 | 0.74 | 0.54 |")
    emit()
    emit("따라서 예비검증의 레짐 전환 결과는 애초부터 본편과 같은 방식으로 계산되고 있었고, "
         "\"계산 방식이 달라 비교 불가\"라는 제약은 존재하지 않았다. 다만 이번 회차는 두 모델을 "
         "**같은 실행·같은 분할·같은 하이퍼파라미터 규모**로 다시 돌렸으므로 그 점에서 정식 "
         "통합의 의미는 그대로다.")
    emit()
    emit("방법 자체로도 종목별 5분위가 맞다. 절대 경계를 모든 종목에 공유하면 가장 낮은 "
         "구간에 **실현변동성이 정확히 0인 검열 관측**이 몰려 실제 평균이 0에 붙고 포착률이 "
         "발산한다(실측: DOGE 주기제거 후 Q1에서 1,892배). 종목 간 변동성 수준이 수십 배 "
         "차이 나는 표본에서는 저변동 종목의 Q5와 고변동 종목의 Q1이 거의 비어 종목 간 "
         "평균 자체가 성립하지 않는다.")
    emit()
    reg, edges_by_cond = regime_decomposition(preds_by_key, acts_by_key)
    reg_pool, pool_edges = regime_decomposition(preds_by_key, acts_by_key, pooled_edges=True)
    if len(reg):
        reg.to_csv(RES / f"{STEM}_regime_decomposition.csv", index=False)
        reg_pool.to_csv(RES / f"{STEM}_regime_decomposition_pooled_edges.csv", index=False)
        order = list(rd.groupby("모델")["QLIKE"].mean().sort_values().index)
        for deper in conds:
            if deper not in edges_by_cond:
                continue
            e = edges_by_cond[deper]
            fp = IMG / f"{STEM}_fig3_regime_{'deper' if deper else 'raw'}.png"
            plot_regime(reg, e, deper, fp, order)
            emit(f"### 주기 제거 {'후' if deper else '전'}")
            emit()
            emit("구간은 종목마다 그 종목의 검증 표본 5분위로 가른다. 아래 경계는 종목별 "
                 f"경계의 중앙값이다(1시간 실현변동성): Q1 | {e[0] * 100:.3f}% | Q2 | "
                 f"{e[1] * 100:.3f}% | Q3 | {e[2] * 100:.3f}% | Q4 | {e[3] * 100:.3f}% | Q5")
            emit()
            emit(f"![그림 3](../../images/{TAG}/{fp.name})")
            emit()
            sub = reg[reg["주기제거"] == deper]
            pq = sub.pivot_table(index="모델", columns="구간", values="QLIKE", aggfunc="mean")
            labs = [c for c in ["Q1", "Q2", "Q3", "Q4", "Q5"] if c in pq.columns]
            emit("| 모델 | " + " | ".join(labs) + " |")
            emit("| :--- | " + " | ".join(["---:"] * len(labs)) + " |")
            for m in [x for x in order if x in pq.index]:
                cells = " | ".join(num(pq.loc[m, c], 3) for c in labs)
                emit(f"| {m} | {cells} |")
            emit()
            winners = {c: pq[c].idxmin() for c in labs if pq[c].notna().any()}
            uniq = sorted(set(winners.values()))
            listed = ", ".join(f"**{c}={winners[c]}**" for c in labs if c in winners)
            if len(uniq) > 1:
                emit(f"→ 구간별 QLIKE 최우수: {listed} — 서로 다른 모델 **{len(uniq)}종**이 "
                     "구간을 나눠 가진다. 전체 평균 1위 하나로 모델을 고르면 다른 구간에서 "
                     "더 나은 모델을 버리게 된다. **왜 이렇게 갈리는지는 §6에서 구간별 "
                     "데이터 특성 실측으로 설명한다.**")
            elif len(uniq) == 1:
                emit(f"→ 구간별 QLIKE 최우수: {listed} — 전 구간에서 **{uniq[0]}**이 최우수다. "
                     "이 조건에서는 단일 승자가 존재한다.")
            emit()
        # ── 보조 표: 풀링 절대경계(다른 질문에 답하는 뷰) ──
        emit("### (보조) 절대 변동성 수준으로 가른 경우")
        emit()
        emit("\"같은 절대 변동성 수준에서 어느 모델이 나은가\"는 위와 다른 질문이다. 20종목 "
             "검증 표본을 전부 모아 5분위 절대 경계를 잡고 모든 종목에 같은 경계를 적용한 "
             "결과를 QLIKE만 싣는다 — 포착률·상대오차는 최저 구간에 검열된 0이 몰려 정의되지 "
             "않는다. 이 표에서는 종목마다 구간별 표본 수가 크게 달라지므로(저변동 종목은 "
             "상위 구간이 거의 비고 고변동 종목은 하위 구간이 거의 빈다) 종목 간 평균을 "
             "**모델 순위 참고용으로만** 본다.")
        emit()
        for deper in conds:
            if deper not in pool_edges:
                continue
            e = pool_edges[deper]
            sub = reg_pool[reg_pool["주기제거"] == deper]
            pq = sub.pivot_table(index="모델", columns="구간", values="QLIKE", aggfunc="mean")
            nq = sub.pivot_table(index="모델", columns="구간", values="n", aggfunc="sum")
            labs = [c for c in ["Q1", "Q2", "Q3", "Q4", "Q5"] if c in pq.columns]
            emit(f"**주기 제거 {'후' if deper else '전'}** — 절대 경계 "
                 + " | ".join(f"{v * 100:.3f}%" for v in e))
            emit()
            emit("| 모델 | " + " | ".join(labs) + " |")
            emit("| :--- | " + " | ".join(["---:"] * len(labs)) + " |")
            for m in [x for x in order if x in pq.index]:
                emit(f"| {m} | " + " | ".join(num(pq.loc[m, c], 3) for c in labs) + " |")
            if "naive" in nq.index:
                emit("| (구간별 관측 수) | " + " | ".join(f"{int(nq.loc['naive', c]):,}" for c in labs) + " |")
            emit()
    else:
        emit("> 구간별 분해에 쓸 예측값이 모이지 않았다(예측 저장 실패).")
        emit()

    # ── 구간별 차이의 원인 설명 (실측 근거 → 알고리즘 특성 → 결과 → 문헌) ──
    emit_regime_explanation(rd, reg, tickers, conds, pd.DataFrame(metas))

    # ── 레짐 전환 2종 집중 비교 ──
    emit("## 7. 레짐 전환 GARCH 2종 — 예비검증에서 본편으로")
    emit()
    emit("MS-GARCH(마르코프 전환)와 TAR-GARCH(임계값 전환)는 국면을 다루는 방식이 서로 "
         "다르다. 하나는 국면이 숨겨져 있다고 보고 확률로 추정하고, 다른 하나는 관측 가능한 "
         "임계변수로 국면을 즉시 가른다. 두 방식이 같은 결론에 도달한다면 특정 구현의 우연일 "
         "가능성이 낮아진다.")
    emit()
    if len(reg):
        for deper in conds:
            sub = reg[reg["주기제거"] == deper]
            pq = sub.pivot_table(index="모델", columns="구간", values="QLIKE", aggfunc="mean")
            base = "GARCH-t"
            if base not in pq.index:
                continue
            emit(f"**주기 제거 {'후' if deper else '전'} — GARCH-t 대비 구간별 QLIKE 차이"
                 "(음수면 레짐 전환 모델이 우세)**")
            emit()
            labs = [c for c in ["Q1", "Q2", "Q3", "Q4", "Q5"] if c in pq.columns]
            emit("| 모델 | " + " | ".join(labs) + " | 전체 평균 |")
            emit("| :--- | " + " | ".join(["---:"] * (len(labs) + 1)) + " |")
            for m in ("MS-GARCH", "TAR-GARCH"):
                if m not in pq.index:
                    continue
                cells = " | ".join(num(pq.loc[m, c] - pq.loc[base, c], 4) for c in labs)
                ov = rd[(rd["주기제거"] == deper) & (rd["모델"] == m)]["QLIKE"].mean() - \
                     rd[(rd["주기제거"] == deper) & (rd["모델"] == base)]["QLIKE"].mean()
                emit(f"| {m} | {cells} | {num(ov, 4)} |")
            emit()

    # ── 소요시간·자원 ──
    td = pd.DataFrame(timings)
    emit("## 8. 소요시간과 자원 — 과잉 계산이 없었는가")
    emit()
    emit("\"오래 걸린다\"가 버그인지 아닌지는 작업별 소요시간이 고르게 분포하는지로 판별한다. "
         "특정 셀만 몇 배로 튀면 수렴 실패나 무한 반복을 의심해야 한다.")
    emit()
    if len(td):
        model_cols = [c for c in td.columns if c not in ("종목", "주기제거", "패스", "총소요초")]
        emit("| 모델 | 셀당 중앙값(초) | 최댓값(초) | 최대/중앙 |")
        emit("| :--- | ---: | ---: | ---: |")
        for c in sorted(model_cols):
            v = td[c].dropna()
            if not len(v) or v.median() <= 0:
                continue
            emit(f"| {c} | {v.median():.1f} | {v.max():.1f} | {v.max() / v.median():.1f}배 |")
        emit()
        emit(f"- 전체 실행 **{(time.time() - t_start) / 60:.0f}분**(마감 상한 {a.deadline_h}시간), "
             f"CPU 워커 {workers}개와 GPU 패스를 겹쳐 돌렸다.")
    emit()

    # ── 한계 ──
    emit("## 9. 이 회차 결과를 어디까지 믿을 수 있는가")
    emit()
    emit("**(1) 가정은 여전히 충족되지 않았다.** 756조합 전수진단(2026-09-07)에서 GARCH 계열이 "
         "요구하는 4개 가정을 동시에 만족하는 조합은 0개였고, Hill 꼬리지수 2.6~2.8로 첨도가 "
         "이론상 무한이다. 규모를 키운다고 이 문제가 풀리지 않는다. 따라서 아래 수치는 "
         "**예측력 비교** 용도이며, 모수 추정치의 해석(지속성·반감기 등)에는 쓰지 않는다.")
    emit()
    emit("**(2) 23번 수치와 직접 빼서 비교하지 않는다.** 이 회차는 학습 구간을 내부학습/"
         "내부검증으로 다시 나눴으므로 학습에 쓰인 행 수 자체가 다르다. 두 회차의 비교는 "
         "\"같은 프로토콜 안에서의 모델 간 순위\" 수준에서만 유효하다.")
    emit()
    emit("**(3) 표본은 성격이 다른 두 집단이 섞여 있다.** 거래대금 상위 10종목과 변동성 상위 "
         "10종목의 내부 상관이 0.555 대 0.156으로 3.6배 차이 난다(23번 (M-2)). 20종목 평균 "
         "하나로 요약하면 이 구조가 가려진다 — 군별 분리 보고는 후속 과제로 남는다.")
    emit()
    emit("**(4) HistGBM의 조기종료는 다른 트리와 방식이 다르다.** sklearn 내장 조기종료는 검증 "
         "분할을 무작위로 뽑아 시계열에서 낙관적으로 멈추므로, 반복 수를 단계적으로 늘리며 "
         "내부검증 손실이 개선되지 않을 때 멈추는 방식으로 대체했다. 탐색 격자가 거칠어 "
         "LightGBM·XGBoost보다 조기종료 지점이 정밀하지 않다.")
    emit()

    out = RES / f"{STEM}_comparison_report.md"
    out.write_text("\n".join(_LINES) + "\n", encoding="utf-8")
    print(f"\n저장: {out} · 총 {(time.time() - t_start) / 60:.1f}분 · 결과 {len(rd)}행", flush=True)


if __name__ == "__main__":
    main()
