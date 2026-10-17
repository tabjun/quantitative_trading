# %% [markdown]
# # 28번: 학습 절차 정비 + 시드 확장(R 재측정으로 시드 수 결정)
#
# ## 실험 설계
#
# **왜 하는가.** 27번은 시드 5개로 "순환 신경망(GRU·LSTM)이 병렬 딥러닝보다 낫다"와 국면별 최선을 보고했다. 그런데
# 2026-10-13 구현 리뷰(연구 트랙 `test/process.md` ①~⑦)에서 학습 절차가 모델마다 달랐다는 점이 확인됐다.
# ① neuralforecast 6종은 조기 종료 뒤 최선 가중치를 복원하지 않았다. ② 같은 6종이 학습 횟수 상한 1000에 걸려
# 끊겼다(Autoformer 35~48%, iTransformer 25~33%). PatchTST·Autoformer의 라이브러리 기본 상한은 5000인데 27번이 1000으로
# 낮췄다. 학습률도 탐색 없이 기본값이었다. ③ GRU·LSTM은 학습률 후보의 상단(2e-3)이 71% 선택돼 범위가 위에서 잘렸을 수
# 있다. ④ 26c의 메모리 부족 되돌리기 코드가 옵티마이저 상태를 얕게 복사했다. ⑥ 보정 계수는 내부 모델로 추정하는데
# 평가 예측은 재적합 모델이 내서, 두 모델이 어긋났다(15분 QLIKE +0.09 악화). 그리고 5시드 검정은 시드 오차를 빼고
# 계산해 1종 오류가 부풀 수 있다(중앙 쌍 약 9%, 90분위 쌍 약 19%, 명목 5%). 이번 실험은 절차를 고친 뒤 시드 수를
# 근거에 따라 정해 다시 돌린다.
#
# **연구 질문.**
# 1. 학습 절차를 맞춘 뒤에도 27번의 결론(순환 신경망 > 병렬 딥러닝, 구간·국면별 최선)이 유지되는가?
# 2. 시드 오차를 검정에 넣고 시드를 충분히 늘렸을 때, 어떤 우위가 남고 어떤 우위가 "구분 안 됨"으로 바뀌는가?
#
# **비교 대상.** 27번 로스터에서 KernelRidge-RBF·SVR-RBF를 뺀 26종(naive 포함, 아래 "설계 변경"). 결정론 모델:
# GARCH-t·MS-GARCH·TAR-GARCH(최우추정), 파운데이션 7종(zero-shot). 시드로 흔들리는 모델 15종: 트리 3종(LightGBM·XGBoost·
# HistGBM), GARCH+LightGBM, Nystroem+Ridge, GRU·LSTM, neuralforecast 6종(PatchTST·iTransformer·TCN·Autoformer·TimesNet·
# TimeXer), 공식 코드 2종(ModernTCN·S-Mamba). 선형 예측기(Linear·Ridge·HAR-RV·DLinear·NLinear)는 2026-09-07·10-05
# 결정대로 넣지 않는다.
#
# **설계 변경(2026-10-15, 시범 실행 시드 0 도중 사용자 결정, 요청 177·179).** KernelRidge-RBF·SVR-RBF를 적합하지 않고
# 순위·검정·결론에서 뺀다. 두 모델만 계산량 때문에 학습 데이터를 잘라 쓴다(SVR 최근 8,000행, KernelRidge 메모리 예산
# 1.5GB로 최근 14,188행, 내부학습 구간 약 6만 2천 행의 13%·23%). 그래서 성적이 알고리즘 때문인지 덜 본 데이터 때문인지
# 구분할 수 없어 공정 비교가 성립하지 않는다. 시드 0에서 먼저 끝난 3종목(BTC·XRP·DOGE)에서 SVR은 무재적합으로 잘라 쓰는
# 창이 평가에서 멀어지자 26c보다 QLIKE가 0.015~0.087 나빠졌다. 지금까지 상위권이었던 적이 없고 개선 여지도 불분명한데
# 실행 시간과 메모리(CPU 작업 최대 5.6GB의 대부분)를 가장 많이 쓴다. 커널 계열은 학습 데이터 전체를 쓰는 Nystroem+Ridge가
# 계속 대표하므로, 2026-09-07 "선형 → 비선형·커널 계열로 대체" 결정의 취지는 유지된다. 이미 끝난 3종목의 두 모델 값은
# 작업 캐시에 기록으로만 남고 결과 파일·검정에 들어가지 않는다.
#
# **설계 변경 2(2026-10-18, 버그 수정, 요청 185).** 26c에서 물려받은 GARCH-t 적합(`garch_t_fit`, 수익률 ×100 척도 1회)이
# 일부 종목에서 arch 시작값(α = 0.1, β = 0.88)에 고착된 채 끝나 최대우도 추정이 아니었다(28번 내부학습 적합 5종목, 26c 전체 적합
# 6종목). 여러 척도로 적합해 수렴했고 시작값이 아닌 해 중 로그우도 최대를 쓰는 `garch_t_fit_robust`로 바꾸고, GARCH-t와 그 예측을
# 특징으로 쓰는 GARCH+LightGBM이 들어 있는 CPU 작업(시드 0~4)을 다시 돌린다. GPU 작업(신경망)은 GARCH를 쓰지 않아 그대로 둔다.
# 고치기 전 CPU 작업 결과는 `parts/s{시드}/cpu_garch_x100/`에 보존해 보고서에서 전후를 비교한다. 26·26c·27번 결과는 완료 실험이라
# 다시 돌리지 않고, 영향 범위를 28번 보고서에 적는다.
#
# **설계 변경 3(2026-10-18, 버그 수정, 요청 185).** 26c가 쓰던 `fit_tar_garch`(τ 후보 × 시작점 2개, Nelder-Mead 600회)가
# 반복 상한에서 멈춘 해를 그대로 쓰고 예외를 조용히 건너뛰었다(ETH·BTC 등에서 이어서 최적화하면 로그우도 +40~+93). τ 후보마다 수렴할
# 때까지 이어서 최적화한 로그우도로 τ를 고르는 `fit_tar_garch_robust`로 바꾸고 진단을 남긴다. GARCH-t 수정과 함께 CPU 작업을 다시 돌린다.
# 같은 날 재사용 적합 함수를 모두 점검했다: MS-GARCH(제약판)는 20종목 수렴 표시이나 9종목이 허용 경계에 닿아 있다(26c부터 기록된 한계).
#
# **설계 변경 4(2026-10-18, 사용자 결정, 요청 187).** GARCH 사양을 데이터로 고르는 결정론 모델 `GARCH-BIC`를 순차 묶음에 더한다.
# 2026-09-07 가정 진단에서 상위 14개 사양이 모두 FIGARCH였는데 번호 실험은 GARCH(1,1)-t만 썼다. 후보(실행 전 고정): 변동성 식
# GARCH(1,1)·GARCH(2,1)·GARCH(1,2)·GJR-GARCH(1,1,1)·FIGARCH(1,d,1) × 분포 t·skew-t = 10개, 평균 0. 종목마다 내부학습 구간에서 각 후보를
# ×100·×1000·×10 척도로 적합해(수렴한 해 중 로그우도 최대) ×100 척도 BIC가 가장 작은 사양을 고른다. 예측은 고른 사양을 고정 모수로
# 전체 계열에 걸어 arch의 해석적 다단계 분산 예측을 쓰고, 보정·두 부분 처리 없음은 GARCH-t와 같다. EGARCH는 해석적 다단계 예측이 없어
# 후보에서 뺐다. 기존 GARCH(1,1)-t는 그대로 두어 사양 선택의 효과를 비교한다. 게이트: GARCH(1,1)-t 후보의 arch 예측 경로가
# 26c 닫힌 식 경로(`garch_multistep`)와 같아야 한다(예측 시점 정렬 확인).
#
# **고정 조건(26c·27과 같음).** DB `upbit_data_20261005.db`, 20종목(`study_universe`), 15분봉, 창 2023-10-06 ~
# 2026-10-05, 분할 2025-11-11(전 종목 공통), 내부학습 85%/내부검증 15%(시간순), 예측 구간 15분·30분·1시간·4시간·
# 12시간, 정시 비겹침 평가, 손실 QLIKE, 가격 정지 두 부분 모형(정지 확률 π × 크기), 모델별 입력 표현은 27번과 같다
# (GRU·LSTM = 15분봉 96개, 신규 신경망 = 블록 로그 RV 이력). 입력 표현 차이로 생기는 교란은 27c에서 따로 통제했고,
# 이번 실험은 입력을 바꾸지 않는다.
#
# **이번 회차에 바꾸는 학습 절차(전 모델 같은 원칙).**
# - **무재적합 통일(⑥)**: 모든 학습 모델은 내부학습 구간으로 한 번만 적합하고, 그 모델이 내부검증 예측(보정 계수
#   추정)과 평가 예측을 함께 낸다. GARCH 3종, 커널, 트리, 신경망, 정지 확률 분류기 모두 같다. 대가: 평가 예측이
#   학습 구간 끝 15%의 데이터로 학습하지 않는다(입력 창으로는 본다).
# - **학습률 탐색(②③)**: 학습하는 신경망 10종 모두 기본 학습률 × {0.25, 1, 4}의 3점을 내부검증 구간의 척도 보정
#   QLIKE(26c 커널 모델과 같은 기준)로 고른다. 기본 학습률: GRU·LSTM 2e-3(26c 최다 선택값), neuralforecast는
#   라이브러리 기본값(PatchTST·Autoformer·TimesNet 1e-4, iTransformer·TCN·TimeXer 1e-3), ModernTCN 1e-4·S-Mamba 7e-5
#   (저자 설정). 탐색은 시드 0에서 (종목, 구간, 모델)마다 한 번 하고, 고른 값을 모든 시드에 고정한다(시드마다
#   다시 고르면 시드 변동에 탐색 변동이 섞인다).
# - **최선 가중치 복원(①)**: neuralforecast 학습에 검증 손실이 가장 낮았던 시점의 가중치를 학습 끝에 되돌리는
#   콜백을 넣는다(GRU·LSTM·ModernTCN·S-Mamba는 이미 복원한다).
# - **학습 횟수 상한(②)**: neuralforecast 6종 모두 5,000(PatchTST·Autoformer의 라이브러리 기본값), 검증 100스텝마다,
#   인내 5회. 상한 도달 비율을 모델별로 보고한다.
# - **되돌리기 코드(④)**: 옵티마이저 상태를 깊은 복사로 저장한다.
#
# **시드 수 결정 규칙(시범 실행 전에 고정).** 시드 0~4로 시범 실행한 뒤 R = σ²_시드 / SE²_시간을 잰다.
# σ²_시드는 시드별 평균 손실차의 시드 간 분산, SE_시간은 시드 평균 손실차 시계열의 Newey-West HAC 표준오차다.
# 대상 쌍은 27번에서 쟀던 것과 같은 "순환 신경망 2종 × 학습하는 병렬 딥러닝 8종 × 5구간 = 80쌍"(전체 평가 기간,
# 종목 평균 손실)이다. 시드 5개 분산은 자유도 4라 흔들리므로 쌍들의 log σ²에 경험적 베이즈 축소를 한 값으로 R을
# 구한다. 그 90분위 R90에 대해 √(1 + R90/S) ≤ 1.10(시드 항이 표준오차를 10% 넘게 키우지 않음)을 만족하는 가장 작은
# S를 후보 {20, 30, 50}에서 고른다. 20 미만은 쓰지 않는다(시드 표준편차 추정에 20개 이상, Colas 외 2018). 50으로도
# 만족하지 않으면 50으로 하고 10%를 넘는 쌍을 보고서에 적는다. 결정한 S는 결과를 보기 전에 기록하고, 결과를 본 뒤
# 바꾸지 않는다(선택적 중단 방지).
#
# **테스트 케이스와 판정 기준.**
# - 게이트: (a) 결정론 모델의 시드별 예측이 시드 0과 같다(최대차 1e-9), (b) 모든 시드 파일에 시드 모델 15종 × 100칸이
#   다 있다, (c) 모든 모델의 평가 시각·실제값이 같다, (d) 파운데이션 모델 재계산이 27번 저장값의 π 교체만 반영한다.
# - 검정: 모든 DM 검정의 표준오차를 SE_전체 = √(SE²_시간 + σ²_시드/S)로 계산하고 Holm 보정을 건다. 판정은 "A 우세 /
#   B 우세 / 구분 안 됨". 시드 항이 표준오차를 10% 넘게 키운 쌍은 따로 적는다.
# - 비교 기준: 27번 결과(같은 데이터·같은 평가)와 칸별 순위·판정이 어떻게 달라졌는지 표와 그림으로 대조한다. 바뀐
#   곳은 절차 수정 중 무엇 때문인지 진단한다(상한 도달 비율, 고른 학습률, 보정 계수).

# %%
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import pickle
import sys
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams["font.family"] = ["Noto Sans CJK KR", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


def _project_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "engine").is_dir() and (p / "test" / "models").is_dir():
            return p
    raise RuntimeError(f"프로젝트 루트(engine/·test/models/)를 찾지 못했다: {start}")


ROOT = _project_root(Path(__file__).resolve())
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "test" / "scripts"))
TAG = "28_seed_expansion_20261015"
STEM = "28_seed_expansion"
RES = ROOT / "test" / "results" / TAG
IMG = ROOT / "test" / "images" / TAG
PARTS = RES / "parts"          # 작업 단위 캐시(커밋하지 않는다)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


X = _load("x27_for28", ROOT / "test" / "models" / "27_model_expansion_test.py")
M = X.M

# %% [markdown]
# ## 설정

# %%
SEEDS_PILOT = (0, 1, 2, 3, 4)
S_CANDIDATES = (20, 30, 50)
INFL_MAX = 1.10
LR_MULT = (0.25, 1.0, 4.0)
DEFAULT_LR = {"GRU": 2e-3, "LSTM": 2e-3, "PatchTST": 1e-4, "iTransformer": 1e-3, "TCN": 1e-3, "Autoformer": 1e-4,
              "TimesNet": 1e-4, "TimeXer": 1e-3, "ModernTCN": 1e-4, "S-Mamba": 7e-5}
NF_MAX_STEPS, NF_VAL_CHECK, NF_PATIENCE = 5000, 100, 5

RNN_MODELS = ("GRU", "LSTM")
NF_MODELS = X.NF_MODELS                                   # neuralforecast 6종
OFFICIAL_MODELS = ("ModernTCN", "S-Mamba")
BLOCK_MODELS = NF_MODELS + OFFICIAL_MODELS
GPU_MODELS = RNN_MODELS + BLOCK_MODELS
CPU_STOCH = ("Nystroem+Ridge", "LightGBM", "XGBoost", "HistGBM", "GARCH+LightGBM")
STOCHASTIC = CPU_STOCH + GPU_MODELS                       # 15종
CPU_DET = ("GARCH-t", "MS-GARCH", "TAR-GARCH")
GARCH_BIC = "GARCH-BIC"            # 설계 변경 4: 시드 0에서 한 번 적합하는 결정론 모델(parts/s0/garchbic)
# 설계 변경(2026-10-15, 요청 177·179): 학습 데이터를 잘라 쓰는 두 커널 모델은 적합하지 않는다(설계 셀 참고).
DECISION_EXCLUDED = ("KernelRidge-RBF", "SVR-RBF")
FM_MODELS = X.FM_MODELS
TTM_UNSUPPORTED_H = X.TTM_UNSUPPORTED_H
ALL_MODELS = ("naive",) + CPU_DET + (GARCH_BIC,) + CPU_STOCH + GPU_MODELS + FM_MODELS
R_PAIRS_A = RNN_MODELS
R_PAIRS_B = BLOCK_MODELS                                  # 학습하는 병렬 딥러닝 8종

FAMILY = {**{m: X.NEW_FAMILY.get(m, M.FAMILY.get(m)) for m in ALL_MODELS}, GARCH_BIC: M.FAMILY.get("GARCH-t")}
PROCESS = {**M.PROCESS, **X.NEW_PROCESS, GARCH_BIC: "다단계 예측(사양 BIC 선택)"}
M.FAMILY[GARCH_BIC] = M.FAMILY["GARCH-t"]          # 26c Scorer가 행을 만들 때 읽는 사전에도 등록
M.PROCESS[GARCH_BIC] = PROCESS[GARCH_BIC]


def lr_grid(name: str) -> tuple:
    return tuple(DEFAULT_LR[name] * k for k in LR_MULT)


def hlabel(H: int) -> str:
    return M.hlabel(H)


# %% [markdown]
# ## 정지 확률 π(무재적합)
#
# 26c·27은 π도 "내부 분류기로 보정용 π, 전체 재적합 분류기로 평가 π"였다. 28번은 내부 분류기 하나로 둘 다 낸다.
# 분류기는 시드와 무관하게 random_state=0, deterministic 모드로 한 번 적합해 모든 모델·시드가 같은 π를 쓴다.

# %%
def fit_pi_all(D: dict, HD: dict, quick: bool) -> tuple[np.ndarray, dict]:
    """반환: 모든 예측 행(HD['j'])의 π(내부 분류기), 진단."""
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    Xz = M.zero_inputs(D, HD, M.make_features(D)["tree"][HD["j"]], M.zero_features(D))
    z = (HD["act"] == 0).astype(int)
    ti, tv, va, te = HD["tr_in_all"], HD["tr_val_all"], HD["va"], HD["te"]
    info = {"정지_내부검증": float(z[va].mean()), "정지_평가": float(z[te].mean()),
            "정지수_내부학습": int(z[ti].sum()), "정지수_내부검증": int(z[tv].sum())}
    if z[ti].sum() < M.ZERO_MIN_IN or z[tv].sum() < M.ZERO_MIN_VAL:
        info.update(분류기="생략(정지 표본 부족, π=0)", 반복=0)
        return np.zeros(len(z)), info
    rounds = M.ZERO_ROUNDS if not quick else 60
    kw = dict(learning_rate=M.ZERO_LR, num_leaves=63, max_depth=8, min_child_samples=100, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.8, n_jobs=M.TREE_THREADS, deterministic=True,
              force_row_wise=True, verbosity=-1, random_state=0)
    clf = lgb.LGBMClassifier(n_estimators=rounds, **kw)
    clf.fit(Xz[ti], z[ti], eval_set=[(Xz[tv], z[tv])], eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(M.ZERO_EARLY_STOP if not quick else 10, verbose=False)])
    pi = np.clip(clf.predict_proba(Xz)[:, 1], 0, M.PI_MAX)
    info.update(분류기="LightGBM(내부 분류기, 무재적합)", 반복=int(clf.best_iteration_ or rounds),
                π평균_평가=float(pi[te].mean()),
                AUC_평가=float(roc_auc_score(z[te], pi[te])) if 0 < z[te].mean() < 1 else np.nan)
    return pi, info


def _pi_path(tk: str, H: int, quick: bool) -> Path:
    return PARTS / "pi" / f"{tk}_{H}{'_quick' if quick else ''}.pkl"


def get_pi(tk: str, H: int, D: dict, HD: dict, quick: bool) -> tuple[np.ndarray, dict]:
    f = _pi_path(tk, H, quick)
    if f.exists():
        pi, info = pickle.loads(f.read_bytes())
        if len(pi) != len(HD["j"]):
            raise AssertionError(f"π 캐시 길이 불일치 {tk} {H}")
        return pi, info
    pi, info = fit_pi_all(D, HD, quick)
    _atomic_write(f, pickle.dumps((pi, info)))
    return pi, info


def _atomic_write(f: Path, b: bytes) -> None:
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(f".tmp{os.getpid()}")
    tmp.write_bytes(b)
    tmp.replace(f)


def pi_job(tk: str, quick: bool) -> str:
    D = M.build_data(tk)
    for H in M.HORIZONS_H:
        get_pi(tk, H, D, M.horizon_data(D, H), quick)
    return tk


# %% [markdown]
# ## GARCH-t 적합(2026-10-18 버그 수정)
#
# 26c의 `garch_t_fit`은 수익률 ×100 척도로 `arch`를 한 번 적합한다. 이 척도에서 일부 종목은 최적화가 arch의 시작값
# (α = 0.1, β = 0.88)에서 움직이지 않은 채 "수렴"으로 끝난다. 26c 전체 적합은 6종목(AQT·BTC·MOC·STRAX·XLM·XRP), 28번 내부학습
# 적합은 5종목(BTC·DOGE·SHIB·XLM·TOKAMAK)이 그랬고, ×1000 척도로 다시 적합하면 로그우도가 68~639 높았다(서비스 트랙 S194 발견,
# 연구 세션 재확인). 최대우도 추정이 아니므로 고친다. 여러 척도(×100·×1000·×10)로 적합해, 수렴 플래그가 0이고 시작값에 고착되지
# 않은 해 중 ×100 척도로 환산한 로그우도가 가장 큰 해를 쓴다. 반환 형식(×100 척도의 ω와 조건부 분산)은 26c와 같다.

# %%
GARCH_SCALES = (100, 1000, 10)


def garch_t_fit_robust(dc: np.ndarray, split_c: int) -> dict:
    from arch import arch_model
    from scipy.signal import lfilter
    tried, best = [], None
    for k in GARCH_SCALES:
        try:
            res = arch_model(dc[:split_c] * k, mean="Zero", vol="GARCH", p=1, q=1, dist="t").fit(disp="off", show_warning=False)
        except Exception as e:
            tried.append({"척도": k, "오류": type(e).__name__})
            continue
        p = res.params
        al, be, nu = float(p["alpha[1]"]), float(p["beta[1]"]), float(p["nu"])
        stuck = abs(al - 0.1) < 1e-6 and abs(be - 0.88) < 1e-6
        ll100 = float(res.loglikelihood) + split_c * np.log(k / 100)       # ×k 척도 로그우도를 ×100 척도로 환산
        ok = int(res.convergence_flag) == 0 and not stuck and np.isfinite(ll100)
        r = {"척도": k, "수렴플래그": int(res.convergence_flag), "시작값고착": stuck, "LL100": ll100, "ok": ok,
             "omega": float(p["omega"]) * (100 / k) ** 2, "alpha": al, "beta": be, "nu": nu,
             "cv_in": np.asarray(res.conditional_volatility) ** 2 * (100 / k) ** 2}
        tried.append({kk: v for kk, v in r.items() if kk != "cv_in"})
        if best is None or (r["ok"], r["LL100"]) > (best["ok"], best["LL100"]):
            best = r
    if best is None or not best["ok"]:
        raise RuntimeError(f"GARCH-t: 모든 척도에서 정상 수렴 실패 {tried}")
    x = dc * 100.0
    n = len(x)
    cv = np.full(n, np.nan)
    cv[:split_c] = best["cv_in"]
    drive = best["omega"] + best["alpha"] * x[split_c - 1:n - 1] ** 2
    cv[split_c:], _ = lfilter([1.0], [1.0, -best["beta"]], drive, zi=np.array([best["beta"] * cv[split_c - 1]]))
    ll100 = {t_["척도"]: t_.get("LL100", np.nan) for t_ in tried}
    return dict(cv=cv, omega=best["omega"], alpha=best["alpha"], beta=best["beta"], nu=best["nu"], scale=best["척도"],
                ll_gain_vs_x100=float(best["LL100"] - ll100.get(100, np.nan)),
                x100_stuck=any(t_.get("척도") == 100 and t_.get("시작값고착") for t_ in tried), tried=tried)


# %% [markdown]
# ## TAR-GARCH 적합(2026-10-18 버그 수정)
#
# 26c는 `engine.regime_garch.fit_tar_garch`를 τ 후보 4개 × 시작점 2개, Nelder-Mead 최대 600회로 부른다. 이 함수는 최적화가 반복
# 상한에 닿아 멈춰도 수렴 여부를 보지 않고, 예외도 조용히 건너뛴다. 20종목 진단(2026-10-18)에서 고른 해가 상한 600회에서 멈췄고,
# 같은 τ에서 최적화를 이어가면 로그우도가 더 올랐다(ETH +92.5, BTC +40.1). 그래서 τ 후보마다 같은 시작점 탐색을 한 뒤 가장 나은 해에서
# 수렴할 때까지(최대 TAR_POLISH_ITER회, 허용오차 1e-4) 이어서 최적화하고, 이어간 로그우도로 τ를 고른다(프로파일 우도). 예외와 미수렴은
# 진단에 남긴다. 공용 engine은 24·26번 재현을 위해 고치지 않고 이 함수를 28번에만 쓴다.

# %%
TAR_POLISH_ITER = 4000


def fit_tar_garch_robust(r_pct: np.ndarray, sw: np.ndarray, split_idx: int, taus, n_restarts: int, maxiter: int,
                         quick: bool = False) -> tuple[float, dict, dict]:
    from scipy.optimize import minimize
    from engine import regime_garch as rg
    eps, swt = r_pct[:split_idx], sw[:split_idx]
    diag = {"τ후보": [], "예외": 0}
    best_all = None
    for tau in taus:
        best = None
        for x0 in rg._TAR_INITS[:n_restarts]:
            try:
                res = minimize(lambda th: -rg.tar_filter(eps, swt, tau, th)[0], x0, method="Nelder-Mead",
                               options=dict(maxiter=maxiter, xatol=1e-3, fatol=1e-3))
            except Exception:
                diag["예외"] += 1
                continue
            if best is None or -res.fun > best[0]:
                best = (-res.fun, res.x, bool(res.success), int(res.nit))
        if best is None:
            diag["τ후보"].append({"tau": float(tau), "상태": "모든 시작점 예외"})
            continue
        res2 = minimize(lambda th: -rg.tar_filter(eps, swt, tau, th)[0], best[1], method="Nelder-Mead",
                        options=dict(maxiter=TAR_POLISH_ITER if not quick else 200, xatol=1e-4, fatol=1e-4))
        ll2 = -res2.fun if np.isfinite(res2.fun) and -res2.fun >= best[0] else best[0]
        x2 = res2.x if ll2 > best[0] else best[1]
        diag["τ후보"].append({"tau": float(tau), "탐색_ll": float(best[0]), "탐색_수렴": best[2], "탐색_반복": best[3],
                             "이어서_ll": float(ll2), "이어서_수렴": bool(res2.success), "이어서_반복": int(res2.nit),
                             "ll이득": float(ll2 - best[0])})
        if best_all is None or ll2 > best_all[0]:
            best_all = (ll2, x2, float(tau), bool(res2.success))
    if best_all is None:
        raise RuntimeError(f"TAR-GARCH: 모든 τ에서 적합 실패 {diag}")
    ll, theta, tau, conv = best_all
    _, info = rg.tar_filter(r_pct, sw, tau, theta, loglik_upto=split_idx)
    if info is None:
        raise RuntimeError("TAR-GARCH: 최종 필터가 발산(info=None)")
    diag.update(선택_tau=tau, 선택_ll=float(ll), 선택_수렴=conv,
                최대_ll이득=float(max(c.get("ll이득", 0.0) for c in diag["τ후보"])))
    return tau, info, diag


# %% [markdown]
# ## GARCH 사양 선택(설계 변경 4): GARCH-BIC

# %%
GARCH_BIC_CANDS = tuple((f"{nm}-{dist}", vol, kw, dist) for nm, vol, kw in (
    ("GARCH(1,1)", "GARCH", dict(p=1, o=0, q=1)), ("GARCH(2,1)", "GARCH", dict(p=2, o=0, q=1)),
    ("GARCH(1,2)", "GARCH", dict(p=1, o=0, q=2)), ("GJR(1,1,1)", "GARCH", dict(p=1, o=1, q=1)),
    ("FIGARCH(1,d,1)", "FIGARCH", dict(p=1, q=1))) for dist in ("t", "skewt"))


def fit_garch_spec(x100: np.ndarray, vol: str, kw: dict, dist: str) -> dict:
    """한 사양을 여러 척도로 적합해 수렴한 해 중 ×100 환산 로그우도 최대를 고른다(P8: 한 번 적합 금지)."""
    from arch import arch_model
    n, best, tried = len(x100), None, []
    for k in GARCH_SCALES:
        try:
            r = arch_model(x100 * (k / 100), mean="Zero", vol=vol, dist=dist, **kw).fit(disp="off", show_warning=False)
        except Exception as e:
            tried.append(f"×{k} {type(e).__name__}")
            continue
        ll = float(r.loglikelihood) + n * np.log(k / 100)
        tried.append(f"×{k} 플래그 {int(r.convergence_flag)} LL {ll:.1f}")
        if int(r.convergence_flag) == 0 and np.isfinite(ll) and (best is None or ll > best["LL100"]):
            best = {"LL100": ll, "scale": k, "params": r.params, "npar": len(r.params)}
    if best is None:
        return {"ok": False, "tried": tried}
    best.update(ok=True, BIC=float(-2 * best["LL100"] + best["npar"] * np.log(n)), tried=tried)
    return best


def garch_spec_paths(dc: np.ndarray, spec: tuple, fit: dict, pos: np.ndarray, m: int) -> np.ndarray:
    """고정 모수로 전체 계열에 걸어 예측 시점 pos(목표 창 첫 봉)마다 m봉 분산 경로(×100 척도 퍼센트²)."""
    from arch import arch_model
    _, vol, kw, dist = spec
    k = fit["scale"]
    start = int(pos.min()) - 1
    am = arch_model(dc * k, mean="Zero", vol=vol, dist=dist, **kw)
    var = am.forecast(params=fit["params"], horizon=m, start=start, method="analytic", reindex=False).variance.to_numpy()
    return var[pos - 1 - start] * (100 / k) ** 2


def garchbic_job(tk: str, quick: bool) -> dict:
    t0 = time.time()
    D = M.build_data(tk)
    comp, dc, _ = M._compact(D)
    HDs = {H: M.horizon_data(D, H) for H in M.HORIZONS_H}
    inner = HDs[M.HORIZONS_H[0]]["inner"]
    if any(HD["inner"] != inner for HD in HDs.values()):
        raise AssertionError("구간마다 내부학습 경계가 다르다")
    ic = int(np.searchsorted(D["grid"][comp], inner))
    cands = GARCH_BIC_CANDS if not quick else GARCH_BIC_CANDS[:2]
    fits = {c[0]: fit_garch_spec(dc[:ic] * 100, c[1], c[2], c[3]) for c in cands}
    ok = {k_: v for k_, v in fits.items() if v["ok"]}
    if not ok:
        raise RuntimeError(f"{tk}: GARCH 사양 후보가 모두 수렴 실패")
    chosen = min(ok, key=lambda k_: ok[k_]["BIC"])
    spec = next(c for c in cands if c[0] == chosen)
    # 게이트: GARCH(1,1)-t 후보의 arch 예측 경로 = 닫힌 식 경로(시점 정렬 확인)
    G = garch_t_fit_robust(dc, ic)
    HDg = HDs[60] if 60 in HDs else HDs[M.HORIZONS_H[0]]
    pos_g = M._origin_pos(comp, HDg["j"], HDg["m"])[HDg["te"]][:200]
    f11 = fit_garch_spec(dc[:ic] * 100, "GARCH", dict(p=1, o=0, q=1), "t")
    a = garch_spec_paths(dc, GARCH_BIC_CANDS[0], f11, pos_g, HDg["m"])
    b = M.garch_multistep(G, pos_g, HDg["m"])
    gap = float(np.max(np.abs(a / b - 1)))
    if gap > 1e-3:
        raise AssertionError(f"{tk}: arch 예측 경로와 닫힌 식 경로가 다르다(최대 상대차 {gap:.3g})")
    results = []
    for H, HD in HDs.items():
        S = M.Scorer(tk, HD)
        j, te, va, m = HD["j"], HD["te"], HD["va"], HD["m"]
        pos_all = M._origin_pos(comp, j, m)
        rawv, _ = M.var_to_rv(garch_spec_paths(dc, spec, ok[chosen], pos_all[va], m), D["cfac"], j[va])
        raw, _ = M.var_to_rv(garch_spec_paths(dc, spec, ok[chosen], pos_all[te], m), D["cfac"], j[te])
        c = S.calib(rawv)
        S.score(GARCH_BIC, raw, c=c, note=f"{chosen} · BIC 선택 · 무재적합")
        results.append(dict(H=H, rows=S.rows, preds=S.preds, act=S.act.astype(np.float32), T=M._ns(HD["T"][te])))
    table = [{"사양": k_, "수렴": v["ok"], "BIC": v.get("BIC", np.nan), "LL100": v.get("LL100", np.nan),
              "척도": v.get("scale"), "시도": "; ".join(v["tried"])} for k_, v in fits.items()]
    return dict(ticker=tk, chosen=chosen, table=table, gate_gap=gap, results=results, elapsed=time.time() - t0)


def _garchbic_task(tk, quick):
    f = PARTS / f"s0{'_quick' if quick else ''}" / "garchbic" / f"{tk}.pkl"
    if f.exists():
        return "skip", tk
    res = garchbic_job(tk, quick)
    _atomic_write(f, pickle.dumps(res))
    return f"{res['elapsed']:.0f}s {res['chosen']} 게이트 {res['gate_gap']:.1e}", tk


# %% [markdown]
# ## CPU 작업: 결정론 모델(시드 0만)과 트리·커널 근사(모든 시드), 무재적합

# %%
def run_cpu_job(tk: str, seed: int, quick: bool) -> dict:
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.kernel_approximation import Nystroem
    from sklearn.pipeline import make_pipeline
    from sklearn.ensemble import HistGradientBoostingRegressor
    import lightgbm as lgb
    import xgboost as xgb
    from engine import regime_garch as rg

    t0 = time.time()
    det = seed == 0
    D = M.build_data(tk)
    F = M.make_features(D)
    comp, dc, split_c = M._compact(D)
    HDs = {H: M.horizon_data(D, H) for H in M.HORIZONS_H}
    fails, timings, meta, results = [], {}, {"종목": tk, "시드": seed}, []

    # GARCH 계열: 내부학습 구간까지로만 적합(보정·평가 예측 모두 이 적합)
    inner_fits: dict = {}
    for inner in sorted({HD_["inner"] for HD_ in HDs.values()}):
        ic = int(np.searchsorted(D["grid"][comp], inner))
        f_ = {}
        t = time.time()
        try:
            f_["GARCH-t"] = garch_t_fit_robust(dc, ic)
            g_ = f_["GARCH-t"]
            meta.update({f"garch_{pd.Timestamp(inner).strftime('%Y%m%d')}_{k}": g_[k]
                         for k in ("scale", "alpha", "beta", "nu", "ll_gain_vs_x100", "x100_stuck")})
        except Exception as e:
            fails.append(("GARCH-t", "*", type(e).__name__, str(e)[:160]))
        timings["GARCH-t"] = timings.get("GARCH-t", 0) + time.time() - t
        if det:
            t = time.time()
            try:
                info_, dg_ = M.fit_ms(dc, ic, quick)
                f_["MS-GARCH"] = info_
                meta.update({f"ms_{k}": v for k, v in dg_.items()}, ms_p11=info_["p11"], ms_p22=info_["p22"])
            except Exception as e:
                fails.append(("MS-GARCH", "*", type(e).__name__, str(e)[:160]))
            timings["MS-GARCH"] = timings.get("MS-GARCH", 0) + time.time() - t
            t = time.time()
            try:
                sw_ = M.tar_switch(dc)
                taus_ = np.quantile(sw_[M.TAR_SWITCH_BARS:ic], list(M.TAR_TAU_Q))
                tau_, info_, tdiag = fit_tar_garch_robust(dc * 100, sw_, ic, taus_,
                                                          n_restarts=M.TAR_RESTARTS if not quick else 1,
                                                          maxiter=M.TAR_MAXITER if not quick else 100, quick=quick)
                f_["TAR-GARCH"] = info_
                meta.update(tar_tau=float(tau_), tar_선택_수렴=tdiag["선택_수렴"], tar_최대_ll이득=tdiag["최대_ll이득"],
                            tar_예외=tdiag["예외"], tar_진단=json.dumps(tdiag["τ후보"], ensure_ascii=False))
            except Exception as e:
                fails.append(("TAR-GARCH", "*", type(e).__name__, str(e)[:160]))
            timings["TAR-GARCH"] = timings.get("TAR-GARCH", 0) + time.time() - t
        inner_fits[inner] = f_

    for H in M.HORIZONS_H:
        HD = HDs[H]
        S = M.Scorer(tk, HD)
        j, te, tr_in, tr_val, va = HD["j"], HD["te"], HD["tr_in"], HD["tr_val"], HD["va"]
        m = HD["m"]
        pi, _ = get_pi(tk, H, D, HD, quick)
        S.set_zero(pi[va], pi[te])
        if det:
            S.score("naive", HD["nai"][te])
        pos_all = M._origin_pos(comp, j, m)
        fits = inner_fits.get(HD["inner"], {})
        g_dep_in = None
        for nm, fn in (("GARCH-t", "g"), ("MS-GARCH", "ms"), ("TAR-GARCH", "tar")):
            info = fits.get(nm)
            if info is None:
                continue
            t = time.time()
            try:
                def path(pos):
                    return (M.garch_multistep(info, pos, m) if fn == "g" else
                            M.ms_multistep(info, pos, m) if fn == "ms" else M.tar_multistep(info, dc, pos, m))
                if fn == "g":
                    raw_all, g_dep_in = M.var_to_rv(path(pos_all), D["cfac"], j)
                    rawv, raw = raw_all[va], raw_all[te]
                else:
                    rawv, _ = M.var_to_rv(path(pos_all[va]), D["cfac"], j[va])
                    raw, _ = M.var_to_rv(path(pos_all[te]), D["cfac"], j[te])
                c_g = S.calib(rawv)
                S.score(nm, raw, c=c_g, note="다단계 예측 · 내부학습 적합 하나로 보정·평가(무재적합)")
                meta[f"{nm}_h{H}_calib"] = c_g
            except Exception as e:
                fails.append((nm, H, type(e).__name__, str(e)[:160]))
            timings[f"{nm}_h{H}"] = time.time() - t

        y = np.log(HD["act_d"].clip(1e-300))
        Xf = {k: v[j] for k, v in F.items()}

        def fit_log_model(name: str, fit_fn, note: str = ""):
            """fit_fn() → (내부 모델의 va 로그 예측, 같은 내부 모델의 평가 로그 예측)."""
            t_ = time.time()
            try:
                pv, pt = fit_fn()
                cal = S.calib2(S.to_raw(pv, va))
                S.score_log(name, S.to_raw(pt, te), cal, note=note)
                meta[f"{name}_h{H}_calib"] = cal["c"]
                meta[f"{name}_h{H}_calib_pos"] = cal["c_pos"]
            except Exception as e:
                fails.append((name, H, type(e).__name__, str(e)[:160]))
                traceback.print_exc()
            timings[f"{name}_h{H}"] = time.time() - t_

        sci = StandardScaler().fit(Xf["log"][tr_in])
        ymi, ysi = float(y[tr_in].mean()), float(y[tr_in].std())
        Zi, Zv, Zt = sci.transform(Xf["log"][tr_in]), sci.transform(Xf["log"][va]), sci.transform(Xf["log"][te])
        yi_s = (y[tr_in] - ymi) / ysi

        def chunked(mdl, Z, chunk=4000):
            return np.concatenate([mdl.predict(Z[i:i + chunk]) for i in range(0, len(Z), chunk)])

        # KernelRidge-RBF·SVR-RBF는 적합하지 않는다(DECISION_EXCLUDED, 설계 변경 2026-10-15). 26c의 두 함수는
        # 학습 데이터 최근 14,188행·8,000행만 써서 다른 모델과 공정 비교가 성립하지 않는다.

        def fit_nys():
            nc = M.NYSTROEM_COMPONENTS if not quick else 256
            best = None
            for a in M.RIDGE_ALPHAS:
                mdl = make_pipeline(Nystroem(kernel="rbf", n_components=nc, random_state=seed),
                                    Ridge(alpha=a)).fit(Zi, yi_s)
                pv = mdl.predict(Zv) * ysi + ymi
                q = S.select_score(S.to_raw(pv, va))
                if best is None or q < best[0]:
                    best = (q, a, pv, mdl.predict(Zt) * ysi + ymi)
            meta[f"nys_alpha_h{H}"] = best[1]
            return best[2], best[3]
        fit_log_model("Nystroem+Ridge", fit_nys, "무재적합")

        rounds = M.TREE_MAX_ROUNDS if not quick else 60
        esr = M.TREE_EARLY_STOP if not quick else 10

        def lgb_on(Xt_, key):
            mdl = lgb.LGBMRegressor(n_estimators=rounds, learning_rate=M.TREE_LR, max_depth=8, num_leaves=127,
                                    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, min_child_samples=40,
                                    n_jobs=M.TREE_THREADS, verbosity=-1, random_state=seed)
            mdl.fit(Xt_[tr_in], y[tr_in], eval_set=[(Xt_[tr_val], y[tr_val])], eval_metric="l2",
                    callbacks=[lgb.early_stopping(esr, verbose=False)])
            best_n = int(mdl.best_iteration_ or rounds)
            meta[f"{key}_h{H}_rounds"] = best_n
            return mdl.predict(Xt_[va], num_iteration=best_n), mdl.predict(Xt_[te], num_iteration=best_n)
        fit_log_model("LightGBM", lambda: lgb_on(Xf["tree"], "lgbm"), "무재적합")

        def fit_xgb():
            Tx = Xf["tree"]
            mdl = xgb.XGBRegressor(n_estimators=rounds, early_stopping_rounds=esr, learning_rate=M.TREE_LR, max_depth=8,
                                   subsample=0.8, colsample_bytree=0.8, min_child_weight=10, tree_method="hist",
                                   device="cpu", n_jobs=M.TREE_THREADS, verbosity=0, random_state=seed)
            mdl.fit(Tx[tr_in], y[tr_in], eval_set=[(Tx[tr_val], y[tr_val])], verbose=False)
            best_n = int(getattr(mdl, "best_iteration", rounds - 1) or 0) + 1
            meta[f"xgb_h{H}_rounds"] = best_n
            return (mdl.predict(Tx[va], iteration_range=(0, best_n)), mdl.predict(Tx[te], iteration_range=(0, best_n)))
        fit_log_model("XGBoost", fit_xgb, "무재적합")

        def fit_hist():
            Tx = Xf["tree"]
            best_loss, best_it, pat, best = np.inf, 0, 0, None
            step = max(50, rounds // 20)
            kw = dict(learning_rate=M.TREE_LR, max_depth=8, max_leaf_nodes=127, min_samples_leaf=40,
                      early_stopping=False, random_state=seed)
            mdl = HistGradientBoostingRegressor(max_iter=step, warm_start=True, **kw)
            for it in range(step, rounds + 1, step):
                mdl.set_params(max_iter=it)
                mdl.fit(Tx[tr_in], y[tr_in])
                loss = float(np.mean((mdl.predict(Tx[tr_val]) - y[tr_val]) ** 2))
                if loss < best_loss - 1e-9:
                    best_loss, best_it, pat, best = loss, it, 0, (mdl.predict(Tx[va]), mdl.predict(Tx[te]))
                else:
                    pat += 1
                    if pat >= 2:
                        break
            meta[f"hist_h{H}_rounds"] = best_it
            return best
        fit_log_model("HistGBM", fit_hist, "무재적합")

        if g_dep_in is not None:
            Xin_ = np.column_stack([Xf["tree"], np.log(g_dep_in + 1e-12)])
            fit_log_model("GARCH+LightGBM", lambda: lgb_on(Xin_, "hyb"), "내부학습 GARCH-t 특성 · 무재적합")
        else:
            fails.append(("GARCH+LightGBM", H, "SkippedDependency", "GARCH-t 적합 실패"))

        results.append(dict(H=H, rows=S.rows, preds=S.preds, rows1=S.rows1, preds1=S.preds1,
                            pi=S.pi_te.astype(np.float32), act=S.act.astype(np.float32),
                            nai=S.nai.astype(np.float32), T=M._ns(HD["T"][te]),
                            nai_tr=HD["nai"][HD["tr"]].astype(np.float32)))
    return dict(ticker=tk, seed=seed, results=results, fails=fails, timings=timings, meta=meta,
                elapsed=time.time() - t0)


# %% [markdown]
# ## GPU 작업 1: GRU·LSTM(15분봉 96개 입력, 26c 학습 절차 + 되돌리기 수정 + 학습률 탐색, 무재적합)

# %%
def train_dl_once(name, Xtr, ytr, Xval, yval, Xva, Xte, dev, max_epochs, patience, batch, lr, seed=0):
    import torch
    import torch.nn as nn
    from engine.models import make_model
    torch.manual_seed(seed)
    model = make_model(name, Xtr.shape[1], Xtr.shape[2], M.DL_HIDDEN).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)
    lossf = nn.MSELoss()
    amp = dev == "cuda"
    xt = torch.as_tensor(Xtr, device=dev); yt = torch.as_tensor(ytr, dtype=torch.float32, device=dev)
    xv = torch.as_tensor(Xval, device=dev); yv = torch.as_tensor(yval, dtype=torch.float32, device=dev)

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
        # ④ 수정: 옵티마이저 상태(모멘텀 텐서)까지 깊은 복사해야 메모리 부족 때 실제로 되돌린다
        snap = ({k: v.detach().clone() for k, v in model.state_dict().items()}, copy.deepcopy(opt.state_dict()))
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
            M.relieve_memory()
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
    pv = infer(torch.as_tensor(Xva, device=dev), bs).cpu().numpy().ravel()
    pt = infer(torch.as_tensor(Xte, device=dev), bs).cpu().numpy().ravel()
    del xt, yt, xv, yv, model, best_state
    M.relieve_memory()
    return pv, pt, dict(epochs_run=epoch, best_epoch=best_epoch, batch=bs, lr=lr, val_mse=best_loss,
                        cap_hit=bool(epoch >= max_epochs and best_epoch >= max_epochs - patience))


def rnn_job(tk: str, H: int, name: str, seed: int, quick: bool) -> dict:
    import torch
    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    D = M.build_data(tk)
    HD = M.horizon_data(D, H)
    S = M.Scorer(tk, HD)
    pi, _ = get_pi(tk, H, D, HD, quick)
    S.set_zero(pi[HD["va"]], pi[HD["te"]])
    j, te, tr_in, tr_val, va = HD["j"], HD["te"], HD["tr_in"], HD["tr_val"], HD["va"]
    y = np.log(HD["act_d"].clip(1e-300))
    ymu, ysd = float(y[tr_in].mean()), float(y[tr_in].std())
    Xs = M.make_sequences(D, j, M.SEQ_LEN if not quick else 16)
    grid = lr_grid(name) if seed == 0 else (chosen_lr(tk, H, name, quick),)
    if quick:
        grid = grid[:2] if seed == 0 else grid
    cands = []
    for lr in grid:
        pv, pt, info = train_dl_once(name, Xs[tr_in], (y[tr_in] - ymu) / ysd, Xs[tr_val], (y[tr_val] - ymu) / ysd,
                                     Xs[va], Xs[te], dev, M.DL_MAX_EPOCHS if not quick else 3,
                                     M.DL_PATIENCE if not quick else 2, M.DL_BATCH, lr, seed=seed)
        q = S.select_score(S.to_raw(pv * ysd + ymu, va))
        cands.append((q, lr, pv, pt, info))
    q, lr, pv, pt, info = min(cands, key=lambda c: c[0])
    cal = S.calib2(S.to_raw(pv * ysd + ymu, va))
    S.score_log(name, S.to_raw(pt * ysd + ymu, te), cal,
                note=f"lr {lr:g} · {info['best_epoch']}/{info['epochs_run']}에폭 · 무재적합")
    meta = {"종목": tk, "H": H, "모델": name, "시드": seed, "lr": lr, "선택점수": q,
            "후보": json.dumps({f"{c[1]:g}": c[0] for c in cands}), "best_step": info["best_epoch"],
            "run_steps": info["epochs_run"], "상한": M.DL_MAX_EPOCHS if not quick else 3, "상한도달": info["cap_hit"],
            "보정계수": cal["c_pos"], "초": time.time() - t0}
    return dict(ticker=tk, H=H, model=name, seed=seed, rows=S.rows, preds=S.preds, rows1=S.rows1,
                preds1=S.preds1, meta=meta)


# %% [markdown]
# ## GPU 작업 2: 블록 입력 신경망 8종(neuralforecast 6 + ModernTCN + S-Mamba), 무재적합
#
# 27번은 (내부학습 → 내부검증 예측)과 (학습 전체 → 평가 예측) 두 번 적합했다. 28번은 내부학습 블록 [0, k_in)으로
# 한 번 적합해 [k_in, nb) 전체(내부검증 + 평가)를 예측한다. 조기 종료 검증은 27번과 같이 [0, k_in)의 끝 15%다.

# %%
def _restore_best_callback():
    import pytorch_lightning as pl

    class RestoreBest(pl.Callback):
        """검증 손실이 가장 낮았던 시점의 가중치를 학습 끝에 되돌린다(① 수정)."""

        def __init__(self):
            self.best, self.state, self.best_step, self.last_step = np.inf, None, 0, 0

        def on_train_start(self, trainer, pl_module):
            self.best, self.state, self.best_step, self.last_step = np.inf, None, 0, 0

        def on_validation_end(self, trainer, pl_module):
            if trainer.sanity_checking:
                return
            v = trainer.callback_metrics.get("ptl/val_loss")
            if v is None:
                return
            v = float(v)
            if np.isfinite(v) and v < self.best - 1e-12:
                self.best, self.best_step = v, int(trainer.global_step)
                self.state = {k: t.detach().cpu().clone() for k, t in pl_module.state_dict().items()}

        def on_train_end(self, trainer, pl_module):
            self.last_step = int(trainer.global_step)
            if self.state is not None:
                pl_module.load_state_dict(self.state)

    return RestoreBest()


def make_nf_model(name: str, H: int, seed: int, lr: float, quick: bool, cb):
    from neuralforecast.losses.pytorch import MSE
    from neuralforecast import models as nfm
    L = X.INPUT_BLOCKS[H] if not quick else 16
    common = dict(h=1, input_size=L, loss=MSE(), valid_loss=MSE(), max_steps=NF_MAX_STEPS if not quick else 60,
                  val_check_steps=NF_VAL_CHECK if not quick else 20, early_stop_patience_steps=NF_PATIENCE,
                  learning_rate=lr, windows_batch_size=X.NF_WINDOWS_BATCH, inference_windows_batch_size=512,
                  valid_batch_size=512, random_seed=seed, logger=False, enable_progress_bar=False,
                  enable_model_summary=False, callbacks=[cb])
    kw = {"iTransformer": dict(n_series=2), "TimeXer": dict(n_series=1)}.get(name, {})
    return getattr(nfm, name)(**common, **kw)


def nf_predict(name: str, B: dict, ymu: float, ysd: float, train_end: int, end: int, H: int, seed: int, lr: float,
               quick: bool) -> tuple[np.ndarray, dict]:
    """27의 nf_predict와 같은 입력·검증 분할. 다른 점: 학습률, 상한, 최선 가중치 복원."""
    from neuralforecast import NeuralForecast
    parts = X.nf_frame(name, B, ymu, ysd, end)
    if name == "iTransformer":
        rsd = float(np.std(B["ret"][:train_end])) or 1.0
        parts.append(pd.DataFrame({"unique_id": "ret", "ds": B["ds"][:end], "y": B["ret"][:end] / rsd,
                                   "available_mask": 1.0}))
    df = pd.concat(parts, ignore_index=True)
    cb = _restore_best_callback()
    mdl = make_nf_model(name, H, seed, lr, quick, cb)
    nf = NeuralForecast(models=[mdl], freq=f"{H}min")
    cv = nf.cross_validation(df, val_size=int(X.VAL_FRAC * train_end), test_size=end - train_end, n_windows=None,
                             step_size=1, refit=False)
    cv = cv[cv["unique_id"] == "rv"].sort_values("ds")
    if len(cv) != end - train_end:
        raise AssertionError(f"{name}: 예측 길이 {len(cv)} != {end - train_end}")
    if not np.array_equal(pd.DatetimeIndex(cv["ds"]).values, B["ds"][train_end:end].values):
        raise AssertionError(f"{name}: 예측 시각이 블록과 다르다")
    p = cv[name].to_numpy(float)
    if not np.all(np.isfinite(p)):
        raise FloatingPointError(f"{name}: 예측에 비유한값")
    # NeuralForecast는 모델을 깊은 복사해 학습한다. 학습에 실제로 쓰인 콜백은 복사본 쪽에 있다.
    used = [c for c in nf.models[0].trainer_kwargs.get("callbacks", []) if type(c).__name__ == "RestoreBest"]
    if len(used) != 1:
        raise RuntimeError(f"{name}: 학습에 쓰인 최선 가중치 콜백을 찾지 못했다({len(used)}개)")
    cb = used[0]
    if cb.state is None:
        raise RuntimeError(f"{name}: 검증 손실이 한 번도 기록되지 않아 최선 가중치를 되돌리지 못했다")
    cap = NF_MAX_STEPS if not quick else 60
    return p, dict(best_step=cb.best_step, run_steps=cb.last_step, 상한=cap, 상한도달=bool(cb.last_step >= cap),
                   val_mse=cb.best)


def smamba_predict(B: dict, ymu: float, ysd: float, train_end: int, end: int, H: int, seed: int, lr: float,
                   quick: bool) -> tuple[np.ndarray, dict]:
    """27 러너(`27_smamba_run.py`)를 격리 venv에서 모듈로 불러 학습률만 바꿔 실행한다(러너 파일은 고치지 않는다)."""
    import subprocess
    import tempfile
    L = X.INPUT_BLOCKS[H] if not quick else 16
    rsd = float(np.std(B["ret"][:train_end])) or 1.0
    libs = ":".join(str(d) for d in sorted((X.SMAMBA_VENV / "lib" / "python3.10" / "site-packages" / "nvidia").glob("*/lib")))
    runner = ROOT / "test" / "models" / "27_smamba_run.py"
    code = ("import importlib.util,sys;"
            f"s=importlib.util.spec_from_file_location('smr',{str(runner)!r});m=importlib.util.module_from_spec(s);"
            f"s.loader.exec_module(m);m.LR={lr!r};sys.argv=['smr','--inp',sys.argv[1],'--out',sys.argv[2]];m.main()")
    with tempfile.TemporaryDirectory() as td:
        fin, fout = Path(td) / "in.npz", Path(td) / "out.npz"
        np.savez(fin, y=((X.locf(B["y"]) - ymu) / ysd).astype(np.float32), ret=(B["ret"] / rsd).astype(np.float32),
                 ok=B["ok"], train_end=train_end, end=end, L=L, seed=seed, quick=quick)
        env = {**os.environ, "LD_LIBRARY_PATH": libs + ":" + os.environ.get("LD_LIBRARY_PATH", "")}
        r = subprocess.run([str(X.SMAMBA_VENV / "bin" / "python"), "-c", code, str(fin), str(fout)],
                           capture_output=True, text=True, env=env)
        if r.returncode != 0:
            raise RuntimeError(f"S-Mamba 러너 실패: {r.stderr[-300:]}")
        p = np.load(fout)["p"]
    return p, dict(best_step=np.nan, run_steps=np.nan, 상한=10, 상한도달=np.nan, val_mse=np.nan)


def moderntcn_predict(B, ymu, ysd, train_end, end, H, seed, lr, quick):
    X.MTCN_LR = lr                      # modern_tcn_predict가 실행 시점에 읽는 모듈 상수
    return X.modern_tcn_predict(B, ymu, ysd, train_end, end, H, seed, quick), dict(
        best_step=np.nan, run_steps=np.nan, 상한=X.MTCN_EPOCHS, 상한도달=np.nan, val_mse=np.nan)


def block_prepare(tk: str, H: int, quick: bool) -> dict:
    D = M.build_data(tk)
    HD = M.horizon_data(D, H)
    B = X.block_data(D, H)
    MK = X.block_masks(D, HD, B)
    pi, _ = get_pi(tk, H, D, HD, quick)
    HD = dict(HD)
    HD["va"] = MK["va"]                 # 27과 같이 보정 시점을 블록 경계 정시로 한정
    S = M.Scorer(tk, HD)
    S.set_zero(pi[MK["va"]], pi[MK["te"]])
    ymu, ysd = X.standardize(B, MK["k_in"])
    return dict(D=D, HD=HD, B=B, MK=MK, S=S, ymu=ymu, ysd=ysd)


def block_job(tk: str, H: int, name: str, seed: int, quick: bool) -> dict:
    t0 = time.time()
    P = block_prepare(tk, H, quick)
    B, MK, S, ymu, ysd = P["B"], P["MK"], P["S"], P["ymu"], P["ysd"]
    fn = {"S-Mamba": smamba_predict, "ModernTCN": moderntcn_predict}.get(name)
    grid = lr_grid(name) if seed == 0 else (chosen_lr(tk, H, name, quick),)
    if quick:
        grid = grid[:2] if seed == 0 else grid
    cands = []
    for lr in grid:
        if fn is None:
            p, info = nf_predict(name, B, ymu, ysd, MK["k_in"], B["nb"], H, seed, lr, quick)
        else:
            p, info = fn(B, ymu, ysd, MK["k_in"], B["nb"], H, seed, lr, quick)
        full = np.full(B["nb"], np.nan)
        full[MK["k_in"]:] = p
        pv = full[MK["kj"][MK["va"]]] * ysd + ymu
        q = S.select_score(S.to_raw(pv, MK["va"]))
        cands.append((q, lr, full, info))
    q, lr, full, info = min(cands, key=lambda c: c[0])
    X.score_blocks(S, name, full, MK, ymu, ysd, f"lr {lr:g} · 최선 가중치 · 무재적합")
    meta = {"종목": tk, "H": H, "모델": name, "시드": seed, "lr": lr, "선택점수": q,
            "후보": json.dumps({f"{c[1]:g}": c[0] for c in cands}), **info,
            "보정계수": S.rows[-1]["보정계수"], "초": time.time() - t0}
    return dict(ticker=tk, H=H, model=name, seed=seed, rows=S.rows, preds=S.preds, rows1=S.rows1,
                preds1=S.preds1, meta=meta)


# %% [markdown]
# ## 작업 저장·이어하기
#
# 작업 하나 = (종목, 구간, 모델, 시드)의 GPU 적합 또는 (종목, 시드)의 CPU 적합. 끝나면 바로 pkl로 저장하고, 다시
# 실행하면 저장된 작업은 건너뛴다. 시드 1 이상은 시드 0이 고른 학습률을 읽으므로 시드 0 저장분이 있어야 한다.

# %%
def part_path(kind: str, tk: str, seed: int, H: int | None = None, name: str | None = None, quick: bool = False) -> Path:
    q = "_quick" if quick else ""
    if kind == "cpu":
        return PARTS / f"s{seed}{q}" / "cpu" / f"{tk}.pkl"
    return PARTS / f"s{seed}{q}" / "gpu" / f"{name}_{tk}_{H}.pkl"


def chosen_lr(tk: str, H: int, name: str, quick: bool) -> float:
    f = part_path("gpu", tk, 0, H, name, quick)
    if not f.exists():
        raise RuntimeError(f"시드 0 학습률 선택이 없다: {f.name}(시드 0을 먼저 실행)")
    return float(pickle.loads(f.read_bytes())["meta"]["lr"])


def _peak_rss_gb() -> float:
    """작업 프로세스의 최대 상주 메모리(GB). 2026-10-15 메모리 초과로 jupyterhub 서비스째 재시작된 뒤 추가."""
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2


def _cpu_task(tk, seed, quick):
    f = part_path("cpu", tk, seed, quick=quick)
    if f.exists():
        return "skip", tk
    res = run_cpu_job(tk, seed, quick)
    res["meta"]["최대RSS_GB"] = _peak_rss_gb()
    _atomic_write(f, pickle.dumps(res))
    return f"{res['elapsed']:.0f}s 실패 {len(res['fails'])}", tk


def _gpu_task(tk, H, name, seed, quick):
    f = part_path("gpu", tk, seed, H, name, quick)
    if f.exists():
        return "skip", f.name
    try:
        res = rnn_job(tk, H, name, seed, quick) if name in RNN_MODELS else block_job(tk, H, name, seed, quick)
    except Exception as e:
        traceback.print_exc()
        return f"실패 {type(e).__name__}: {str(e)[:160]}", f.name
    res["meta"]["최대RSS_GB"] = _peak_rss_gb()
    _atomic_write(f, pickle.dumps(res))
    return f"{res['meta']['초']:.0f}s lr {res['meta']['lr']:g} RSS {res['meta']['최대RSS_GB']:.1f}GB", f.name


# 공유 서버 자원 한도(AGENTS.md 2.14). 2026-10-15 03:11 GPU 4 + CPU 6 워커가 RAM 30GB를 넘겨 jupyterhub.service가
# 재시작되고 그 안의 모든 프로세스(서비스 세션 포함)가 종료됐다. 고정 워커 수 대신, 작업을 하나 넣을 때마다
# "사용 가능 메모리 − 이미 돌고 있는 작업이 앞으로 더 쓸 몫 ≥ 예비분 + 새 작업 추정치"일 때만 넣는다.
# 한도는 LIMITS 파일로 실행 중에도 바꿀 수 있다(다른 세션이 무거운 작업을 할 때 낮춘다).
LIMITS = RES / "parts" / "_limits.json"
LIMIT_DEFAULT = {"reserve_gb": 6.0, "floor_gb": 4.5, "gpu_reserve_gb": 4.0, "max_gpu_workers": 3, "max_cpu_workers": 4,
                 "track_ram_gb": 15.0,
                 "est_gb": {"cpu_s0": 3.5, "cpu": 3.0, "gpu": 2.8, "pi": 2.0}}
# cpu_s0 실측: 커널 2종 포함 5.6GB(종목당 약 20분) → 제외 후 3.15GB(약 10~12분), 2026-10-17 ETH·SOL.


def limits() -> dict:
    d = json.loads(json.dumps(LIMIT_DEFAULT))
    try:
        d.update(json.loads(LIMITS.read_text()))
    except Exception:
        pass
    return d


def _avail_gb() -> float:
    return M.free_ram_gb()


def _gpu_free_gb() -> float:
    try:
        import torch
        return torch.cuda.mem_get_info()[0] / 1024 ** 3 if torch.cuda.is_available() else 99.0
    except Exception:
        return 99.0


def _pool_rss_gb(pool) -> float:
    import psutil
    tot = 0.0
    for pid in list(getattr(pool, "_processes", {}) or {}):
        try:
            tot += psutil.Process(pid).memory_info().rss / 1024 ** 3
        except Exception:
            pass
    return tot


def _track_rss_gb() -> float:
    """연구 대기열 전체(같은 세션 ID = setsid로 띄운 queue.sh와 그 하위 프로세스)의 상주 메모리 합(GB).
    GPU 줄과 CPU 줄이 서로 다른 프로세스라, 트랙 몫(AGENTS.md 2.14 연구 RAM 15GB)은 세션 단위로 센다."""
    import psutil
    sid, tot = os.getsid(0), 0.0
    for p in psutil.process_iter(["pid"]):
        try:
            if os.getsid(p.info["pid"]) == sid:
                tot += p.memory_info().rss / 1024 ** 3
        except Exception:
            pass
    return tot


def run_pool(tasks: list, fn, workers: int, label: str, retries: int = 1, kind: str = "cpu") -> list:
    """작업을 메모리 한도 안에서 병렬로 돌린다. workers는 상한이고, 실제 동시 실행 수는 메모리가 정한다.
    실패한 작업은 retries번 다시 돌리고, 끝까지 실패한 작업 목록을 돌려준다."""
    import multiprocessing as mp
    from concurrent.futures import wait, FIRST_COMPLETED
    ctx = mp.get_context("spawn")
    pending = list(tasks)
    for attempt in range(retries + 1):
        failed, done_n, t0, last_note = [], 0, time.time(), 0.0
        queue = list(pending)
        with ProcessPoolExecutor(max_workers=max(workers, 1), mp_context=ctx) as pool:
            running: dict = {}
            while queue or running:
                L = limits()
                cap = min(workers, int(L["max_gpu_workers"] if kind == "gpu" else L["max_cpu_workers"]))
                est = float(L["est_gb"].get(kind, 3.0))
                if queue and len(running) < max(cap, 1):
                    owed = max(0.0, len(running) * est - _pool_rss_gb(pool))       # 돌고 있는 작업이 더 쓸 몫(추정)
                    room = _avail_gb() - owed
                    gpu_ok = kind != "gpu" or _gpu_free_gb() >= float(L["gpu_reserve_gb"])
                    track_ok = _track_rss_gb() + owed + est <= float(L["track_ram_gb"])
                    if (not running) and _avail_gb() >= float(L["floor_gb"]) + est and gpu_ok or \
                            room >= float(L["reserve_gb"]) + est and gpu_ok and track_ok:
                        t = queue.pop(0)
                        running[pool.submit(fn, *t)] = t
                        continue
                    if time.time() - last_note > 600:
                        print(f"  [{label}] 메모리 대기: 사용 가능 {_avail_gb():.1f}GB, 연구 합계 {_track_rss_gb():.1f}GB, 예약 {owed:.1f}GB, "
                              f"동시 {len(running)}/{cap}", flush=True)
                        last_note = time.time()
                if not running:
                    time.sleep(15)
                    continue
                fin, _ = wait(list(running), timeout=15, return_when=FIRST_COMPLETED)
                for fu in fin:
                    t = running.pop(fu)
                    try:
                        msg, nm = fu.result()
                    except Exception as e:
                        msg, nm = f"실패 {type(e).__name__}: {str(e)[:160]}", str(t)
                    if msg.startswith("실패"):
                        failed.append(t)
                    done_n += 1
                    print(f"  [{label} {done_n}/{len(pending)}] {nm} {msg} · 경과 {(time.time() - t0) / 60:.1f}분 · "
                          f"여유 {_avail_gb():.1f}GB · 동시 {len(running) + 1}", flush=True)
        if not failed:
            return []
        print(f"[{label}] 실패 {len(failed)}개 재시도({attempt + 1}/{retries})", flush=True)
        pending = failed
    return pending


def status(msg: str) -> None:
    RES.mkdir(parents=True, exist_ok=True)
    with open(RES / "parts" / "_status.txt" if (RES / "parts").exists() else RES / "_status.txt", "a") as fh:
        fh.write(f"{time.strftime('%F %T')} {msg}\n")
    print(f"[상태] {msg}", flush=True)


# %% [markdown]
# ## 시드별 결과 묶기
#
# 시드 s의 예측 파일 = 결정론 모델(naive·GARCH 3종·커널 2종, 시드 0 적합) + 시드 s의 무작위 모델 15종 + 파운데이션
# 7종. 파운데이션은 zero-shot이라 재실행하지 않고 27번 저장 예측의 π만 28번 π로 바꾼다. 두 부분 결합은
# p = p_raw·√(c₊·(1−π))이고 c₊는 π와 무관하게(RV>0 시점만으로) 추정되므로, p₂₈ = p₂₇·√((1−π₂₈)/(1−π₂₇))가 정확하다.

# %%
FM_SRC = X.RES / "27_model_expansion_test_predictions.npz"


def _seed_file(seed: int, kind: str = "test_predictions.npz") -> Path:
    return RES / f"{STEM}_seed{seed}_{kind}"


def combine_seed(seed: int, tickers: list[str], quick: bool = False) -> dict:
    store: dict = {}
    rows: list[dict] = []
    metas: list[dict] = []
    fails: list[dict] = []
    for tk in tickers:
        c0 = pickle.loads(part_path("cpu", tk, 0, quick=quick).read_bytes())
        cs = c0 if seed == 0 else pickle.loads(part_path("cpu", tk, seed, quick=quick).read_bytes())
        fails += [dict(종목=tk, 시드=seed, 모델=f[0], H=f[1], 예외=f[2], 메시지=f[3]) for f in cs["fails"]]
        # 적합 품질 게이트(known_pitfalls P8): 최대우도 적합이 시작값에 멈춘 채 "수렴"으로 끝나면 결과를 쓰지 않는다
        for mt in (c0["meta"], cs["meta"]):
            if not any(k.startswith("garch_") and k.endswith("_alpha") for k in mt):
                raise RuntimeError(f"[적합 품질 게이트] {tk} 시드 {mt.get('시드')}: GARCH-t 진단이 없다(수정 전 CPU 작업 결과)")
            for k in [k for k in mt if k.startswith("garch_") and k.endswith("_alpha")]:
                be = mt[k.replace("_alpha", "_beta")]
                if abs(mt[k] - 0.1) < 1e-6 and abs(be - 0.88) < 1e-6:
                    raise RuntimeError(f"[적합 품질 게이트] {tk} GARCH-t가 arch 시작값(α=0.1, β=0.88)에 고착")
        if "tar_선택_수렴" not in c0["meta"]:
            raise RuntimeError(f"[적합 품질 게이트] {tk}: TAR-GARCH 수렴 진단이 없다(수정 전 CPU 작업 결과)")
        r0 = {R["H"]: R for R in c0["results"]}
        for R in cs["results"]:
            H = R["H"]
            S = store.setdefault((tk, H), {"preds": {}})
            S.update(act=R["act"], nai=R["nai"], T=R["T"], nai_tr=R["nai_tr"], pi=R["pi"])
            for nm, p in r0[H]["preds"].items():
                if nm in CPU_DET or nm == "naive":
                    S["preds"][nm] = p
            rows += [r for r in r0[H]["rows"] if r["모델"] in CPU_DET or r["모델"] == "naive"]
            if seed:
                g = R["preds"].get("GARCH-t")
                if g is None or float(np.max(np.abs(g - r0[H]["preds"]["GARCH-t"]))) > 1e-9:
                    raise RuntimeError(f"[결정론 게이트] {tk} H={H} 시드 {seed}의 GARCH-t가 시드 0과 다르다")
                if not (np.array_equal(R["T"], r0[H]["T"]) and np.array_equal(R["act"], r0[H]["act"])):
                    raise RuntimeError(f"[정렬 게이트] {tk} H={H} 시드 {seed}의 평가 시각·실제값이 시드 0과 다르다")
            for nm in CPU_STOCH:
                if nm in R["preds"]:
                    S["preds"][nm] = R["preds"][nm]
            rows += [r for r in R["rows"] if r["모델"] in CPU_STOCH]
        metas.append(cs["meta"])
        gb_f = PARTS / f"s0{'_quick' if quick else ''}" / "garchbic" / f"{tk}.pkl"
        if not gb_f.exists():
            raise RuntimeError(f"[시드 완전성 게이트] {tk}: GARCH-BIC 저장분이 없다(--family garchbic를 먼저 실행)")
        gb = pickle.loads(gb_f.read_bytes())
        for R in gb["results"]:
            if not np.array_equal(R["T"], store[(tk, R["H"])]["T"]):
                raise RuntimeError(f"[정렬 게이트] {tk} H={R['H']} GARCH-BIC 평가 시각이 다르다")
            store[(tk, R["H"])]["preds"][GARCH_BIC] = R["preds"][GARCH_BIC]
            rows += R["rows"]
        for H in M.HORIZONS_H:
            for nm in GPU_MODELS:
                f = part_path("gpu", tk, seed, H, nm, quick)
                if not f.exists():
                    continue
                g = pickle.loads(f.read_bytes())
                p = g["preds"][nm]
                if len(p) != len(store[(tk, H)]["act"]):
                    raise AssertionError(f"{tk} H={H} {nm} 예측 길이 불일치")
                store[(tk, H)]["preds"][nm] = p
                rows += g["rows"]
                metas.append(g["meta"])
    # 파운데이션: π 교체
    if FM_SRC.exists() and not quick:
        z = np.load(FM_SRC)
        for (tk, H), S in store.items():
            pi27 = z[f"{tk}|{H}|정지확률"].astype(float)
            act27 = z[f"{tk}|{H}|실제"].astype(float)
            if not np.allclose(act27, S["act"].astype(float), rtol=0, atol=1e-7):
                raise RuntimeError(f"[정렬 게이트] {tk} H={H} 27번 실제값과 28번 실제값이 다르다")
            pi28 = S["pi"].astype(float)
            nai_f = S["preds"]["naive"].astype(float)      # naive 예측 = Scorer가 필터를 적용한 직전 RV
            for nm in FM_MODELS:
                k = f"{tk}|{H}|{nm}"
                if k not in z.files:
                    continue
                p = z[k].astype(float) * np.sqrt((1 - pi28) / (1 - pi27))
                S["preds"][nm] = p.astype(np.float32)
                rows.append({"종목": tk, "H": H, "모델": nm, "계열": FAMILY[nm], "처리방식": PROCESS.get(nm, ""),
                             "n": len(p), "비고": "zero-shot · 27번 예측에 28번 π 적용", "정지처리": "두 부분",
                             **M.evaluate(S["act"].astype(float), p, nai_f)})
    rd = pd.DataFrame(rows)
    return dict(store=store, rd=rd, meta=pd.DataFrame(metas), fails=pd.DataFrame(fails))


def check_complete(store: dict, seed: int, tickers: list[str]) -> list[str]:
    """시드 완전성: 모든 (종목, 구간)에 시드 모델 15종 + 결정론 6종(naive 포함)이 있어야 한다."""
    miss = []
    for tk in tickers:
        for H in M.HORIZONS_H:
            have = set(store.get((tk, H), {}).get("preds", {}))
            for nm in STOCHASTIC + CPU_DET + (GARCH_BIC, "naive"):
                if nm not in have:
                    miss.append(f"{tk}|{H}|{nm}")
    return miss


def save_seed(seed: int, tickers: list[str], quick: bool = False) -> None:
    out = combine_seed(seed, tickers, quick)
    miss = check_complete(out["store"], seed, tickers)
    if miss:
        raise RuntimeError(f"[시드 완전성 게이트] 시드 {seed}: {len(miss)}칸 없음, 예: {miss[:6]}")
    M.save_npz(_seed_file(seed), out["store"], aux=True)
    out["rd"].to_csv(_seed_file(seed, "model_comparison.csv"), index=False)
    mt = out["meta"]
    gpu = mt["모델"].notna() if "모델" in mt else pd.Series(False, index=mt.index)
    mt[~gpu].dropna(axis=1, how="all").to_csv(_seed_file(seed, "cpu_meta.csv"), index=False)
    mt[gpu].dropna(axis=1, how="all").to_csv(_seed_file(seed, "gpu_meta.csv"), index=False)
    out["fails"].to_csv(_seed_file(seed, "fit_failures.csv"), index=False)
    print(f"[묶기] 시드 {seed}: 칸 {len(out['store'])} · 행 {len(out['rd'])} · 실패 {len(out['fails'])}", flush=True)


def load_seed_stores(seeds) -> dict:
    out = {}
    for s in seeds:
        f = _seed_file(s)
        if not f.exists():
            raise RuntimeError(f"[시드 완전성 게이트] 시드 {s} 묶음 파일이 없다: {f.name}")
        out[s] = M._npz_to_store(f)
    return out


# %% [markdown]
# ## 시드 오차 비율 R과 시드 수 결정(설계 절의 규칙 그대로)

# %%
def hac_mean_test(d: np.ndarray) -> tuple[float, float, float]:
    """26b와 같은 Newey-West HAC(Bartlett, 시차 ⌊4(n/100)^{2/9}⌋). 반환: (평균, 표준오차, 양측 p)."""
    import scipy.stats as st
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 20:
        return np.nan, np.nan, np.nan
    mu = d.mean()
    e = d - mu
    lag = int(np.floor(4 * (n / 100) ** (2 / 9)))
    v = e @ e / n
    for k in range(1, lag + 1):
        v += 2 * (1 - k / (lag + 1)) * (e[k:] @ e[:-k]) / n
    se = np.sqrt(max(v, 1e-30) / n)
    return float(mu), float(se), float(2 * st.t.sf(abs(mu / se), df=n - 1))


def holm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * p[i])
        adj[i] = min(run, 1.0)
    return adj


def panel_losses(stores: dict, H: int, models) -> dict:
    """시드 → 모델 → 종목 평균 QLIKE 시계열(시각 인덱스). 종목마다 평가 시각이 같다(26c 공통 분할·정시 평가)."""
    out = {}
    for s, st_ in stores.items():
        out[s] = {}
        for nm in models:
            cols = {}
            for (tk, h), S in st_.items():
                if h != H or nm not in S["preds"]:
                    continue
                T = pd.DatetimeIndex(np.asarray(S["T"]).astype("datetime64[ns]"))
                cols[tk] = pd.Series(M.qlike_vec(S["act"].astype(float), S["preds"][nm].astype(float)), index=T)
            if cols:
                out[s][nm] = pd.DataFrame(cols).sort_index()
    return out


def seed_error_table(stores: dict, pairs_a, pairs_b) -> pd.DataFrame:
    """쌍마다 σ²_시드(시드별 평균 손실차의 분산), SE_시간(시드 평균 손실차 시계열의 HAC), 원시 R."""
    seeds = sorted(stores)
    rows = []
    for H in M.HORIZONS_H:
        L = panel_losses(stores, H, tuple(pairs_a) + tuple(pairs_b))
        for a in pairs_a:
            for b in pairs_b:
                if any(a not in L[s] or b not in L[s] for s in seeds):
                    continue
                ds = [(L[s][a] - L[s][b]).mean(axis=1) for s in seeds]
                dbar = np.array([d.mean() for d in ds])
                dm = sum(ds) / len(ds)
                mu, se, p = hac_mean_test(dm.to_numpy())
                s2 = float(np.var(dbar, ddof=1))
                rows.append({"H": H, "A": a, "B": b, "평균차": mu, "SE_시간": se, "σ²_시드": s2,
                             "R_원시": s2 / se ** 2, "시드수": len(seeds), "시각수": int(dm.notna().sum())})
    return pd.DataFrame(rows)


def eb_shrink_R(t: pd.DataFrame) -> pd.DataFrame:
    """log s²에 경험적 베이즈 축소. s²(자유도 k)의 log는 평균 log σ² + ψ(k/2) − log(k/2), 분산 ψ'(k/2)."""
    from scipy.special import digamma, polygamma
    k = int(t["시드수"].iloc[0]) - 1
    bias, var_e = digamma(k / 2) - np.log(k / 2), float(polygamma(1, k / 2))
    z = np.log(t["σ²_시드"].clip(lower=1e-30)) - bias
    mu = float(z.mean())
    tau2 = max(float(z.var(ddof=1)) - var_e, 0.0)
    w = tau2 / (tau2 + var_e) if tau2 + var_e > 0 else 0.0
    t = t.copy()
    t["σ²_시드_축소"] = np.exp(mu + w * (z - mu))
    t["R"] = t["σ²_시드_축소"] / t["SE_시간"] ** 2
    t.attrs.update(축소가중=w, 사전평균=mu, 사전분산=tau2, 측정분산=var_e)
    return t


def decide_seeds(R90: float) -> tuple[int, str]:
    for S in S_CANDIDATES:
        if np.sqrt(1 + R90 / S) <= INFL_MAX:
            return S, f"√(1+R90/S) = {np.sqrt(1 + R90 / S):.3f} ≤ {INFL_MAX}을 만족하는 가장 작은 후보"
    S = S_CANDIDATES[-1]
    return S, f"후보 최대 {S}에서도 √(1+R90/S) = {np.sqrt(1 + R90 / S):.3f} > {INFL_MAX} — 50으로 하고 초과 쌍을 보고"


def fig_R(t: pd.DataFrame, R90: float, S: int, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    for H, mk in zip(M.HORIZONS_H, "osD^v"):
        g = t[t["H"] == H].sort_values("R")
        ax.scatter(g["R_원시"], g["R"], s=18, marker=mk, label=hlabel(H), alpha=.8)
    lim = [max(1e-3, min(t["R_원시"].min(), t["R"].min()) * .8), max(t["R_원시"].max(), t["R"].max()) * 1.2]
    ax.plot(lim, lim, color="#999", lw=.8, ls=":")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.axhline(R90, color="#c0392b", lw=1, ls="--")
    ax.text(lim[0] * 1.1, R90 * 1.08, f"R90 = {R90:.2f}", color="#c0392b", fontsize=9)
    ax.set_xlabel("원시 R(시드 5개)"); ax.set_ylabel("축소한 R")
    ax.set_title("쌍별 시드 오차 비율 R", fontsize=10)
    ax.legend(fontsize=8, title="예측 구간", title_fontsize=8)
    ax = axes[1]
    Ss = np.arange(5, 81)
    for q, lab, col in ((t["R"].median(), "중앙 쌍", "#2a78d6"), (R90, "90분위 쌍", "#c0392b"),
                        (t["R"].max(), "최대 쌍", "#8a6b2e")):
        ax.plot(Ss, np.sqrt(1 + q / Ss), color=col, label=f"{lab}(R={q:.2f})")
    ax.axhline(INFL_MAX, color="#555", ls=":", lw=1)
    for c in S_CANDIDATES:
        ax.axvline(c, color="#ddd", lw=.8)
    ax.axvline(S, color="#0b0b0b", lw=1.2)
    ax.text(S + 1, ax.get_ylim()[1] * .98 if ax.get_ylim()[1] > 1.2 else 1.2, f"채택 S = {S}", va="top", fontsize=9)
    ax.set_xlabel("시드 수 S"); ax.set_ylabel("표준오차 배율 √(1+R/S)")
    ax.set_title("시드 수별 표준오차 증가", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def measure_and_decide(tickers: list[str]) -> dict:
    stores = load_seed_stores(SEEDS_PILOT)
    t = eb_shrink_R(seed_error_table(stores, R_PAIRS_A, R_PAIRS_B))
    R90 = float(t["R"].quantile(.9))
    S, why = decide_seeds(R90)
    t["배율_S"] = np.sqrt(1 + t["R"] / S)
    t.to_csv(RES / f"{STEM}_seed_error_ratio.csv", index=False)
    allp = eb_shrink_R(seed_error_table(stores, STOCHASTIC, STOCHASTIC + CPU_DET))
    allp = allp[allp["A"] != allp["B"]]
    allp.to_csv(RES / f"{STEM}_seed_error_ratio_allpairs.csv", index=False)
    fig_R(t, R90, S, IMG / f"{STEM}_fig2_seed_error_ratio.png")
    dec = {"R_중앙": float(t["R"].median()), "R90": R90, "R_최대": float(t["R"].max()), "S": S, "근거": why,
           "쌍수": len(t), "축소가중": t.attrs["축소가중"], "R_원시_중앙": float(t["R_원시"].median()),
           "R_원시_90": float(t["R_원시"].quantile(.9)), "결정시각": time.strftime("%F %T"),
           "27번_참고": {"R_중앙": 1.62, "R90": 6.28}}
    (RES / f"{STEM}_seed_decision.json").write_text(json.dumps(dec, ensure_ascii=False, indent=2))
    return dec


# %% [markdown]
# ## 최종 분석: 시드 항을 넣은 검정
#
# 손실은 QLIKE, 시각마다 범위(전체 또는 국면 Q1~Q5)에 든 종목의 평균을 내 시계열을 만든다(27d와 같은 시각 동일 가중).
# 국면은 직전 H분 RV의 종목별 학습 구간 5분위다(예측 시점에 아는 값). 무작위 모델은 시드마다 손실을 구해 평균한다.
# 모든 DM 검정의 표준오차는 SE_전체 = √(SE²_시간 + σ²_시드/S)다. SE_시간은 시드 평균 손실차 시계열의 HAC,
# σ²_시드는 시드별 평균 손실차(시각 평균)의 시드 간 분산이다. 결정론 모델끼리는 σ²_시드 = 0이다.

# %%
SEQ = ("GARCH-t", "GARCH-BIC", "MS-GARCH", "TAR-GARCH", "GRU", "LSTM")
PAR_FEAT = ("LightGBM", "XGBoost", "HistGBM", "GARCH+LightGBM", "Nystroem+Ridge")   # 커널 2종 제외(DECISION_EXCLUDED)
PAR_DEEP = BLOCK_MODELS + tuple(FM_MODELS)
GROUP_OF = {**{m: "순차" for m in SEQ}, **{m: "병렬(특징 기반)" for m in PAR_FEAT}, **{m: "병렬(딥러닝)" for m in PAR_DEEP}}
GROUP_COLOR = {"순차": "#2a78d6", "병렬(특징 기반)": "#eb6834", "병렬(딥러닝)": "#8a6b2e", "기준선": "#888888"}
COMPARISONS = (("순차 − 병렬 전체", SEQ, PAR_FEAT + PAR_DEEP), ("순차 − 병렬(특징 기반)", SEQ, PAR_FEAT),
               ("순차 − 병렬(딥러닝)", SEQ, PAR_DEEP), ("순환 신경망 − 병렬(딥러닝)", ("GRU", "LSTM"), PAR_DEEP))
SCOPES = ("전체", "Q1", "Q2", "Q3", "Q4", "Q5")
ALPHA = 0.05
SHORT = {**M.SHORT, **(X.SHORT if hasattr(X, "SHORT") else {}), "GARCH-BIC": "G-BIC"}


def sname(m: str) -> str:
    return SHORT.get(m, m)


def cube(stores: dict, H: int) -> tuple[dict, pd.DatetimeIndex, dict]:
    """반환: 모델 → 배열[시드, 범위, 시각](범위별 종목 평균 QLIKE), 시각, 범위별 관측 수."""
    seeds = sorted(stores)
    s0 = stores[seeds[0]]
    tks = sorted(k[0] for k in s0 if k[1] == H)
    T = pd.DatetimeIndex(np.asarray(s0[(tks[0], H)]["T"]).astype("datetime64[ns]"))
    act = np.column_stack([s0[(tk, H)]["act"].astype(float) for tk in tks])
    for tk in tks:
        if not np.array_equal(np.asarray(s0[(tk, H)]["T"]), np.asarray(s0[(tks[0], H)]["T"])):
            raise RuntimeError(f"[정렬 게이트] {tk} H={H}의 평가 시각이 다른 종목과 다르다")
    Q = np.column_stack([np.digitize(s0[(tk, H)]["nai"], np.quantile(s0[(tk, H)]["nai_tr"], [.2, .4, .6, .8]))
                         for tk in tks])
    masks = np.stack([np.ones_like(Q, bool)] + [Q == q for q in range(5)])          # [범위, 시각, 종목]
    cnt = masks.sum(axis=2)
    models = [m for m in ALL_MODELS if all(m in s0[(tk, H)]["preds"] for tk in tks)]
    out = {}
    for nm in models:
        per = []
        for s in (seeds if nm in STOCHASTIC else seeds[:1]):
            P = np.column_stack([stores[s][(tk, H)]["preds"][nm].astype(float) for tk in tks])
            L = M.qlike_vec(act.ravel(), P.ravel()).reshape(act.shape)
            with np.errstate(invalid="ignore", divide="ignore"):
                per.append((masks * L[None]).sum(axis=2) / np.where(cnt > 0, cnt, np.nan))
        out[nm] = np.stack(per)                                                       # [시드 또는 1, 범위, 시각]
    return out, T, {sc: int(masks[i].sum()) for i, sc in enumerate(SCOPES)}


def dm_seed(dA: np.ndarray, S_used: int) -> dict:
    """dA: [시드, 시각] 손실차(시드 1개면 결정론). 반환: 평균차, SE_시간, σ²_시드, SE_전체, p(정규 근사), 배율."""
    dm = np.nanmean(dA, axis=0)
    ok = np.isfinite(dm)
    mu, se_t, _ = hac_mean_test(dm[ok])
    s2 = float(np.var(np.nanmean(dA[:, ok], axis=1), ddof=1)) if dA.shape[0] > 1 else 0.0
    se = float(np.sqrt(se_t ** 2 + s2 / max(S_used, 1)))
    import scipy.stats as st
    p = float(2 * st.norm.sf(abs(mu / se))) if se > 0 else np.nan
    return dict(평균차=mu, SE_시간=se_t, σ2_시드=s2, SE_전체=se, p=p, 배율=se / se_t if se_t > 0 else np.nan,
                시각수=int(ok.sum()))


def _diff(C: dict, a, b, si: int) -> np.ndarray:
    A, B = C[a][:, si], C[b][:, si]
    n = max(A.shape[0], B.shape[0])
    return np.broadcast_to(A, (n,) + A.shape[1:]) - np.broadcast_to(B, (n,) + B.shape[1:])


def _group(C: dict, members, si: int) -> np.ndarray:
    have = [m for m in members if m in C]
    n = max(C[m].shape[0] for m in have)
    return sum(np.broadcast_to(C[m][:, si], (n, C[m].shape[2])) for m in have) / len(have)


def analyze(stores: dict, label: str) -> dict:
    """칸별 평균·최선·동률, 순차 대 병렬 묶음 검정. stores = {시드: store}."""
    S_used = len(stores)
    cells, ties, sp = [], [], []
    for H in M.HORIZONS_H:
        C, T, nobs = cube(stores, H)
        for si, sc in enumerate(SCOPES):
            means = {m: float(np.nanmean(np.nanmean(C[m][:, si], axis=0))) for m in C}
            sds = {m: float(np.std(np.nanmean(C[m][:, si], axis=1), ddof=1)) if C[m].shape[0] > 1 else 0.0 for m in C}
            best = min((m for m in means if m != "naive"), key=means.get)
            rows = []
            for m in C:
                r = {"H": H, "범위": sc, "모델": m, "묶음": GROUP_OF.get(m, "기준선"), "평균QLIKE": means[m],
                     "시드SD": sds[m], "최선": best, "관측수": nobs[sc]}
                if m != best:
                    r.update(dm_seed(_diff(C, m, best, si), S_used))
                else:
                    r.update(평균차=0.0, p=1.0)
                rows.append(r)
            t = pd.DataFrame(rows)
            nb = t["모델"] != best
            t.loc[nb, "p_holm"] = holm(t.loc[nb, "p"].fillna(1).to_numpy())
            t.loc[~nb, "p_holm"] = 1.0
            t["동률"] = t["p_holm"] > ALPHA
            t["순위"] = t["평균QLIKE"].rank().astype(int)
            ties.append(t)
            for name, A, B in COMPARISONS:
                gA, gB = _group(C, A, si), _group(C, B, si)
                n = max(gA.shape[0], gB.shape[0])
                d = np.broadcast_to(gA, (n, gA.shape[1])) - np.broadcast_to(gB, (n, gB.shape[1]))
                sp.append({"비교": name, "H": H, "범위": sc, **dm_seed(d, S_used),
                           "A구성원": len([m for m in A if m in C]), "B구성원": len([m for m in B if m in C])})
        print(f"  [{label}] {hlabel(H)} 분석 완료", flush=True)
    tt = pd.concat(ties, ignore_index=True)
    spt = pd.DataFrame(sp)
    out = []
    for name, g in spt.groupby("비교", sort=False):
        g = g.copy()
        g["p_holm"] = holm(g["p"].fillna(1).to_numpy())
        g["판정"] = np.where(g["p_holm"] >= ALPHA, "구분 안 됨", np.where(g["평균차"] < 0, "순차 우세", "병렬 우세"))
        out.append(g)
    return dict(ties=tt, seqpar=pd.concat(out, ignore_index=True), S=S_used)


def stores27() -> dict:
    """27번 시드 0~4 저장 예측을 같은 형식으로 읽는다(시드 s = 시드 0 + 무작위 모델만 시드 s 예측)."""
    base = M._npz_to_store(X.RES / "27_model_expansion_test_predictions.npz")
    out = {0: base}
    for s in (1, 2, 3, 4):
        z = M._npz_to_store(X.RES / f"27_model_expansion_seed{s}_test_predictions.npz")
        st_ = {}
        for k, S in base.items():
            st_[k] = {**S, "preds": {**S["preds"], **{m: p for m, p in z.get(k, {}).get("preds", {}).items()
                                                     if m in X.STOCHASTIC_ALL}}}
        out[s] = st_
    return out


def gpu_meta_all(seeds) -> pd.DataFrame:
    fr = [pd.read_csv(_seed_file(s, "gpu_meta.csv")) for s in seeds if _seed_file(s, "gpu_meta.csv").exists()]
    return pd.concat(fr, ignore_index=True) if fr else pd.DataFrame()


def fig_procedure(gm: pd.DataFrame, path: Path) -> None:
    g0 = gm[gm["시드"] == 0].copy()
    g0["선택"] = [("하단(×0.25)", "중심(×1)", "상단(×4)")[int(np.argmin([abs(np.log(lr / (DEFAULT_LR[m] * k))) for k in LR_MULT]))]
                for m, lr in zip(g0["모델"], g0["lr"])]
    share = g0.groupby(["모델", "선택"]).size().unstack(fill_value=0)
    share = share.reindex(columns=["하단(×0.25)", "중심(×1)", "상단(×4)"], fill_value=0)
    share = share.div(share.sum(axis=1), axis=0).reindex([m for m in GPU_MODELS if m in share.index])
    cap = gm[gm["상한도달"].isin([True, False])].groupby("모델")["상한도달"].mean().reindex(share.index)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    left = np.zeros(len(share))
    for col, c in zip(share.columns, ("#9ec5f0", "#2a78d6", "#0b3d7a")):
        ax.barh(range(len(share)), share[col], left=left, color=c, label=col)
        left += share[col].to_numpy()
    ax.set_yticks(range(len(share))); ax.set_yticklabels([sname(m) for m in share.index])
    ax.invert_yaxis(); ax.set_xlabel("비율(시드 0, 종목 × 구간)"); ax.set_title("고른 학습률 위치", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax = axes[1]
    ax.barh(range(len(cap)), cap.fillna(0).to_numpy(), color="#eb6834")
    for i, v in enumerate(cap.to_numpy()):
        ax.text((0 if not np.isfinite(v) else v) + 0.005, i, "해당 없음(기록 안 함)" if not np.isfinite(v) else f"{v:.1%}",
                va="center", fontsize=8)
    ax.set_yticks(range(len(cap))); ax.set_yticklabels([sname(m) for m in cap.index]); ax.invert_yaxis()
    ax.set_xlim(0, max(0.2, float(np.nanmax(cap.fillna(0))) * 1.4)); ax.set_xlabel("학습 횟수 상한 도달 비율(전 시드)")
    ax.set_title("상한 도달", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


def fig_ranking(tt: pd.DataFrame, path: Path) -> None:
    hs = list(M.HORIZONS_H)
    fig, axes = plt.subplots(1, len(hs), figsize=(4.2 * len(hs), 7.2), sharey=False)
    for ax, H in zip(axes, hs):
        g = tt[(tt["H"] == H) & (tt["범위"] == "전체") & (tt["모델"] != "naive")].sort_values("평균QLIKE")
        gap = g["평균QLIKE"] - g["평균QLIKE"].min()
        y = np.arange(len(g))
        for yi, (_, r), gp in zip(y, g.iterrows(), gap):
            c = GROUP_COLOR[r["묶음"]]
            ax.errorbar(gp, yi, xerr=r["시드SD"], fmt="o" if r["동률"] else "x", color=c, ms=5, capsize=2)
        ax.set_yticks(y); ax.set_yticklabels([sname(m) for m in g["모델"]], fontsize=7); ax.invert_yaxis()
        ax.set_xscale("symlog", linthresh=0.01); ax.set_title(hlabel(H), fontsize=10)
        ax.set_xlabel("1위와의 QLIKE 격차(시드 평균, 막대 = 시드 SD)", fontsize=7)
    handles = [plt.Line2D([], [], color=c, marker="o", ls="", label=k) for k, c in GROUP_COLOR.items() if k != "기준선"]
    handles += [plt.Line2D([], [], color="#333", marker="o", ls="", label="1위와 구분 안 됨"),
                plt.Line2D([], [], color="#333", marker="x", ls="", label="1위보다 유의하게 나쁨")]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig(path, dpi=300); plt.close(fig)


def fig_seqpar(sp: pd.DataFrame, path: Path) -> None:
    comps = [c[0] for c in COMPARISONS]
    rows = [(c, H) for c in comps for H in M.HORIZONS_H]
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(rows) + 1.2))
    for i, (c, H) in enumerate(rows):
        for j, sc in enumerate(SCOPES):
            r = sp[(sp["비교"] == c) & (sp["H"] == H) & (sp["범위"] == sc)].iloc[0]
            col = {"순차 우세": "#2a78d6", "병렬 우세": "#eb6834", "구분 안 됨": "#e5e5e0"}[r["판정"]]
            ax.add_patch(plt.Rectangle((j, i), 1, 1, color=col, ec="white"))
            ax.text(j + .5, i + .5, f"{r['평균차']:+.3f}", ha="center", va="center", fontsize=7,
                    color="white" if r["판정"] != "구분 안 됨" else "#333")
    ax.set_xlim(0, len(SCOPES)); ax.set_ylim(len(rows), 0)
    ax.set_xticks(np.arange(len(SCOPES)) + .5); ax.set_xticklabels(SCOPES)
    ax.set_yticks(np.arange(len(rows)) + .5); ax.set_yticklabels([f"{c} · {hlabel(H)}" for c, H in rows], fontsize=7)
    ax.set_xlabel("범위(Q1 잔잔 ~ Q5 요동, 직전 RV 기준)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#2a78d6", label="순차 우세"), Patch(color="#eb6834", label="병렬 우세"),
                       Patch(color="#e5e5e0", label="구분 안 됨")], loc="upper center", bbox_to_anchor=(.5, -0.05 * 30 / len(rows)),
              ncol=3, fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


def fig_regime(tt: pd.DataFrame, path: Path) -> None:
    hs = list(M.HORIZONS_H)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for i, H in enumerate(hs):
        for j, sc in enumerate(SCOPES):
            g = tt[(tt["H"] == H) & (tt["범위"] == sc)]
            b = g["최선"].iloc[0]
            nt = int(g["동률"].sum()) - 1
            ax.add_patch(plt.Rectangle((j, i), 1, 1, color=GROUP_COLOR[GROUP_OF.get(b, "기준선")], alpha=.85, ec="white"))
            ax.text(j + .5, i + .42, sname(b), ha="center", va="center", fontsize=8, color="white")
            ax.text(j + .5, i + .75, f"동률 {nt}", ha="center", va="center", fontsize=6.5, color="white")
    ax.set_xlim(0, len(SCOPES)); ax.set_ylim(len(hs), 0)
    ax.set_xticks(np.arange(len(SCOPES)) + .5); ax.set_xticklabels(SCOPES)
    ax.set_yticks(np.arange(len(hs)) + .5); ax.set_yticklabels([hlabel(H) for H in hs])
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, label=k) for k, c in GROUP_COLOR.items() if k != "기준선"],
              loc="upper center", bbox_to_anchor=(.5, -0.1), ncol=3, fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


def fig_vs27(tt: pd.DataFrame, t27: pd.DataFrame, path: Path) -> None:
    hs = list(M.HORIZONS_H)
    fig, axes = plt.subplots(1, len(hs), figsize=(4 * len(hs), 4.2))
    for ax, H in zip(axes, hs):
        a = tt[(tt["H"] == H) & (tt["범위"] == "전체")].set_index("모델")["순위"]
        b = t27[(t27["H"] == H) & (t27["범위"] == "전체")].set_index("모델")["순위"]
        k = a.index.intersection(b.index)
        for m in k:
            ax.scatter(b[m], a[m], color=GROUP_COLOR[GROUP_OF.get(m, "기준선")], s=22)
            if abs(a[m] - b[m]) >= 4:
                ax.annotate(sname(m), (b[m], a[m]), fontsize=7, xytext=(3, 3), textcoords="offset points")
        mx = max(a.max(), b.max()) + 1
        ax.plot([0, mx], [0, mx], color="#999", lw=.8, ls=":")
        ax.set_xlabel("27번 순위(시드 5개 평균)"); ax.set_ylabel("28번 순위") if H == hs[0] else None
        ax.set_title(hlabel(H), fontsize=10); ax.invert_xaxis(); ax.invert_yaxis()
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


GROUP_ORDER = ("순차", "병렬(특징 기반)", "병렬(딥러닝)")
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e5e5e0"
GROUP_LABEL = ["순차(GARCH·GRU·LSTM)", "병렬 특징 기반(트리·커널 근사)", "병렬 딥러닝(어텐션·합성곱·파운데이션)"]


def fig_regime_lines(tt: pd.DataFrame, path: Path, S: int) -> None:
    """포스터 그림 3 자리(27d fig1b와 같은 형식): 구간마다 처리 방식 묶음별 가장 나은 모델의 1위 대비 격차를 국면 순서로 잇는다."""
    Hs = list(M.HORIZONS_H)
    fig, axes = plt.subplots(1, len(Hs), figsize=(19, 5.6))
    x = np.arange(len(SCOPES) - 1)
    for ax, H in zip(axes, Hs):
        for grp in GROUP_ORDER:
            ys, tie = [], []
            for sc in SCOPES[1:]:
                g = tt[(tt["H"] == H) & (tt["범위"] == sc) & tt["모델"].map(GROUP_OF).eq(grp)]
                ys.append(max(float(g["평균차"].min()), 0.0))
                tie.append(bool(g["동률"].any()))
            ax.plot(x, ys, color=GROUP_COLOR[grp], lw=2.4, zorder=2)
            for xi, yi, ti in zip(x, ys, tie):
                ax.scatter(xi, yi, s=70, zorder=3, color=GROUP_COLOR[grp] if ti else "white",
                           edgecolor=GROUP_COLOR[grp], linewidth=2)
        ax.axhline(0, color=MUTED, lw=1)
        ax.set_xticks(x, ["Q1\n잔잔", "Q2", "Q3", "Q4", "Q5\n요동"])
        ax.set_title(hlabel(H), fontsize=13, loc="left", color=INK, fontweight="bold")
        ax.grid(axis="y", color=GRID)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    fig.supylabel(f"1위 모델 대비 QLIKE 격차(0 = 그 묶음의 모델이 1위, 시드 {S}개 평균)", fontsize=11.5, x=0.0)
    handles = [plt.Line2D([], [], color=GROUP_COLOR[g_], lw=2.4, marker="o", markersize=8, label=lb)
               for g_, lb in zip(GROUP_ORDER, GROUP_LABEL)]
    handles += [plt.Line2D([], [], color=MUTED, lw=0, marker="o", markersize=8,
                           label="채운 점 = 그 묶음에 1위와 구분 안 되는 모델이 있음(시드 항 포함 검정, Holm p ≥ 0.05)"),
                plt.Line2D([], [], color=MUTED, lw=0, marker="o", markersize=8, markerfacecolor="white",
                           label="빈 점 = 그 묶음 모델이 모두 1위보다 유의하게 나쁨")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=10.5, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_top5(tt: pd.DataFrame, path: Path, S: int) -> None:
    """포스터 그림 2 자리(27d fig5와 같은 형식): 예측 구간별 상위 5개(위 전체 기간, 아래 가장 요동 Q5)."""
    Hs = list(M.HORIZONS_H)
    rows = (("전체", "전체 기간"), ("Q5", "가장 요동칠 때(Q5)"))
    fig, axes = plt.subplots(len(rows), len(Hs), figsize=(25, 6.3))
    for r, (sc, rlab) in enumerate(rows):
        for c, H in enumerate(Hs):
            ax = axes[r, c]
            g = tt[(tt["H"] == H) & (tt["범위"] == sc) & (tt["모델"] != "naive")].sort_values("평균QLIKE").reset_index(drop=True)
            g["등수"] = np.arange(1, len(g) + 1)
            g["격차"] = g["평균QLIKE"] - g["평균QLIKE"].iloc[0]
            deep = g[g["모델"].map(GROUP_OF).eq("병렬(딥러닝)")].head(1)
            extra = deep if len(deep) and int(deep["등수"].iloc[0]) > 5 else deep.iloc[0:0]
            items = list(g.head(5).itertuples()) + list(extra.itertuples())
            ys = np.arange(len(items))[::-1].astype(float)
            if len(extra):
                ys[-1] -= 0.6
                ax.axhline(ys[-1] + 0.8, color=MUTED, lw=0.8, ls=(0, (3, 3)))
            xmax = max(float(it.격차) for it in items) * 1.45 or 1e-3
            for y, it in zip(ys, items):
                col = GROUP_COLOR[GROUP_OF[it.모델]]
                v = max(float(it.격차), 0.0)
                ax.plot([0, v], [y, y], color=col, lw=2.2, alpha=0.6, zorder=1)
                ax.scatter(v, y, s=80, zorder=3, color=col if it.동률 else "white", edgecolor=col, linewidth=2.2)
                lab = "1위" if it.등수 == 1 else (f"+{v:.4f}" if v < 0.001 else f"+{v:.3f}")
                ax.text(v + xmax * 0.05, y, lab, va="center", fontsize=10.5, color=INK)
            ax.set_yticks(ys, [f"{it.등수}. {it.모델}" for it in items], fontsize=11.5)
            for tl, it in zip(ax.get_yticklabels(), items):
                tl.set_color(GROUP_COLOR[GROUP_OF[it.모델]])
            ax.set_xlim(-xmax * 0.03, xmax)
            ax.set_ylim(min(ys) - 0.6, max(ys) + 0.6)
            ax.grid(axis="x", color=GRID)
            ax.tick_params(axis="y", length=0)
            ax.tick_params(axis="x", labelsize=9.5)
            for s_ in ("top", "right", "left"):
                ax.spines[s_].set_visible(False)
            if r == 0:
                ax.set_title(hlabel(H), fontsize=15, loc="left", color=INK, fontweight="bold", pad=12)
            if c == 0:
                ax.annotate(rlab, xy=(-0.78, 0.5), xycoords="axes fraction", rotation=90, va="center", ha="center",
                            fontsize=14, fontweight="bold", color=INK)
    fig.supxlabel(f"1위 모델 대비 QLIKE 격차(0 = 1위, 작을수록 좋음, 시드 {S}개 평균, 칸마다 가로축 다름)", fontsize=12.5)
    handles = [plt.Line2D([], [], color=GROUP_COLOR[g_], lw=0, marker="o", markersize=10, label=lb)
               for g_, lb in zip(GROUP_ORDER, GROUP_LABEL)]
    handles += [plt.Line2D([], [], color=MUTED, lw=0, marker="o", markersize=10, label="채운 점 = 1위와 통계적으로 구분 안 됨(시드 항 포함, Holm p ≥ 0.05)"),
                plt.Line2D([], [], color=MUTED, lw=0, marker="o", markersize=10, markerfacecolor="white", label="빈 점 = 1위보다 유의하게 나쁨"),
                plt.Line2D([], [], color=MUTED, lw=0.8, ls=(0, (3, 3)), label="점선 아래 = 5위 밖 병렬 딥러닝 중 가장 나은 모델")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=12, bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout(rect=(0.015, 0.0, 1, 1), h_pad=2.5, w_pad=5.5)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def data_target_check(s0: dict, path: Path) -> str:
    """0절: 타깃(RV)과 이번 회차에 바뀐 정지 확률 π를 점검한다(AGENTS.md 2.9.3 ①②를 이번 변경분에 다시 적용).
    타깃·손실 정의는 26c와 같고, 바뀐 것은 π(전체 재적합 분류기 → 내부 분류기)뿐이라 π가 실제 정지 비율을 얼마나 따라가는지 본다."""
    from report_header import describe_markdown
    z27 = np.load(FM_SRC) if FM_SRC.exists() else None
    rows, rv_desc, pi_desc = [], {}, {}
    for H in M.HORIZONS_H:
        acts, pis = [], []
        for (tk, h), S in sorted(s0.items()):
            if h != H:
                continue
            a, pi = S["act"].astype(float), S["pi"].astype(float)
            p27 = z27[f"{tk}|{H}|정지확률"].astype(float) if z27 is not None and f"{tk}|{H}|정지확률" in z27.files else np.full(len(a), np.nan)
            rows.append({"H": H, "종목": tk, "실제정지": float(np.mean(a == 0)), "π28": float(pi.mean()), "π27": float(np.nanmean(p27)),
                         "평가시각": len(a)})
            acts.append(a); pis.append(pi)
        rv_desc[hlabel(H)] = pd.Series(np.concatenate(acts) * 100).describe()
        pi_desc[hlabel(H)] = pd.Series(np.concatenate(pis)).describe()
    d = pd.DataFrame(rows)
    d.to_csv(RES / f"{STEM}_data_target_check.csv", index=False)
    fig, axes = plt.subplots(1, len(M.HORIZONS_H), figsize=(19, 4.2))
    for ax, H in zip(axes, M.HORIZONS_H):
        g = d[d["H"] == H]
        mx = max(0.02, float(np.nanmax(g[["실제정지", "π28", "π27"]].to_numpy())) * 1.1)
        ax.plot([0, mx], [0, mx], color=MUTED, lw=0.8, ls=":")
        ax.scatter(g["실제정지"], g["π27"], s=30, facecolor="white", edgecolor="#8a6b2e", label="27번 π(전체 재적합 분류기)")
        ax.scatter(g["실제정지"], g["π28"], s=30, color="#2a78d6", label="28번 π(내부 분류기, 무재적합)")
        ax.set_xlim(0, mx); ax.set_ylim(0, mx)
        ax.set_title(hlabel(H), fontsize=12, loc="left", fontweight="bold")
        ax.set_xlabel("평가 기간 실제 정지 비율(RV = 0)")
    axes[0].set_ylabel("평가 기간 평균 정지 확률 π")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)
    m = d.groupby("H")[["실제정지", "π28", "π27"]].mean()
    L = ["- **이번 회차에 바뀐 것은 정지 확률 π뿐이다.** 타깃(다음 H분 실현변동성 RV)·손실(QLIKE)·두 부분 모형 정의는 26c와 같다. "
         "26c·27은 π를 학습 구간 전체로 재적합한 분류기로 냈고, 28번은 내부학습 구간 분류기 하나로 보정용·평가용 π를 모두 낸다(random_state 0, 모든 모델·시드 공통).",
         "- **점검 ① 타깃 분포**: RV = 0(가격 정지)은 관측 실패가 아니라 실제로 가격이 한 칸도 움직이지 않은 구간이다(26c 2.9.3 점검). "
         "QLIKE는 실제값 0에서도 유한하고, 예측값이 0이면 발산하므로 π 상한 0.99와 극단 예측 필터를 그대로 둔다(②).",
         "", "| 구간 | 실제 정지 비율(종목 평균) | 28번 π 평균 | 27번 π 평균 | 28번 π − 실제 |", "| :--- | ---: | ---: | ---: | ---: |"]
    for H, r in m.iterrows():
        L.append(f"| {hlabel(H)} | {r['실제정지']:.4f} | {r['π28']:.4f} | "
                 + ("해당 없음(27번 파일 없음)" if not np.isfinite(r['π27']) else f"{r['π27']:.4f}") + f" | {r['π28'] - r['실제정지']:+.4f} |")
    under = d[d["π28"] < d["실제정지"]].groupby("H").size()
    skip = d[(d["π28"] == 0) & (d["실제정지"] > 0)]
    L += ["", "- **π 과소예측(26c부터 이어진 한계, 이번 회차 버그 아님)**: 평가 기간 실제 정지 비율보다 π가 낮은 종목 수 = "
          + ", ".join(f"{hlabel(H)} {int(under.get(H, 0))}/{int((d['H'] == H).sum())}" for H in M.HORIZONS_H)
          + ". 정지 비율이 학습 기간보다 평가 기간에 크게 늘어난 탓이다(26b EDA: 15분 내부검증 11.3% → 평가 26.7%).",
          "- **π = 0인데 실제 정지가 있는 칸(내부학습 정지 표본이 부족해 분류기를 생략)**: "
          + (", ".join(f"{hlabel(r.H)} {r.종목}(실제 {r.실제정지:.3f})" for r in skip.itertuples()) or "없음")
          + ". 이 칸의 두 부분 모형은 사실상 크기 모형만 쓴다. 모든 로그 타깃 모델이 같은 π를 쓰므로 모델 간 비교에는 공통으로 작용한다."]
    L += ["", f"![데이터·타깃 점검]({os.path.relpath(path, RES)})", "",
          "- **그림 읽는 법**: 점 하나가 종목 하나다. 가로 = 평가 기간 실제 정지 비율, 세로 = 평균 π. 대각선 위면 정지를 많게, 아래면 적게 예측한다. "
          "파랑(28번)과 갈색 빈 점(27번)의 차이가 무재적합으로 바뀐 몫이다. 4·12시간은 정지가 거의 없어 원점에 모인다.",
          "", "**타깃 RV(%) describe(평가 시점, 20종목 합침)**", "",
          describe_markdown(pd.DataFrame(rv_desc).T, "구간"), "",
          "**28번 정지 확률 π describe(평가 시점, 20종목 합침)**", "",
          describe_markdown(pd.DataFrame(pi_desc).T, "구간")]
    return "\n".join(L)


def fig_calibration(seeds, path: Path) -> None:
    """시드 0의 모델별 보정 계수 c₊ 분포(종목 × 구간 100칸). CPU 모델은 cpu_meta, GPU 모델은 gpu_meta에서 읽는다."""
    vals: dict = {}
    cm = pd.read_csv(_seed_file(0, "cpu_meta.csv"))
    for col in cm.columns:
        if col.endswith("_calib_pos") or (col.endswith("_calib") and col.split("_h")[0] in CPU_DET):
            nm = col.split("_h")[0]
            vals.setdefault(nm, []).extend(cm[col].dropna().tolist())
    gm = pd.read_csv(_seed_file(0, "gpu_meta.csv"))
    for nm, g in gm.groupby("모델"):
        vals[nm] = g["보정계수"].dropna().tolist()
    order = [m for m in ALL_MODELS if m in vals and m != "naive"]
    fig, ax = plt.subplots(figsize=(13, 4.6))
    bp = ax.boxplot([vals[m] for m in order], patch_artist=True, widths=0.6, showfliers=True)
    for patch, m in zip(bp["boxes"], order):
        patch.set_facecolor(GROUP_COLOR[GROUP_OF.get(m, "기준선")]); patch.set_alpha(0.55)
    ax.axhline(1, color=MUTED, lw=1, ls=":")
    ax.set_yscale("log")
    from matplotlib.ticker import FixedLocator, FormatStrFormatter, NullLocator
    ax.yaxis.set_major_locator(FixedLocator([0.25, 0.5, 1, 2, 4, 8]))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xticks(range(1, len(order) + 1), [sname(m) for m in order], rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("보정 계수 c₊(로그 축, 시드 0)")
    ax.grid(axis="y", color=GRID)
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


def garch_fix_section(path: Path, seeds) -> str:
    """GARCH-t 수렴 버그 수정 전후(설계 변경 2): 고치기 전 CPU 작업(parts/s*/cpu_garch_x100)과 고친 뒤의 GARCH-t·GARCH+LightGBM QLIKE."""
    rows = []
    for s in seeds:
        old_dir, new_dir = PARTS / f"s{s}" / "cpu_garch_x100", PARTS / f"s{s}" / "cpu"
        if not old_dir.exists():
            continue
        for f in sorted(new_dir.glob("*.pkl")):
            o = old_dir / f.name
            if not o.exists():
                continue
            new, old = pickle.loads(f.read_bytes()), pickle.loads(o.read_bytes())
            gm = {k: v for k, v in new["meta"].items() if k.startswith("garch_")}
            stuck = any(v is True for k, v in gm.items() if k.endswith("x100_stuck"))
            gain = max([v for k, v in gm.items() if k.endswith("ll_gain_vs_x100")] or [np.nan])
            for Rn, Ro in zip(new["results"], old["results"]):
                qn = {r["모델"]: r["QLIKE"] for r in Rn["rows"]}
                qo = {r["모델"]: r["QLIKE"] for r in Ro["rows"]}
                for nm in ("GARCH-t", "GARCH+LightGBM"):
                    if nm in qn and nm in qo:
                        rows.append({"시드": s, "종목": new["ticker"], "H": Rn["H"], "모델": nm, "QLIKE_수정전": qo[nm],
                                     "QLIKE_수정후": qn[nm], "차(후−전)": qn[nm] - qo[nm], "x100_시작값고착": stuck,
                                     "LL이득": gain})
    if not rows:
        return "- 해당 없음(수정 전 저장분이 없다)."
    d = pd.DataFrame(rows)
    d.to_csv(RES / f"{STEM}_garchfix_before_after.csv", index=False)
    d0 = d[d["시드"] == d["시드"].min()]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, nm in zip(axes, ("GARCH-t", "GARCH+LightGBM")):
        g = d0[d0["모델"] == nm]
        big = g["LL이득"] > 1.0
        lo, hi = float(min(g["QLIKE_수정전"].min(), g["QLIKE_수정후"].min())), float(max(g["QLIKE_수정전"].max(), g["QLIKE_수정후"].max()))
        ax.plot([lo, hi], [lo, hi], color=MUTED, lw=0.8, ls=":")
        ax.scatter(g.loc[~big, "QLIKE_수정전"], g.loc[~big, "QLIKE_수정후"], s=18, color="#9aa5b1", label="로그우도 변화 없음(정상 적합 종목)")
        ax.scatter(g.loc[big, "QLIKE_수정전"], g.loc[big, "QLIKE_수정후"], s=24, color="#c0392b", label="×100 적합이 최대우도가 아니었던 종목")
        ax.set_title(nm, fontsize=11, loc="left", fontweight="bold")
        ax.set_xlabel("QLIKE 수정 전(×100 척도 1회 적합)")
        ax.set_ylabel("QLIKE 수정 후(여러 척도 중 최대우도)")
        ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)
    aff = sorted(d0.loc[d0["LL이득"] > 1.0, "종목"].unique())
    summ = d0.groupby(["모델", d0["LL이득"] > 1.0])["차(후−전)"].agg(["mean", "min", "max", "count"])
    L = ["- **무엇이 문제였나**: 26c의 `garch_t_fit`은 수익률 ×100 척도로 arch를 한 번만 적합한다. 일부 종목은 최적화가 arch 시작값(α = 0.1, β = 0.88) "
         "근처에서 멈춘 채 끝나 최대우도 추정이 아니었다. 28번 내부학습 적합에서 영향받은 종목: " + (", ".join(aff) or "없음") +
         f"(×100 대비 최대 로그우도 이득 {d0['LL이득'].max():.1f}). 서비스 트랙(S194)이 26c 전체 적합에서 먼저 발견했고(AQT·BTC·MOC·STRAX·XLM·XRP, 26c 저장 모수가 정확히 시작값), 연구 세션이 다시 확인했다.",
         "- **어떻게 고쳤나**: ×100·×1000·×10 척도로 적합해 수렴 플래그 0이고 시작값이 아닌 해 중 ×100 척도로 환산한 로그우도 최대를 쓴다(`garch_t_fit_robust`). "
         "합성 GARCH(1,1)-t 복원 시험을 자체 시험에 넣었다. GARCH-t와 그 예측을 특징으로 쓰는 GARCH+LightGBM이 든 CPU 작업(시드 0~4)을 다시 돌렸다. 신경망(GPU)은 GARCH를 쓰지 않아 그대로다.",
         "- **수정 전후 QLIKE(시드 0, 종목 × 구간)**:", "",
         "| 모델 | 영향 종목 여부 | 평균 차(후 − 전) | 최소 | 최대 | 칸 수 |", "| :--- | :--- | ---: | ---: | ---: | ---: |"]
    for (nm, a_), r in summ.iterrows():
        L.append(f"| {nm} | {'영향 종목' if a_ else '정상 종목'} | {r['mean']:+.4f} | {r['min']:+.4f} | {r['max']:+.4f} | {int(r['count'])} |")
    L += ["", f"![GARCH 수정 전후]({os.path.relpath(path, RES)})", "",
          "- **그림 읽는 법**: 점 하나가 (종목, 구간) 하나다. 대각선 아래면 수정 뒤 QLIKE가 작아졌다(좋아졌다). 빨간 점이 ×100 적합이 최대우도가 아니었던 종목이다.",
          "- **완료 실험에 대한 영향**: 26·26c·27번 GARCH-t(그리고 26c·27번 GARCH+LightGBM)는 같은 함수를 썼으므로 26c 전체 적합 기준 6종목의 수치가 최대우도 결과가 아니다. "
          "완료 실험은 다시 돌리지 않고 이 절로 영향을 알린다. GARCH 계열이 1위였던 결론(26b·27번 15분)은 28번 수치로 다시 판단한다."]
    return "\n".join(L)


def poster_numbers(tt: pd.DataFrame, sp: pd.DataFrame, S: int) -> str:
    """포스터 6차 요청문에 들어가는 숫자를 결과 표에서 직접 뽑는다(손으로 옮기다 틀리는 것을 막는다).
    27번 포스터의 결과 1(가장 요동칠 때 병렬 딥러닝 비교), 결과 2·표 3(묶음 검정 칸 수), 결론 1(칸별 1위)에 대응한다."""
    L = [f"# 28번 포스터 숫자(시드 {S}개, 시드 항 포함 검정, Holm 0.05)", "",
         "> `28_seed_expansion_test.py`의 `poster_numbers`가 결과 표에서 자동으로 만든다. 포스터에는 이 파일의 숫자만 쓴다.", ""]
    q5 = tt[(tt["범위"] == "Q5") & tt["모델"].map(GROUP_OF).eq("병렬(딥러닝)")]
    worse = q5[~q5["동률"]]
    L += ["## 결과 1: 가장 요동칠 때(Q5) 병렬 딥러닝 대 1위", "",
          f"- 비교 수: {len(q5)}번(모델 × 예측 구간). 1위보다 유의하게 나쁨 {len(worse)}번, 1위이거나 구분 안 됨 {len(q5) - len(worse)}번.",
          "- 구분 안 됨(또는 1위)인 칸: " + (", ".join(f"{hlabel(r.H)} {r.모델}" for r in q5[q5["동률"]].itertuples()) or "없음") + ".",
          "- 구간별 모델 수: " + ", ".join(f"{hlabel(H)} {int((q5['H'] == H).sum())}종" for H in M.HORIZONS_H) + ".", ""]
    L += ["## 결과 2·표 3: 처리 방식 묶음 검정(30칸 = 예측 구간 5 × 범위 6)", "",
          "| 비교(A − B) | A 우세 | B 우세 | 구분 안 됨 | 가장 요동칠 때 A 우세 구간 |", "| :--- | ---: | ---: | ---: | :--- |"]
    for name, g in sp.groupby("비교", sort=False):
        a_win = g["판정"].eq("순차 우세")
        q5w = g[(g["범위"] == "Q5") & a_win]["H"]
        L.append(f"| {name} | {int(a_win.sum())} | {int(g['판정'].eq('병렬 우세').sum())} | {int(g['판정'].eq('구분 안 됨').sum())} | "
                 + ("·".join(hlabel(h) for h in q5w) or "없음") + " |")
    L += ["", "## 결론 1: 칸별 1위(1위와 구분 안 되는 모델 수)", "",
          "| 구간 | " + " | ".join(SCOPES) + " |", "| :--- |" + " :--- |" * len(SCOPES)]
    for H in M.HORIZONS_H:
        cells = []
        for sc in SCOPES:
            g = tt[(tt["H"] == H) & (tt["범위"] == sc)]
            cells.append(f"{g['최선'].iloc[0]}({GROUP_OF.get(g['최선'].iloc[0], '기준선')}, 동률 {int(g['동률'].sum()) - 1})")
        L.append(f"| {hlabel(H)} | " + " | ".join(cells) + " |")
    full = tt[tt["범위"] == "전체"]
    L += ["", "## 후보 수", "",
          "- 구간별 1위 후보(naive 제외): " + ", ".join(f"{hlabel(H)} {int(((full['H'] == H) & (full['모델'] != 'naive')).sum())}개"
                                             for H in M.HORIZONS_H) + "."]
    return "\n".join(L) + "\n"


def _design_section() -> str:
    from report_header import render_design_section
    return render_design_section(__file__)


def standard_header(tickers: list[str]) -> str:
    from report_header import render_standard_header, use_source
    split_frac = float(np.mean(M.load_close("KRW-BTC").index < M.SPLIT))
    use_source(M.DB_PATH, M.DATA_START, M.DATA_END)
    try:
        return render_standard_header(tickers=tickers, train_frac=split_frac, rep_ticker="KRW-BTC", analyzed_tickers=tickers,
                                      transforms=[
            ("15분 시간 격자 복원·점검 구간 제외·로그수익률", "26c·27과 같다"),
            ("타깃 = 다음 H분 실현변동성(RV), 손실 QLIKE, 가격 정지 두 부분 모형", "26c·27과 같다"),
            ("무재적합(내부 모델 하나로 보정·평가 예측)", "보정 계수와 평가 예측 모델의 불일치 제거(설계 ⑥)"),
            ("정지 확률 π도 내부 분류기 하나(random_state 0)", "모든 모델·시드가 같은 π를 쓴다"),
            ("학습 신경망 10종 학습률 3점 탐색(시드 0에서 고정)", "같은 튜닝 기회(설계 ②③)"),
            ("시드 S개 손실 평균, 검정 표준오차에 시드 항", "1종 오류 팽창 방지")])
    finally:
        use_source()


def write_report(tickers: list[str]) -> None:
    dec = json.loads((RES / f"{STEM}_seed_decision.json").read_text())
    S = int(dec["S"])
    seeds = list(range(S))
    stores = load_seed_stores(seeds)
    R = analyze(stores, "28")
    tt, sp = R["ties"], R["seqpar"]
    tt.to_csv(RES / f"{STEM}_best_ties.csv", index=False)
    sp.to_csv(RES / f"{STEM}_seqpar_dm.csv", index=False)
    R27 = analyze(stores27(), "27")
    R27["ties"].to_csv(RES / f"{STEM}_ref27_best_ties.csv", index=False)
    R27["seqpar"].to_csv(RES / f"{STEM}_ref27_seqpar_dm.csv", index=False)
    gm = gpu_meta_all(seeds)
    gm.to_csv(RES / f"{STEM}_gpu_meta_all_seeds.csv", index=False)
    # 그림 번호 = 보고서에 나오는 순서(1 데이터·타깃 점검 → 10 27번 대비)
    figs = {"data": IMG / f"{STEM}_fig1_data_target.png", "r": IMG / f"{STEM}_fig2_seed_error_ratio.png",
            "proc": IMG / f"{STEM}_fig3_procedure.png", "cal": IMG / f"{STEM}_fig4_calibration.png",
            "rank": IMG / f"{STEM}_fig5_ranking.png", "top5": IMG / f"{STEM}_fig6_top5_by_horizon.png",
            "sp": IMG / f"{STEM}_fig7_seqpar.png", "rg": IMG / f"{STEM}_fig8_regime_best.png",
            "lines": IMG / f"{STEM}_fig9_regime_lines.png", "vs": IMG / f"{STEM}_fig10_vs27_rank.png"}
    data_md = data_target_check(stores[0], figs["data"])
    fig_calibration(seeds, figs["cal"])
    fig_procedure(gm, figs["proc"]); fig_ranking(tt, figs["rank"]); fig_seqpar(sp, figs["sp"])
    fig_regime(tt, figs["rg"]); fig_vs27(tt, R27["ties"], figs["vs"])
    fig_top5(tt, figs["top5"], S); fig_regime_lines(tt, figs["lines"], S)
    (RES / f"{STEM}_poster_numbers.md").write_text(poster_numbers(tt, sp, S), encoding="utf-8")
    rel = lambda p: os.path.relpath(p, RES)
    rt = pd.read_csv(RES / f"{STEM}_seed_error_ratio.csv")
    L: list[str] = []
    e = L.append
    e("# 28번: 학습 절차 정비 + 시드 확장(R 재측정으로 시드 수 결정)")
    e("")
    e("## 실행·분석 환경")
    e("")
    e(f"- 서버 RTX 4090 24GB, `.venv`(torch·neuralforecast 3.2.2), S-Mamba는 `{X.SMAMBA_VENV.name}`. 드라이버 `test/models/28_seed_expansion_test.py`.")
    e(f"- 시드 {S}개(0~{S - 1}), 종목 {len(tickers)}개, 예측 구간 {', '.join(hlabel(h) for h in M.HORIZONS_H)}.")
    e("")
    e(_design_section())
    e(standard_header(tickers))
    e("")
    e("## 0. 데이터·타깃 점검(이전 회차 EDA와 같은 방식, 이번 회차에 바뀐 것 중심)")
    e("")
    e(data_md)
    e("")
    e("## 1. 시드 수 결정(시범 실행 시드 0~4)")
    e("")
    e(f"![시드 오차 비율]({rel(figs['r'])})")
    e("")
    e("- **그림 읽는 법**: 왼쪽은 쌍(순환 신경망 2종 × 학습하는 병렬 딥러닝 8종 × 5구간)마다 시드 5개로 잰 원시 R(가로)과 경험적 베이즈로 "
      "축소한 R(세로)이다. 빨간 점선이 축소 R의 90분위다. 오른쪽은 시드 수에 따라 시드 항이 표준오차를 몇 배로 키우는지(√(1+R/S))다. "
      "점선 1.10을 넘지 않는 가장 작은 후보가 채택 S다.")
    e(f"- 결과: 축소 R 중앙값 {dec['R_중앙']:.2f}, 90분위 {dec['R90']:.2f}, 최대 {dec['R_최대']:.2f}"
      f"(원시 R 중앙값 {dec['R_원시_중앙']:.2f}·90분위 {dec['R_원시_90']:.2f}, 축소 가중 {dec['축소가중']:.2f}). "
      f"27번 시드 5개 측정값은 중앙 1.62·90분위 6.28이었다.")
    e(f"- 결정: **S = {S}**. {dec['근거']}. 결정 시각 {dec['결정시각']}(본 실행 결과를 보기 전).")
    over = rt[np.sqrt(1 + rt["R"] / S) > INFL_MAX]
    e(f"- S = {S}에서 시드 항이 표준오차를 10% 넘게 키우는 쌍: " +
      ("없음." if over.empty else ", ".join(f"{hlabel(r.H)} {sname(r.A)}−{sname(r.B)}(배율 {np.sqrt(1 + r.R / S):.2f})"
                                          for r in over.itertuples()) + "."))
    e("")
    e("## 2. 학습 절차 정비 진단")
    e("")
    e(f"![학습 절차 진단]({rel(figs['proc'])})")
    e("")
    e("- **그림 읽는 법**: 왼쪽은 시드 0에서 고른 학습률이 격자의 어디였는지(하단 = 기본값 × 0.25, 상단 = × 4)를 모델별로 보인다. "
      "상단이나 하단 비율이 높으면 범위가 그쪽에서 잘렸을 수 있다. 오른쪽은 학습 횟수 상한에 닿은 적합의 비율이다(27번 neuralforecast "
      "6종은 상한 1,000에서 1~48%).")
    if not gm.empty:
        cap = gm[gm["상한도달"].isin([True, False])].groupby("모델")["상한도달"].mean()
        e("")
    e(f"![보정 계수]({rel(figs['cal'])})")
    e("")
    e("- **그림 읽는 법**: 모델마다 (종목, 구간) 100칸의 보정 계수 c₊(시드 0)다. 무재적합이라 보정 계수를 추정한 모델과 평가 예측을 낸 모델이 같다. "
      "로그 타깃 모델은 역변환 편향(Jensen) 때문에 1보다 큰 값이 정상이고, 1에서 한 자릿수 배 넘게 벗어나거나 칸마다 크게 흩어지면 적합 이상을 의심한다. "
      "GARCH 계열은 분산을 직접 예측해 1 근처가 정상이다.")
    e("- 상한 도달 비율: " + ", ".join(f"{sname(m)} {v:.1%}" for m, v in cap.items()) +
          ". ModernTCN·S-Mamba는 저자 학습 루프가 단계 수를 돌려주지 않아 해당 없음(기록 안 함)이다.")
    e("")
    e("### 2-1. GARCH-t 수렴 버그와 수정(설계 변경 2)")
    e("")
    e(garch_fix_section(IMG / f"{STEM}_garchfix_before_after.png", seeds))
    e("")
    e("## 3. 예측 구간별 순위(전체 기간)")
    e("")
    e(f"![순위]({rel(figs['rank'])})")
    e("")
    e("- **그림 읽는 법**: 예측 구간마다 모든 모델의 1위 대비 QLIKE 격차(시드 평균, 막대 = 시드 표준편차, 가로축 대칭 로그)다. 색 = 처리 방식, ○ = 1위와 구분 안 됨, × = 1위보다 유의하게 나쁨.")
    e("")
    e(f"![상위 5개]({rel(figs['top5'])})")
    e("")
    e("- **그림 읽는 법(포스터 그림 2 자리, 27d 그림 5와 같은 형식)**: 위는 전체 기간, 아래는 가장 요동칠 때(Q5)의 예측 구간별 상위 5개 모델이다. "
      "가로축은 1위와의 QLIKE 차이, 색은 처리 방식, 채운 점은 1위와 구분 안 됨, 빈 점은 1위보다 유의하게 나쁨이다. 점선 아래는 5위 밖 병렬 딥러닝 중 가장 나은 모델이다.")
    e("")
    e("- **검정**: 비교 집단은 구간 1위 모델 대 나머지 각 모델, 시각마다 20종목 평균 QLIKE의 차이(대응 표본)다. H0: 두 모델의 기대 QLIKE가 "
      "같다. H1: 같지 않다(양측). 표준오차는 SE_전체(HAC + 시드 항), 정규 근사 p에 모델 전체 Holm 보정. H0를 기각하지 못하면 \"1위와 구분 안 됨\"(동률), "
      "기각하면 1위보다 유의하게 나쁘다.")
    for H in M.HORIZONS_H:
        g = tt[(tt["H"] == H) & (tt["범위"] == "전체")].sort_values("평균QLIKE")
        best = g["최선"].iloc[0]
        tie = [m for m in g["모델"] if g.set_index("모델").loc[m, "동률"] and m != best]
        e(f"- {hlabel(H)}: 1위 {best}, 동률 {len(tie)}종(" + (", ".join(tie) if tie else "없음") + ").")
    e("")
    e("| 구간 | 순위 | 모델 | 묶음 | 평균 QLIKE | 시드 SD | 1위와 차 | SE_전체 | 시드 항 배율 | Holm p | 판정 |")
    e("| :--- | ---: | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |")
    for H in M.HORIZONS_H:
        g = tt[(tt["H"] == H) & (tt["범위"] == "전체")].sort_values("평균QLIKE")
        for r in g.itertuples():
            verdict = "1위" if r.모델 == r.최선 else ("구분 안 됨" if r.동률 else "1위보다 나쁨")
            se = getattr(r, "SE_전체", np.nan)
            infl = getattr(r, "배율", np.nan)
            e(f"| {hlabel(H)} | {r.순위} | {r.모델} | {r.묶음} | {r.평균QLIKE:.4f} | {r.시드SD:.4f} | {r.평균차:+.4f} | "
              f"{'해당 없음(1위)' if r.모델 == r.최선 else f'{se:.4f}'} | "
              f"{'해당 없음' if not np.isfinite(infl) else f'{infl:.2f}'} | {r.p_holm:.3g} | {verdict} |")
    e("")
    e("## 4. 순차 대 병렬 묶음 검정")
    e("")
    e(f"![순차 대 병렬]({rel(figs['sp'])})")
    e("")
    e("- **검정**: 비교 집단은 묶음 구성원 손실의 단순 평균끼리(A − B, 음수면 순차 쪽이 낫다). H0: 두 묶음의 기대 QLIKE가 같다. H1: 같지 않다. "
      "SE_전체, 비교마다 30칸(구간 5 × 범위 6)에 Holm. 범위의 Q1~Q5는 직전 RV의 종목별 학습 구간 5분위다.")
    e("")
    e("| 비교 | 구간 | 범위 | 평균차 | SE_시간 | SE_전체 | Holm p | 판정 | 27번 판정(같은 방식, 시드 5) |")
    e("| :--- | :--- | :--- | ---: | ---: | ---: | ---: | :--- | :--- |")
    s27 = R27["seqpar"].set_index(["비교", "H", "범위"])
    for r in sp.itertuples():
        v27 = s27.loc[(r.비교, r.H, r.범위), "판정"] if (r.비교, r.H, r.범위) in s27.index else "해당 없음"
        e(f"| {r.비교} | {hlabel(r.H)} | {r.범위} | {r.평균차:+.4f} | {r.SE_시간:.4f} | {r.SE_전체:.4f} | {r.p_holm:.3g} | {r.판정} | {v27} |")
    e("")
    e("## 5. 국면별 1위와 동률")
    e("")
    e(f"![국면별 1위]({rel(figs['rg'])})")
    e("")
    e("- 칸마다 1위 모델(색 = 묶음)과 1위와 구분되지 않는 모델 수다. 검정은 3절과 같고 범위만 국면으로 바꿨다.")

    e("")
    e(f"![국면별 처리 방식]({rel(figs['lines'])})")
    e("")
    e("- **그림 읽는 법(포스터 그림 3 자리, 27d 그림 1b와 같은 형식)**: 국면(Q1 가장 잔잔 → Q5 가장 요동)마다 처리 방식 묶음별로 가장 나은 모델이 "
      "1위와 얼마나 차이 나는지다. 0이면 그 묶음의 모델이 1위다.")
    e("")
    e("## 6. 27번과의 차이")
    e("")
    e(f"![27번 대비 순위]({rel(figs['vs'])})")
    e("")
    e("- 27번 순위는 27번 저장 예측(시드 0~4)을 이 보고서와 같은 함수로 다시 계산한 것이다(재적합 있음, 학습률 탐색 없음). "
      "4계단 이상 바뀐 모델에 이름을 달았다.")
    t27 = R27["ties"]
    ch = []
    for H in M.HORIZONS_H:
        a = tt[(tt["H"] == H) & (tt["범위"] == "전체")].set_index("모델")
        b = t27[(t27["H"] == H) & (t27["범위"] == "전체")].set_index("모델")
        e(f"- {hlabel(H)}: 1위 27번 {b['최선'].iloc[0]} → 28번 {a['최선'].iloc[0]}.")
        for m in a.index.intersection(b.index):
            ch.append({"H": H, "모델": m, "27번순위": int(b.loc[m, "순위"]), "28번순위": int(a.loc[m, "순위"]),
                       "27번QLIKE": b.loc[m, "평균QLIKE"], "28번QLIKE": a.loc[m, "평균QLIKE"]})
    pd.DataFrame(ch).to_csv(RES / f"{STEM}_vs27_rank.csv", index=False)
    e("")
    e("## 7. 한계")
    e("")
    e("- KernelRidge-RBF·SVR-RBF는 학습 데이터를 최근 14,188행·8,000행만 잘라 써서 공정 비교가 성립하지 않아 이번 회차에서 "
      "적합하지 않았다(설계 변경, 2026-10-15 사용자 결정). 커널 계열은 학습 데이터 전체를 쓰는 Nystroem+Ridge가 대표한다. "
      "27번 대비 순위 비교에서도 두 모델을 뺐다.")
    e("- **전처리 통계 범위(2026-10-17 적대적 리뷰, 서비스 트랙 S188에서 확인)**: 26c에서 물려받은 주기 스케일(`build_from_close`, "
      "`tr_bar = grid < 외부 분할`)과 GRU·LSTM 입력 표준화(`make_sequences`)가 외부 분할(2025-11-11) 이전 전체로 통계를 잡는다. "
      "그래서 내부검증 15%(학습률 선택·보정 계수 추정에 쓰는 구간)가 전처리 통계에 들어간다. 외부 평가 구간은 쓰지 않으므로 평가 누설은 아니다. "
      "주기 스케일은 모든 모델이 같이 쓰므로 영향은 공통이고, 내부검증 기반 선택(학습률·보정 계수)이 약간 낙관적일 수 있다. "
      "트리·커널 근사의 특징 표준화와 블록 입력 신경망의 타깃 표준화는 내부학습 구간까지만 쓴다. 포스터 일정상 28번은 이 설계로 완주했고, "
      "수정(전처리 통계를 내부학습 구간으로 한정)은 29번 재학습 설계에 반영한다.")
    e("- 무재적합이라 평가 예측 모델이 학습 구간 끝 15%를 학습하지 않는다(입력 창으로는 본다). 27번보다 모든 학습 모델이 같은 정도로 불리하다.")
    e("- 학습률은 시드 0에서 골라 고정했다. 시드 0은 자기 검증 구간에서 고른 학습률을 쓰므로 미세하게 유리할 수 있다.")
    e("- 시드는 같은 데이터·설정 안의 학습 난수만 다룬다. 분할 시점, 하이퍼파라미터 선택, 평가 기간 선택의 변동은 다루지 않는다(Bouthillier 외 2021).")
    e("- 파운데이션 7종은 재추론하지 않고 27번 예측에 28번 π만 반영했다(c₊는 π와 무관해 정확한 교체다).")
    e("- 기존 평가 구간(2025-11-11~2026-10-05)은 이미 여러 번 본 구간이라 이 결과는 탐색 성격이다. 확증은 2026-10-06 이후 구간에서 한다(연구 계획 4단계).")
    (RES / f"{STEM}_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[보고서] {RES / f'{STEM}_report.md'}", flush=True)


# %% [markdown]
# ## 자체 시험(합성 데이터, 수 초)

# %%
def selftest() -> None:
    rng = np.random.default_rng(0)
    # HAC: 독립 표본이면 표준오차 ≈ σ/√n
    d = rng.normal(0.1, 1.0, 4000)
    mu, se, p = hac_mean_test(d)
    assert abs(se - 1 / np.sqrt(4000)) < 0.004 and p < 1e-3
    # 시드 결정 규칙
    assert decide_seeds(1.0)[0] == 20 and decide_seeds(6.28)[0] == 30 and decide_seeds(9.0)[0] == 50
    assert decide_seeds(200.0)[0] == 50
    # EB 축소: 참값이 모두 같으면 축소 가중이 0에 가깝고 R이 모두 같다
    k = 4
    t = pd.DataFrame({"σ²_시드": rng.chisquare(k, 200) / k * 2.0, "SE_시간": 1.0, "시드수": k + 1})
    s = eb_shrink_R(t)
    assert s.attrs["축소가중"] < 0.35 and s["R"].std() < 0.6, s.attrs
    # 학습률 격자
    assert lr_grid("PatchTST") == (2.5e-5, 1e-4, 4e-4)
    assert len(STOCHASTIC) == 15 and not set(STOCHASTIC) & set(CPU_DET + FM_MODELS)
    assert not set(DECISION_EXCLUDED) & set(ALL_MODELS + PAR_FEAT + PAR_DEEP + SEQ)
    # π 교체식: π가 같으면 예측이 그대로
    p27, pi = rng.uniform(.1, 1, 50), rng.uniform(0, .5, 50)
    assert np.allclose(p27 * np.sqrt((1 - pi) / (1 - pi)), p27)
    # GARCH-t 복원: 15분 수익률 크기(표준편차 약 0.3%)의 합성 GARCH(1,1)-t에서 시작값이 아닌 참값 근처를 찾아야 한다
    n, om, al, be, nu = 6000, 2e-7, 0.15, 0.80, 5.0
    e = rng.standard_t(nu, n) * np.sqrt((nu - 2) / nu)
    r_, h = np.zeros(n), om / (1 - al - be)
    for i in range(n):
        r_[i] = np.sqrt(h) * e[i]
        h = om + al * r_[i] ** 2 + be * h
    g = garch_t_fit_robust(r_, n - 500)
    assert abs(g["alpha"] - al) < 0.06 and abs(g["beta"] - be) < 0.08 and np.isfinite(g["cv"]).all(), (g["alpha"], g["beta"])
    print("[자체 시험] 통과", flush=True)


# %% [markdown]
# ## 실행

# %%
def main(argv=None) -> None:
    from report_header import study_universe
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True,
                    choices=["selftest", "pi", "cpu", "gpu", "garchbic", "combine", "rmeasure", "report", "interim"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", default="", help="combine·report용 시드 목록(쉼표). 비우면 결정 파일의 S")
    ap.add_argument("--models", default=",".join(GPU_MODELS))
    ap.add_argument("--horizons", default=",".join(str(h) for h in M.HORIZONS_H))
    ap.add_argument("--tickers", default="", help="쉼표 목록(속도 측정용). 비우면 20종목")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--retries", type=int, default=1)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args(argv)
    selftest()
    if a.family == "selftest":
        return
    RES.mkdir(parents=True, exist_ok=True)
    IMG.mkdir(parents=True, exist_ok=True)
    tickers = a.tickers.split(",") if a.tickers else study_universe()[0][:20]
    if a.quick and not a.tickers:
        tickers = tickers[:2]
    M.halt_labels()
    t0 = time.time()
    if a.family == "pi":
        left = run_pool([(tk, a.quick) for tk in tickers], pi_job_task, a.workers, "π", a.retries, kind="pi")
    elif a.family == "cpu":
        left = run_pool([(tk, a.seed, a.quick) for tk in tickers], _cpu_task, a.workers, f"CPU s{a.seed}", a.retries,
                        kind="cpu_s0" if a.seed == 0 else "cpu")
    elif a.family == "gpu":
        hs = [int(h) for h in a.horizons.split(",")]
        tasks = [(tk, H, nm, a.seed, a.quick) for nm in a.models.split(",") for H in hs for tk in tickers]
        left = run_pool(tasks, _gpu_task, a.workers, f"GPU s{a.seed}", a.retries, kind="gpu")
    elif a.family == "garchbic":
        left = run_pool([(tk, a.quick) for tk in tickers], _garchbic_task, a.workers, "GARCH-BIC", a.retries, kind="cpu")
    elif a.family == "combine":
        for s in [int(x) for x in a.seeds.split(",")] if a.seeds else [a.seed]:
            save_seed(s, tickers, a.quick)
        left = []
    elif a.family == "rmeasure":
        dec = measure_and_decide(tickers)
        print(f"[시드 수 결정] {json.dumps(dec, ensure_ascii=False)}", flush=True)
        left = []
    else:
        write_report(tickers)
        left = []
    if left:
        print(f"[미완] 끝까지 실패한 작업 {len(left)}개: {left[:5]}", flush=True)
        sys.exit(1)
    print(f"[{a.family} 완료] {(time.time() - t0) / 60:.1f}분", flush=True)


def pi_job_task(tk, quick):
    return "ok", pi_job(tk, quick)


if __name__ == "__main__":
    main()
