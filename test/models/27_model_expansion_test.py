# %% [markdown]
# # 27번: 신규 알고리즘 확대(병렬 어텐션·합성곱·파운데이션)
#
# 26c번의 평가 틀을 그대로 쓴다: 같은 데이터 창(2023-10-06 ~ 2026-10-05, 분할 2025-11-11), 시간 격자, 정시 예측 시점,
# 5개 예측 구간, 내부검증 보정, 가격 정지 두 부분 모형(공유 분류기 π). 모델 목록만 신규로 바꾼다.
#
# | 계열 | 모델 | 구현 |
# | :--- | :--- | :--- |
# | 병렬 어텐션 | PatchTST, iTransformer, Autoformer, TimeXer | neuralforecast 공식 패키지 |
# | 합성곱 | TCN, TimesNet | neuralforecast 공식 패키지 |
# | 합성곱 | ModernTCN | 저자 공식 GitHub 코드(어댑터 `27_adapters.py`) |
# | 파운데이션(zero-shot) | Chronos-Bolt, TimesFM, TTM, Moirai-2, Sundial, Time-MoE, Lag-Llama | 각 공식 패키지, 격리 venv는 서브프로세스 |
#
# 제외: S-Mamba(`mamba_ssm`이 CUDA 컴파일을 요구하는데 서버에 `nvcc`가 없어 설치 실패), xLSTM·TimeGPT·DLinear·NLinear
# (제외 이유는 `test/research_materials/model_catalog.md`).
#
# ## 입력 형식(공통)
#
# 신규 모델은 시계열 예측 라이브러리라 "한 시계열의 과거 → 다음 값"을 푼다. 26c의 GRU·LSTM처럼 15분봉 96개를 그대로 읽는
# 대신, **예측 구간 길이 H의 블록 시계열**을 만든다. 블록 k의 값은 `log RV_d`, 곧 그 블록 안 주기 제거 수익률 제곱합의
# 제곱근의 로그이며, 이것이 26c의 타깃과 같다. 블록은 자정에 맞춰 겹치지 않게 자른다. 따라서 예측 시점(정시)은 항상 블록
# 경계이고, 목표는 다음 한 블록(h=1)이다. RV=0이거나 점검 구간을 걸친 블록은 값이 없으므로 **직전값 유지(LOCF) + 손실 마스크
# (`available_mask=0`)**로 처리한다(2026-10-05 결정: 20종목에서 가린 값을 복원해 비교한 결과 인과 평활의 최적해가 LOCF).
#
# ## 학습 절차(26c와 같은 두 단계)
#
# 1. **내부학습/내부검증**: `inner` 이전 블록으로 학습하고(끝 15%는 조기종료용 검증), `inner` 이후 블록의 예측으로 보정
#    계수 c(내부검증 정시 시점 QLIKE 최적 배율)를 정한다.
# 2. **전체 재적합**: 분할 시각 이전 전체로 다시 학습하고(끝 15%는 조기종료용), 평가 구간을 예측한다.
#
# 신경망 라이브러리의 조기종료 검증 구간은 학습 구간의 끝에서 떼어 쓰므로 26c의 GRU·LSTM(내부검증 구간으로 조기종료)과 정확히
# 같지는 않다. 이 차이는 보고서에 단서로 적는다. 하이퍼파라미터는 라이브러리 기본값을 쓰고 튜닝하지 않는다(입력 길이와 최대
# 스텝만 지정). 손실은 표준화한 로그 RV의 MSE다(GRU·LSTM과 같음).
#
# 정시 예측 시점이 15분·30분 블록 경계에 정확히 놓이고, 4시간·12시간은 블록이 자정 기준 4시간·12시간 단위라 26c의 평가
# 시점(`EVAL_HOURS`)과 일치한다. 보정 구간(내부검증)은 블록 경계에 맞춘 시각(`inner_blk`)부터이므로 4시간·12시간에서 26c의
# `inner`(정시)와 최대 11시간 다를 수 있다.

# %%
from __future__ import annotations

import argparse
import importlib.util
import logging
import multiprocessing as mp
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
import numpy as np
import pandas as pd

for _l in ("pytorch_lightning", "lightning_fabric", "lightning", "neuralforecast"):
    logging.getLogger(_l).setLevel(logging.ERROR)


def _project_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "engine").is_dir() and (p / "AGENTS.md").exists():
            return p
    return start


ROOT = _project_root(Path(__file__).resolve())
for _p in (ROOT, ROOT / "test" / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

TAG = "27_model_expansion_20261006"
STEM = "27_model_expansion"
SEED = int(os.environ.get("RUN27_SEED", "0"))
RUN_STEM = STEM if SEED == 0 else f"{STEM}_seed{SEED}"
RES = ROOT / "test" / "results" / TAG
IMG = ROOT / "test" / "images" / TAG
SRC26C = ROOT / "test" / "results" / "26c_recent_twopart_20261005"


def _load_m26c():
    spec = importlib.util.spec_from_file_location("m26c_for27", ROOT / "test" / "models" / "26c_recent_twopart_test.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = _load_m26c()

# %% [markdown]
# ## 설정

# %%
NF_MODELS = ("PatchTST", "iTransformer", "TCN", "Autoformer", "TimesNet", "TimeXer")
CONV_MODELS = ("ModernTCN",)
SSM_MODELS = ("S-Mamba",)
FM_MODELS = ("Chronos-Bolt", "TimesFM", "TTM", "Moirai-2", "Sundial", "Time-MoE", "Lag-Llama")
NEW_MODELS = NF_MODELS + CONV_MODELS + SSM_MODELS + FM_MODELS

NEW_FAMILY = {"PatchTST": "어텐션", "iTransformer": "어텐션", "Autoformer": "어텐션", "TimeXer": "어텐션",
              "TCN": "합성곱", "TimesNet": "합성곱", "ModernTCN": "합성곱", "S-Mamba": "상태공간",
              **{m_: "파운데이션" for m_ in FM_MODELS}}
NEW_PROCESS = {"PatchTST": "병렬 어텐션(패치)", "iTransformer": "병렬 어텐션(변수 토큰)", "Autoformer": "병렬 어텐션(자기상관)",
               "TimeXer": "병렬 어텐션(패치+변수 토큰)", "TCN": "병렬 합성곱(인과 팽창)", "TimesNet": "병렬 합성곱(2D 주기)",
               "ModernTCN": "병렬 합성곱(순수)", "S-Mamba": "선택적 상태공간(학습 병렬·추론 재귀)", "Chronos-Bolt": "사전학습 zero-shot(인코더-디코더)",
               "TimesFM": "사전학습 zero-shot(디코더)", "TTM": "사전학습 zero-shot(MLP-Mixer)",
               "Moirai-2": "사전학습 zero-shot(디코더)", "Sundial": "사전학습 zero-shot(플로우 매칭)",
               "Time-MoE": "사전학습 zero-shot(전문가 혼합)", "Lag-Llama": "사전학습 zero-shot(lag 디코더)"}
M.FAMILY.update(NEW_FAMILY)
M.PROCESS.update(NEW_PROCESS)
M.FAM_ORDER[:] = ["통계", "하이브리드", "트리", "딥러닝", "커널", "어텐션", "합성곱", "상태공간", "파운데이션"]
M.FAM_COLOR.update({"어텐션": "#4a3aa7", "합성곱": "#e87ba4", "상태공간": "#a0522d", "파운데이션": "#6b6b66"})
M.FAM_MARK.update({"어텐션": "X", "합성곱": "v", "상태공간": "h", "파운데이션": "*"})
M.FAM_LABEL.update({"어텐션": "병렬 어텐션", "합성곱": "병렬 합성곱", "상태공간": "선택적 상태공간", "파운데이션": "파운데이션(zero-shot)"})
M.SHORT.update({"PatchTST": "PTST", "iTransformer": "iTrans", "Autoformer": "Autof", "TimesNet": "TNet",
                "ModernTCN": "MTCN", "S-Mamba": "SMmb", "Chronos-Bolt": "Chr", "TimesFM": "TFM", "Moirai-2": "Moi",
                "Sundial": "Sun", "Time-MoE": "TMoE", "Lag-Llama": "LagL"})

INPUT_BLOCKS = {15: 96, 30: 96, 60: 96, 240: 60, 720: 30}   # 입력 길이(블록 수): 하루~며칠치 이력
# 26c 모듈의 목록을 신규 모델까지 넓힌다(모든 신규 모델은 로그 타깃이라 정지 확률 π를 받는다)
M.ALL_MODELS = tuple(M.ALL_MODELS) + NEW_MODELS
M.LOG_TARGET_MODELS = tuple(M.LOG_TARGET_MODELS) + NEW_MODELS
# 시드 집합은 모든 무작위 모델에 같게 쓴다. 신경망 6종의 계산량(시드 1개 약 15~20시간) 때문에 0·1·2로 정했고, 26c 모델도
# 이미 있는 0~4 중 0~2만 쓴다. 환경변수 RUN27_SEEDS로 바꿀 수 있다(예: "0,1,2,3,4").
# TTM r2 모델카드: 분·시간 해상도(10분·15분·1시간)만 지원. 4시간·12시간 블록 결과는 결론·검정에서 뺀다.
TTM_UNSUPPORTED_H = (240, 720)
SEEDS_USED = tuple(int(x) for x in os.environ.get("RUN27_SEEDS", "0,1,2").split(","))
STOCHASTIC_NEW = NF_MODELS + CONV_MODELS + SSM_MODELS          # 시드로 흔들리는 신규 모델(파운데이션은 zero-shot이라 학습 시드 없음)

NF_MAX_STEPS, NF_VAL_CHECK, NF_PATIENCE = 1000, 100, 3
NF_WINDOWS_BATCH = 256        # 모델마다 기본 배치가 달라(Autoformer 1,024 등) 계산량이 크게 갈려 모두 같은 값으로 맞춘다
VAL_FRAC = 0.15


# %% [markdown]
# ## 블록 시계열과 행 대응

# %%
def block_data(D: dict, H: int) -> dict:
    """한 종목·구간의 블록 시계열. 블록은 자정에 맞춘 H분 단위, 겹치지 않는다."""
    g, m = D["grid"], H // 15
    n = len(g)
    off = int(round((g[0] - g[0].floor("D")) / pd.Timedelta("15min"))) % m
    a0 = (-off) % m
    nb = (n - a0) // m
    lo = a0 + np.arange(nb) * m
    hi = lo + m
    d = np.nan_to_num(D["d"], nan=0.0)
    cs = np.r_[0.0, np.cumsum(d ** 2)]
    cr = np.r_[0.0, np.cumsum(d)]
    cb = np.r_[0.0, np.cumsum(D["bad"].astype(float))]
    rv = np.sqrt(np.maximum(cs[hi] - cs[lo], 0.0))
    badblk = (cb[hi] - cb[lo]) > 0
    ok = ~badblk & (rv > 0)
    y = np.where(ok, np.log(np.where(ok, rv, 1.0)), np.nan)
    ret = np.where(badblk, 0.0, cr[hi] - cr[lo])
    ds = pd.DatetimeIndex(g[a0] + pd.to_timedelta(np.arange(nb) * H, unit="m"))
    return dict(H=H, m=m, a0=a0, nb=nb, y=y, ret=ret, ok=ok, ds=ds)


def block_masks(D: dict, HD: dict, B: dict) -> dict:
    """행(예측 시점) ↔ 블록 대응과 학습·검증·평가 구분. 블록 경계에 맞춘 내부검증 시작 inner_blk을 쓴다."""
    H, m, a0 = B["H"], B["m"], B["a0"]
    j, T = HD["j"], HD["T"]
    aligned = ((j - a0) % m) == 0
    kj = (j - a0) // m
    inner = HD["inner"]
    day = inner.floor("D")
    inner_blk = day + ((inner - day) // pd.Timedelta(minutes=H)) * pd.Timedelta(minutes=H)
    split = HD["split"]
    ds = B["ds"]
    k_in = int(np.searchsorted(ds.values, np.datetime64(inner_blk)))        # 첫 내부검증 블록
    k_sp = int(np.searchsorted(ds.values, np.datetime64(split)))            # 첫 평가 블록
    va = HD["va"] & aligned & np.asarray(T >= inner_blk)
    te = HD["te"]
    if not np.all(aligned[te]):
        raise AssertionError("평가 시점이 블록 경계에 놓이지 않았다")
    return dict(aligned=aligned, kj=kj, inner_blk=inner_blk, k_in=k_in, k_sp=k_sp, va=va, te=te)


def standardize(B: dict, k_in: int) -> tuple[float, float]:
    y = B["y"][:k_in]
    y = y[np.isfinite(y)]
    return float(y.mean()), float(y.std())


def locf(y: np.ndarray) -> np.ndarray:
    return pd.Series(y).ffill().bfill().to_numpy()


# %% [markdown]
# ## neuralforecast 모델 적합(두 단계)

# %%
def make_nf_model(name: str, H: int, seed: int, quick: bool):
    from neuralforecast.losses.pytorch import MSE
    from neuralforecast import models as nfm
    L = INPUT_BLOCKS[H] if not quick else 16
    common = dict(h=1, input_size=L, loss=MSE(), valid_loss=MSE(), max_steps=NF_MAX_STEPS if not quick else 40,
                  val_check_steps=NF_VAL_CHECK if not quick else 20, early_stop_patience_steps=NF_PATIENCE,
                  windows_batch_size=NF_WINDOWS_BATCH, inference_windows_batch_size=512, valid_batch_size=512,
                  random_seed=seed, logger=False, enable_progress_bar=False, enable_model_summary=False)
    if name == "PatchTST":
        return nfm.PatchTST(**common)
    if name == "iTransformer":
        return nfm.iTransformer(n_series=2, **common)
    if name == "TCN":
        return nfm.TCN(**common)
    if name == "Autoformer":
        return nfm.Autoformer(**common)
    if name == "TimesNet":
        return nfm.TimesNet(**common)
    if name == "TimeXer":
        return nfm.TimeXer(n_series=1, **common)
    raise KeyError(name)


def nf_frame(name: str, B: dict, ymu: float, ysd: float, end: int) -> pd.DataFrame:
    """블록 [0, end)의 입력 프레임. iTransformer는 (log RV, 부호 있는 블록 수익률) 두 변수를 쓴다."""
    ds = B["ds"][:end]
    y = (locf(B["y"])[:end] - ymu) / ysd
    avail = B["ok"][:end].astype(float)
    parts = [pd.DataFrame({"unique_id": "rv", "ds": ds, "y": y, "available_mask": avail})]
    return parts


MODERNTCN_DIR = ROOT / "third_party" / "ModernTCN" / "ModernTCN-Long-term-forecasting"
MTCN_EPOCHS, MTCN_PATIENCE, MTCN_BATCH, MTCN_LR = 60, 10, 512, 1e-4


def modern_tcn_predict(B: dict, ymu: float, ysd: float, train_end: int, end: int, H: int, seed: int, quick: bool):
    """ModernTCN 공식 코드(저자 GitHub)를 같은 두 단계 절차로 학습·예측. 설정은 저자의 ETTh1 스크립트
    (patch 8, stride 4, ffn_ratio 1, 블록 1개, 큰 커널 51·작은 커널 5, dims 64, dropout 0.3, head_dropout 0, lr 1e-4,
    batch 512, 학습률 감소 type3, RevIN 사용, 다중 스케일 끔)를 따른다. 변수 1개(log RV), 다음 1블록 예측."""
    import types
    import torch
    import torch.nn as nn
    if str(MODERNTCN_DIR) not in sys.path:
        sys.path.insert(0, str(MODERNTCN_DIR))
    from models.ModernTCN import Model
    L = INPUT_BLOCKS[H] if not quick else 16
    cfg = types.SimpleNamespace(stem_ratio=6, downsample_ratio=2, ffn_ratio=1, num_blocks=[1], large_size=[51], small_size=[5],
                                dims=[64, 64, 64, 64], dw_dims=[256, 256, 256, 256], enc_in=1, small_kernel_merged=False,
                                dropout=0.3, head_dropout=0.0, use_multi_scale=False, revin=1, affine=0, subtract_last=0,
                                freq="h", seq_len=L, individual=0, pred_len=1, kernel_size=25, patch_size=8, patch_stride=4,
                                decomposition=0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = Model(cfg).to(dev)
    yl = ((locf(B["y"]) - ymu) / ysd).astype(np.float32)
    ok = B["ok"]
    win = lambda e: yl[e - L:e]                                        # 블록 e 직전 L개
    val_n = int(VAL_FRAC * train_end)
    fit_end = train_end - val_n
    tr_idx = np.array([e for e in range(L, fit_end) if ok[e]])
    va_idx = np.array([e for e in range(fit_end, train_end) if ok[e]])
    X = lambda idx: torch.tensor(np.stack([win(e) for e in idx])[:, :, None]).to(dev)
    Xtr, ytr = X(tr_idx), torch.tensor(yl[tr_idx]).to(dev)
    Xva, yva = X(va_idx), torch.tensor(yl[va_idx]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=MTCN_LR)
    lossf = nn.MSELoss()

    def infer(x):
        model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(x), 2048):
                out.append(model(x[i:i + 2048])[:, 0, 0].float())
        return torch.cat(out)

    best, best_state, bad = np.inf, None, 0
    for ep in range(MTCN_EPOCHS if not quick else 2):
        for g in opt.param_groups:                                      # 저자 lradj type3: 3에폭 이후 0.9배씩 감소
            g["lr"] = MTCN_LR * (0.9 ** max(0, ep - 3))
        model.train()
        perm = torch.randperm(len(Xtr), device=dev)
        for i in range(0, len(Xtr), MTCN_BATCH):
            b = perm[i:i + MTCN_BATCH]
            opt.zero_grad(set_to_none=True)
            loss = lossf(model(Xtr[b])[:, 0, 0], ytr[b])
            loss.backward()
            opt.step()
        vl = float(lossf(infer(Xva), yva).item())
        if vl < best - 1e-6:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= MTCN_PATIENCE:
                break
    model.load_state_dict(best_state)
    te_idx = np.arange(train_end, end)
    p = infer(X(te_idx)).cpu().numpy().astype(float)
    if not np.all(np.isfinite(p)):
        raise FloatingPointError("ModernTCN: 예측에 비유한값")
    return p


SMAMBA_VENV = ROOT / ".venvs" / "smamba_py310_20261006"


def smamba_predict(B: dict, ymu: float, ysd: float, train_end: int, end: int, H: int, seed: int, quick: bool):
    """S-Mamba는 구버전 torch·mamba_ssm이 필요해 격리 venv의 러너(`27_smamba_run.py`)를 서브프로세스로 부른다."""
    import subprocess
    import tempfile
    L = INPUT_BLOCKS[H] if not quick else 16
    rsd = float(np.std(B["ret"][:train_end])) or 1.0
    libs = ":".join(str(d) for d in sorted((SMAMBA_VENV / "lib" / "python3.10" / "site-packages" / "nvidia").glob("*/lib")))
    with tempfile.TemporaryDirectory() as td:
        fin, fout = Path(td) / "in.npz", Path(td) / "out.npz"
        np.savez(fin, y=((locf(B["y"]) - ymu) / ysd).astype(np.float32), ret=(B["ret"] / rsd).astype(np.float32),
                 ok=B["ok"], train_end=train_end, end=end, L=L, seed=seed, quick=quick)
        env = {**os.environ, "LD_LIBRARY_PATH": libs + ":" + os.environ.get("LD_LIBRARY_PATH", "")}
        r = subprocess.run([str(SMAMBA_VENV / "bin" / "python"), str(ROOT / "test" / "models" / "27_smamba_run.py"),
                            "--inp", str(fin), "--out", str(fout)], capture_output=True, text=True, env=env)
        if r.returncode != 0:
            raise RuntimeError(f"S-Mamba 러너 실패: {r.stderr[-300:]}")
        return np.load(fout)["p"]


def nf_predict(name: str, B: dict, ymu: float, ysd: float, train_end: int, end: int, H: int, seed: int, quick: bool):
    if name == "ModernTCN":
        return modern_tcn_predict(B, ymu, ysd, train_end, end, H, seed, quick)
    if name == "S-Mamba":
        return smamba_predict(B, ymu, ysd, train_end, end, H, seed, quick)
    """블록 [0, end)로 모델 하나를 학습(앞 train_end개 중 끝 15%는 조기종료 검증)하고 [train_end, end)를 예측.
    반환: 길이 end-train_end의 표준화 로그 RV 예측."""
    from neuralforecast import NeuralForecast
    parts = nf_frame(name, B, ymu, ysd, end)
    if name == "iTransformer":
        r = B["ret"][:end]
        rsd = float(np.std(B["ret"][:train_end])) or 1.0
        parts.append(pd.DataFrame({"unique_id": "ret", "ds": B["ds"][:end], "y": r / rsd,
                                   "available_mask": 1.0}))
    df = pd.concat(parts, ignore_index=True)
    mdl = make_nf_model(name, H, seed, quick)
    nf = NeuralForecast(models=[mdl], freq=f"{H}min")
    cv = nf.cross_validation(df, val_size=int(VAL_FRAC * train_end), test_size=end - train_end, n_windows=None,
                             step_size=1, refit=False)
    cv = cv[cv["unique_id"] == "rv"].sort_values("ds")
    if len(cv) != end - train_end:
        raise AssertionError(f"{name}: 예측 길이 {len(cv)} != {end - train_end}")
    exp = B["ds"][train_end:end]
    if not np.array_equal(pd.DatetimeIndex(cv["ds"]).values, exp.values):
        raise AssertionError(f"{name}: 예측 시각이 블록과 다르다")
    p = cv[name].to_numpy(float)
    if not np.all(np.isfinite(p)):
        raise FloatingPointError(f"{name}: 예측에 비유한값")
    return p


def score_blocks(S, name: str, full: np.ndarray, MK: dict, ymu: float, ysd: float, note: str) -> None:
    """블록별 표준화 로그 RV 예측(길이 nb, 필요한 블록만 유한) → 행 순서 원 스케일 → 보정 → 두 부분 점수."""
    va, te = MK["va"], MK["te"]
    pv = full[MK["kj"][va]] * ysd + ymu
    pt = full[MK["kj"][te]] * ysd + ymu
    if not (np.all(np.isfinite(pv)) and np.all(np.isfinite(pt))):
        raise FloatingPointError(f"{name}: 필요한 블록에 예측이 없다")
    cal = S.calib2(S.to_raw(pv, va))
    S.score_log(name, S.to_raw(pt, te), cal, note=note)


def prepare_job(ticker: str, H: int, quick: bool) -> dict:
    """공통 준비: 데이터, 블록, 마스크, 정지 확률, Scorer."""
    D = M.build_data(ticker)
    HD = M.horizon_data(D, H)
    B = block_data(D, H)
    MK = block_masks(D, HD, B)
    HD = dict(HD)
    HD["va"] = MK["va"]                      # 보정·π 검증 시점을 블록 경계 정시로 한정
    S = M.Scorer(ticker, HD)
    pi_va, pi_te, zinfo = load_or_fit_pi(ticker, H, D, HD, quick)
    S.set_zero(pi_va, pi_te)
    ymu, ysd = standardize(B, MK["k_in"])
    return dict(D=D, HD=HD, B=B, MK=MK, S=S, ymu=ymu, ysd=ysd, zinfo={"종목": ticker, "H": H, **zinfo})


PI_DIR = RES / "parts" / "pi_cache"


def run_fingerprint(quick: bool, models: tuple, family: str) -> dict:
    """작업 저장분을 재사용해도 되는지 판단하는 설정 지문. 결과를 바꾸는 설정이 하나라도 다르면 재사용하지 않는다."""
    return {"quick": bool(quick), "family": family, "models": list(models), "seed": SEED,
            "data": [str(M.DATA_START), str(M.DATA_END), str(M.SPLIT), str(M.DB_PATH.name)],
            "input_blocks": {str(k): v for k, v in INPUT_BLOCKS.items()}, "val_frac": VAL_FRAC,
            "nf": [NF_MAX_STEPS, NF_VAL_CHECK, NF_PATIENCE, NF_WINDOWS_BATCH],
            "moderntcn": [MTCN_EPOCHS, MTCN_PATIENCE, MTCN_BATCH, MTCN_LR], "pi": "26c 시드0 설정(random_state=0)"}


def load_or_fit_pi(ticker: str, H: int, D: dict, HD: dict, quick: bool):
    """정지 분류기는 26c 시드 0과 같은 설정(random_state=0)이라 시드·모델과 무관하다. 종목×구간마다 한 번만 적합해 저장한다."""
    import pickle
    f = PI_DIR / f"{ticker}_{H}{'_quick' if quick else ''}.pkl"
    if f.exists():
        return pickle.loads(f.read_bytes())
    out = M.zero_model(D, HD, M.make_features(D), M.zero_features(D), quick)
    PI_DIR.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".tmp")
    tmp.write_bytes(pickle.dumps(out))
    tmp.replace(f)
    return out


def pi_job(ticker: str, H: int, quick: bool) -> str:
    D = M.build_data(ticker)
    HD = M.horizon_data(D, H)
    B = block_data(D, H)
    MK = block_masks(D, HD, B)
    HD = dict(HD)
    HD["va"] = MK["va"]
    load_or_fit_pi(ticker, H, D, HD, quick)
    return f"{ticker}_{H}"


def finish_job(ticker: str, H: int, P: dict, fails: list, meta: dict, t0: float) -> dict:
    S, HD = P["S"], P["HD"]
    return dict(ticker=ticker, H=H, rows=S.rows, preds=S.preds, rows1=S.rows1, preds1=S.preds1,
                pi=S.pi_te.astype(np.float32), act=S.act.astype(np.float32), T=M._ns(HD["T"][HD["te"]]),
                zinfo=P["zinfo"], fails=fails, meta=meta, elapsed=time.time() - t0)


def run_nf_job(ticker: str, H: int, quick: bool, models: tuple, seed: int) -> dict:
    t0 = time.time()
    P = prepare_job(ticker, H, quick)
    B, MK, S, ymu, ysd = P["B"], P["MK"], P["S"], P["ymu"], P["ysd"]
    fails, meta = [], {"종목": ticker, "H": H}
    for nm in models:
        t1 = time.time()
        try:
            p1 = nf_predict(nm, B, ymu, ysd, MK["k_in"], MK["k_sp"], H, seed, quick)       # 내부학습 → 내부검증 구간
            p2 = nf_predict(nm, B, ymu, ysd, MK["k_sp"], B["nb"], H, seed, quick)          # 학습 전체 → 평가 구간
            full = np.full(B["nb"], np.nan)
            full[MK["k_in"]:MK["k_sp"]] = p1
            full[MK["k_sp"]:] = p2
            score_blocks(S, nm, full, MK, ymu, ysd, "라이브러리 기본값 · 조기종료 · 전체 재적합")
            meta[f"{nm}_calib"] = S.rows[-1]["보정계수"]
        except Exception as e:
            fails.append((nm, H, type(e).__name__, str(e)[:200]))
            traceback.print_exc()
        meta[f"{nm}_초"] = time.time() - t1
    return finish_job(ticker, H, P, fails, meta, t0)


# %% [markdown]
# ## 파운데이션 모델: 입력 작성 → 러너(모델별 venv) → 점수

# %%
FM_DIR = RES / "fm_io"


def fm_prepare_job(ticker: str, H: int, quick: bool) -> str:
    D = M.build_data(ticker)
    HD = M.horizon_data(D, H)
    B = block_data(D, H)
    MK = block_masks(D, HD, B)
    ymu, ysd = standardize(B, MK["k_in"])
    series = ((locf(B["y"]) - ymu) / ysd).astype(np.float32)
    orig = np.unique(np.r_[MK["kj"][MK["va"]], MK["kj"][MK["te"]]]).astype(np.int64)
    (FM_DIR / "in").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(FM_DIR / "in" / f"{ticker}_{H}.npz", series=series, orig=orig, H=H, ds0=str(B["ds"][0]))
    return f"{ticker}_{H}"


def run_fm_score_job(ticker: str, H: int, quick: bool, models: tuple) -> dict:
    t0 = time.time()
    P = prepare_job(ticker, H, quick)
    B, MK, S, ymu, ysd = P["B"], P["MK"], P["S"], P["ymu"], P["ysd"]
    fails, meta = [], {"종목": ticker, "H": H}
    for nm in models:
        try:
            z = np.load(FM_DIR / "out" / nm / f"{ticker}_{H}.npz")
            full = np.full(B["nb"], np.nan)
            full[z["orig"]] = z["pred"]
            score_blocks(S, nm, full, MK, ymu, ysd, "zero-shot(재학습 없음) · 중앙값 점예측")
            meta[f"{nm}_calib"] = S.rows[-1]["보정계수"]
        except Exception as e:
            fails.append((nm, H, type(e).__name__, str(e)[:200]))
            traceback.print_exc()
    return finish_job(ticker, H, P, fails, meta, t0)


# %% [markdown]
# ## 자체 시험

# %%
def selftest() -> None:
    rng = np.random.default_rng(3)
    n = 96 * 60
    idx = pd.date_range("2025-08-01", periods=n, freq=M.BAR)
    ret = rng.normal(0, 0.003, n) * (1 + 2 * (np.arange(n) % 96 > 40))
    px = pd.Series(100 * np.exp(np.cumsum(ret)), index=idx)
    halt = idx[96 * 20 + 8: 96 * 20 + 24]
    close = px.drop(halt)
    split = pd.Timestamp("2025-09-15 00:00")
    D = M.build_from_close(close, pd.DatetimeIndex(halt), split=split)
    D["ticker"] = "SYN"
    for H in M.HORIZONS_H:
        HD = M.horizon_data(D, H, split=split)
        B = block_data(D, H)
        MK = block_masks(D, HD, B)
        assert np.all(B["ds"][1:] - B["ds"][:-1] == pd.Timedelta(minutes=H)), "블록 등간격"
        assert (B["ds"][0] - B["ds"][0].floor("D")) % pd.Timedelta(minutes=H) == pd.Timedelta(0), "자정 기준 정렬"
        te = MK["te"]
        k = MK["kj"][te]
        assert np.array_equal(B["ds"][k].values, HD["T"][te].values), "평가 행의 블록 시작 = 예측 시점"
        # 블록 값 = 26c의 타깃
        ok = B["ok"][k]
        assert np.allclose(np.exp(B["y"][k][ok]), HD["act_d"][te][ok], rtol=1e-9), "블록 RV = 26c 타깃(RV_d)"
        assert B["ds"][MK["k_sp"]] == split and B["ds"][MK["k_in"]] == MK["inner_blk"], "내부검증·평가 시작 블록"
        # 학습 블록의 타깃 구간이 분할을 넘지 않는다
        assert B["ds"][MK["k_sp"] - 1] + pd.Timedelta(minutes=H) <= split, "학습 블록은 분할 이전에 끝남"
        if MK["va"].any():
            assert np.all(HD["T"][MK["va"]] >= MK["inner_blk"]) and np.all(MK["aligned"][MK["va"]])
    # 점검 구간을 걸친 블록은 값이 없다
    B60 = block_data(D, 60)
    kh = int(np.searchsorted(B60["ds"].values, np.datetime64(halt[0].floor("h"))))
    assert not B60["ok"][kh] and np.isnan(B60["y"][kh]), "점검 블록은 결측"
    assert np.allclose(B60["ret"][kh], 0.0), "점검 블록 수익률 0"
    print("[selftest 27] 모든 시험 통과", flush=True)


# %% [markdown]
# ## 저장과 실행

# %%
def _read_family(stem: str, seed: int = 0) -> list[tuple[pd.DataFrame, pd.DataFrame, dict, dict]]:
    """한 시드의 신규 모델 산출물(nf·conv·fm)을 (rows, rows1, store, store1)로 읽는다. 없는 계열은 건너뛴다."""
    out = []
    pre = f"{STEM}" if seed == 0 else f"{STEM}_seed{seed}"
    per_model = [f"nf-{m_}" for m_ in NF_MODELS]
    split = any((RES / f"{pre}_{t}_model_comparison.csv").exists() for t in per_model)
    for tag in per_model + ["nf", "conv", "ssm", "fm"]:
        f = RES / f"{pre}_{tag}_model_comparison.csv"
        if not f.exists():
            continue
        if tag == "nf" and split:      # 알고리즘별 순차 실행 이전의 6종 동시 실행 산출물. 모델별 산출물이 있으면 그쪽만 쓴다
            print(f"  [시드 {seed}] 옛 nf 묶음 산출물은 건너뜀(모델별 산출물 사용)", flush=True)
            continue
        rd = pd.read_csv(f)
        rd1 = pd.read_csv(RES / f"{pre}_{tag}_onepart_comparison.csv")
        st = M._npz_to_store(RES / f"{pre}_{tag}_test_predictions.npz")
        st1 = M._npz_to_store(RES / f"{pre}_{tag}_onepart_predictions.npz")
        out.append((rd, rd1, st, st1))
    if out:
        allr = pd.concat([r_ for r_, _, _, _ in out], ignore_index=True)
        dup = allr[allr.duplicated(["종목", "H", "모델"], keep=False)]
        if len(dup):
            raise AssertionError(f"시드 {seed}: 계열 산출물 사이에 (종목, H, 모델) 중복 {len(dup)}행 — {sorted(set(dup['모델']))}")
    return out


def combine() -> None:
    """26c의 12종+naive와 27번 신규 모델을 한 묶음으로 합친다(26b 검정 도구와 보고서가 읽는 형식).

    평가 시각·실제값은 모든 산출물에서 같아야 하므로 같은지 대조한다. 시드 파일에는 신규 모델의 시드별 예측과 26c의
    시드별 예측을 함께 담는다."""
    rd, store, rd1, store1 = M.load_saved()                        # 26c(MS-GARCH 재적합 반영)
    fam = _read_family(STEM, 0)
    if not fam:
        raise RuntimeError("27번 신규 모델 산출물이 없다")
    for rn, rn1, sn, sn1 in fam:
        rd, rd1 = pd.concat([rd, rn], ignore_index=True), pd.concat([rd1, rn1], ignore_index=True)
        for key, S in sn.items():
            T0 = store[key]["T"]
            if not (np.array_equal(S["T"], T0) and np.allclose(S["act"], store[key]["act"])):
                raise AssertionError(f"{key}: 신규 모델과 26c의 평가 시각·실제값이 다르다")
            store[key]["preds"].update(S["preds"])
            store1[key]["preds"].update(sn1[key]["preds"])
    drop = (rd["모델"] == "TTM") & rd["H"].isin(TTM_UNSUPPORTED_H)
    drop1 = (rd1["모델"] == "TTM") & rd1["H"].isin(TTM_UNSUPPORTED_H)
    rd[drop].to_csv(RES / f"{STEM}_excluded_ttm_unsupported_resolution.csv", index=False)
    rd, rd1 = rd[~drop].reset_index(drop=True), rd1[~drop1].reset_index(drop=True)
    for (tk, H), S in store.items():
        if H in TTM_UNSUPPORTED_H:
            S["preds"].pop("TTM", None)
            store1[(tk, H)]["preds"].pop("TTM", None)
    rd.to_csv(RES / f"{STEM}_model_comparison.csv", index=False)
    rd1.to_csv(RES / f"{STEM}_onepart_comparison.csv", index=False)
    M.save_npz(RES / f"{STEM}_test_predictions.npz", store, aux=True)
    M.save_npz(RES / f"{STEM}_onepart_predictions.npz", store1, aux=False)
    import shutil
    for f in ("chosen_hyperparams.csv", "zero_classifier.csv"):
        shutil.copy(SRC26C / f"26c_recent_twopart_{f}", RES / f"{STEM}_{f}")
    # 시드 묶음
    got = []
    for sd in [x for x in SEEDS_USED if x != 0]:
        f26 = SRC26C / f"26c_recent_twopart_seed{sd}_test_predictions.npz"
        fam_s = _read_family(STEM, sd)
        have = set().union(*[set(r_["모델"]) for r_, _, _, _ in fam_s]) if fam_s else set()
        missing = [m_ for m_ in STOCHASTIC_NEW if m_ not in have]
        if not f26.exists() or missing:
            print(f"  [시드 {sd}] 신규 모델 산출물 부족({', '.join(missing) or '26c 시드 파일 없음'}) → 이 시드는 합치지 않는다", flush=True)
            continue
        pr = {}
        for key, v in M._npz_to_store(f26).items():
            pr.setdefault(key, {}).update(v["preds"])
        rows_s = [pd.read_csv(SRC26C / f"26c_recent_twopart_seed{sd}_model_comparison.csv")]
        for rn, _, sn, _ in fam_s:
            rows_s.append(rn)
            for key, S in sn.items():
                pr.setdefault(key, {}).update(S["preds"])
        M.save_npz(RES / f"{STEM}_seed{sd}_test_predictions.npz", {k: {"preds": v} for k, v in pr.items()}, aux=False)
        pd.concat(rows_s, ignore_index=True).to_csv(RES / f"{STEM}_seed{sd}_model_comparison.csv", index=False)
        got.append(sd)
    models = [m_ for m_ in M.ALL_MODELS if m_ in set(rd["모델"])]
    cl = M.cell_losses(rd, store, models)
    tt = M.tier_table(cl)
    tt.to_csv(RES / f"{STEM}_tier_table.csv", index=False)
    print(f"[합치기 완료] 모델 {len(models)}종(naive 포함) · 행 {len(rd)} · 시드 {got}", flush=True)


def seed_mean_losses(rd: pd.DataFrame, store: dict, models: list[str]) -> tuple[pd.DataFrame, dict]:
    """시드 0~4의 칸별 손실을 평균한 cell_losses. 무작위성이 있는 모델은 시드마다 그 시드의 예측·보정계수로 손실을 구하고
    (손실의 평균이며 예측의 평균이 아니다), 나머지 모델은 시드와 무관하므로 같은 값이 평균된다. 반환: (평균 손실표, 모델별 시드 수)."""
    frames = [M.cell_losses(rd, store, models).assign(seed=0)]
    n_seed = {m_: 1 for m_ in models}
    for sd in [x for x in SEEDS_USED if x != 0]:
        f_rd, f_np = RES / f"{STEM}_seed{sd}_model_comparison.csv", RES / f"{STEM}_seed{sd}_test_predictions.npz"
        if not (f_rd.exists() and f_np.exists()):
            continue
        rds = pd.read_csv(f_rd)
        pr = {}
        for key, v in M._npz_to_store(f_np).items():
            pr[key] = v["preds"]
        st = {key: {**{a_: b_ for a_, b_ in S.items() if a_ != "preds"}, "preds": {**S["preds"], **pr.get(key, {})}}
              for key, S in store.items()}
        have = set(rds["모델"])
        rd_s = pd.concat([rd[~rd["모델"].isin(have)], rds[rds["모델"].isin(models)]], ignore_index=True)
        frames.append(M.cell_losses(rd_s, st, models).assign(seed=sd))
        for m_ in models:
            if m_ in have and m_ in set(M.LOG_TARGET_MODELS) and m_ not in FM_MODELS:
                n_seed[m_] += 1
    cl = pd.concat(frames, ignore_index=True)
    cl = cl.groupby(["종목", "H", "구간", "기준", "모델"], as_index=False)["QLIKE"].mean()
    return cl, n_seed


ROSTER = [
    # (계열, 모델, 구현, 입력, 학습·추론 방식, 시드 반복)
    ("통계", "GARCH-t·MS-GARCH·TAR-GARCH", "arch 패키지 / 자체 엔진(`engine/regime_garch.py`)", "15분 수익률", "최대우도, 다단계 재귀 예측", "해당 없음(결정적)"),
    ("커널", "KernelRidge-RBF·SVR-RBF·Nystroem+Ridge", "scikit-learn", "로그 특성 12개", "내부검증으로 설정 선택 후 전체 재적합", "Nystroem만(근사 난수)"),
    ("트리", "LightGBM·XGBoost·HistGBM", "공식 패키지", "원 스케일 특성 17개", "조기종료 후 전체 재적합", "예(HistGBM은 결과에 영향 없음)"),
    ("하이브리드", "GARCH+LightGBM", "위 두 구현의 결합", "트리 특성 + GARCH 예측", "조기종료 후 전체 재적합", "예"),
    ("순환 딥러닝", "GRU·LSTM", "PyTorch(`engine/models.py`)", "15분봉 96개 (d,|d|)", "내부검증 조기종료, 학습률 3개 비교, 전체 재적합", "예"),
    ("어텐션", "PatchTST·iTransformer·Autoformer·TimeXer", "neuralforecast 공식 패키지", "H분 블록 로그 RV 이력(iTransformer는 +블록 수익률)", "라이브러리 기본값, 조기종료, 전체 재적합", "예"),
    ("합성곱", "TCN·TimesNet", "neuralforecast 공식 패키지", "H분 블록 로그 RV 이력", "라이브러리 기본값, 조기종료, 전체 재적합", "예"),
    ("상태공간", "S-Mamba", "저자 공식 GitHub 코드(`third_party/S-D-Mamba`), 격리 venv", "H분 블록 로그 RV 이력 + 블록 수익률(2변수)", "저자 ETTh1 설정, 조기종료, 전체 재적합", "예"),
    ("합성곱", "ModernTCN", "저자 공식 GitHub 코드(`third_party/ModernTCN`)", "H분 블록 로그 RV 이력", "저자 ETTh1 설정, 조기종료, 전체 재적합", "예"),
    ("파운데이션", "Chronos-Bolt·TimesFM·TTM", "공식 패키지(메인 venv)", "직전 512블록 로그 RV", "zero-shot(재학습 없음), 중앙값 점예측", "해당 없음(결정적)"),
    ("파운데이션", "Moirai-2·Sundial·Time-MoE", "공식 패키지(격리 venv)", "직전 512블록 로그 RV", "zero-shot, Moirai-2 중앙 분위수 / Sundial 표본 50개 중앙값 / Time-MoE 점예측", "Sundial은 표본 추출(시드 고정, 반복 안 함)"),
    ("파운데이션", "Lag-Llama", "공식 GitHub 코드(`third_party/lag-llama`)", "직전 1,124블록 로그 RV", "zero-shot, 표본 100개 중앙값", "표본 추출(시드 고정, 반복 안 함)"),
]


def write_report27(elapsed_note: str = "") -> None:
    """합친 결과로 27번 보고서를 쓴다. 순위·동률·계열 요약은 26c의 함수를 재사용한다."""
    M.RES, M.STEM, M.IMG = RES, STEM, IMG
    M._LINES.clear()
    emit = M.emit
    rd, store, rd1, store1 = M.load_saved(STEM)
    models = [m_ for m_ in M.ALL_MODELS if m_ in set(rd["모델"])]
    new_in = [m_ for m_ in NEW_MODELS if m_ in set(rd["모델"])]
    new_out = [m_ for m_ in NEW_MODELS if m_ not in set(rd["모델"])]
    emit("# 27번: 신규 알고리즘 확대(병렬 어텐션·합성곱·파운데이션)")
    emit()
    emit("## 0. 이 회차가 한 것")
    emit()
    emit(f"26c번의 평가 틀(데이터 창 2023-10-06 ~ 2026-10-05, 분할 2025-11-11, 정시 예측 시점, 5개 예측 구간, 내부검증 보정, 정지 두 부분 모형)을 "
         f"그대로 두고, 26c의 12종 + naive에 **신규 {len(new_in)}종**을 같은 조건으로 더해 비교했다. 20종목 × 5구간 × {len(models)}모델 = "
         f"{20 * 5 * len(models)}행이 기대값이고 실제 {len(rd)}행이다.")
    emit()
    emit("- **26c 결과는 다시 계산하지 않고 그대로 가져왔다.** 평가 시각과 실제값이 모든 모델에서 같은지 합치는 단계에서 대조했다.")
    emit(f"- 신규 모델 중 이번 결과에 들어온 것: {', '.join(new_in) or '없음'}. 들어오지 못한 것: {', '.join(new_out) or '없음(전부 포함)'}.")
    emit("- **S-Mamba**: 공식 구현이 `mamba_ssm`(구버전 `torch==2.0.1` 필요)에 의존하고 서버에 `nvcc`가 없어 처음에는 설치에 실패했다. "
         "저자 요구 버전에 맞춰 미리 컴파일된 wheel로 **격리 venv**(`.venvs/smamba_py310_20261006`)를 구성해 해결했다(시스템 변경 없음). "
         "메인 환경의 torch 버전은 그대로다.")
    emit("- **TTM의 4시간·12시간 결과는 제외했다**: TTM r2 모델카드가 분·시간 해상도(10분·15분·1시간)만 지원한다고 명시한다. 이 두 구간은 "
         "공식 지원 범위 밖이라 순위·동률·검정에서 뺐고, 계산값은 `excluded_ttm_unsupported_resolution.csv`에 기록으로만 남겼다.")
    emit("- 제외 모델(Linear·Ridge·HAR-RV·DLinear·NLinear 등)의 제외 이유는 `test/research_materials/model_catalog.md`에 있다.")
    emit(f"- {elapsed_note}" if elapsed_note else "- 소요 시간 기록: 해당 없음(여러 실행으로 나뉘어 합산하지 않았다).")
    emit()
    emit("## 1. 비교 대상 구성")
    emit()
    emit("| 계열 | 모델 | 구현 | 입력 | 학습·추론 방식 | 시드 반복 |")
    emit("| :--- | :--- | :--- | :--- | :--- | :--- |")
    for r in ROSTER:
        emit("| " + " | ".join(r) + " |")
    emit()
    emit("신규 모델은 \"한 시계열의 과거에서 다음 값\"을 예측하는 라이브러리라, 예측 구간 H분 길이의 **블록 시계열**(블록 값 = 그 블록의 "
         "로그 RV_d, 26c의 타깃과 같음)로 입력을 만들었다. 입력이 15분봉 96개를 그대로 읽는 GRU·LSTM과 다르므로 순환 대 병렬 어텐션의 차이에는 "
         "입력 표현의 차이가 섞여 있다. 이 한계는 해석에서 다시 언급한다.")
    emit()
    emit("## 2. 파운데이션 모델 사전학습 시점과 평가 구간의 겹침 확인")
    emit()
    emit("zero-shot 모델의 가중치가 평가 구간(2025-11-11~) 데이터로 학습됐다면 결과가 부풀 수 있다. Hugging Face 저장소에서 가중치 파일이 "
         "마지막으로 바뀐 날짜를 조회했다(2026-10-06).")
    emit()
    try:
        wd = pd.read_csv(RES / f"{STEM}_fm_weight_dates.csv")
        emit("| 모델 | 저장소 | 가중치 마지막 변경일 | 평가 시작 전 여유(일) |")
        emit("| :--- | :--- | :--- | ---: |")
        for _, x in wd.iterrows():
            emit(f"| {x['모델']} | {x['Hugging Face 저장소']} | {x['가중치 마지막 변경일']} | {int(x['평가 시작 전 여유(일)'])} |")
        emit()
        emit(f"**해석**: 가중치가 평가 시작일보다 늦게 바뀐 모델은 {int((wd['평가 시작 전 여유(일)'] <= 0).sum())}개이고, 가장 가까운 TimesFM 2.5도 "
             f"{int(wd['평가 시작 전 여유(일)'].min())}일 앞선다. 곧 평가 구간 데이터가 가중치에 들어갈 수 없다. 단, 사전학습 데이터의 구체적 구성"
             "(업비트 데이터 포함 여부)은 공개되지 않아 확인하지 못했다. 암호화폐 일반 시계열은 포함됐을 수 있으나 평가 기간과는 겹치지 않는다.")
    except FileNotFoundError:
        emit("조회 결과 파일이 없다.")
    emit()
    cl, n_seed = seed_mean_losses(rd, store, models)
    cl.to_csv(RES / f"{STEM}_cell_losses_seedmean.csv", index=False)
    tt = M.tier_table(cl)
    cl0 = cl[(cl["구간"] == "전체") & (cl["기준"] == "보정후")].copy()
    cl0["rank"] = cl0.groupby(["종목", "H"])["QLIKE"].rank()
    tt.to_csv(RES / f"{STEM}_tier_table.csv", index=False)
    M.family_votes(tt).to_csv(RES / f"{STEM}_tier_family_votes.csv", index=False)
    emit("## 3. 구간별 결과(전 모델)")
    emit()
    emit("**이 절의 모든 손실은 시드 평균이다.** 무작위성이 있는 모델(트리, Nystroem, GRU·LSTM, 신규 신경망)은 시드마다 그 시드의 예측과 보정계수로 "
         "손실을 구한 뒤 평균했고(예측을 평균한 앙상블이 아니다), 시드와 무관한 모델(GARCH 3종, 커널 2종, 파운데이션 7종)은 같은 값이 평균된다. "
         "한 시드만 쓰면 신경망의 순위가 시드에 따라 바뀌기 때문이다(26c에서 GRU 15분 종목 평균 손실이 시드에 따라 -9.74~-9.81).")
    emit()
    emit("| 모델 | " + " | ".join(m_ for m_ in models if m_ != "naive") + " |")
    emit("| :--- | " + " | ".join(["---:"] * (len(models) - 1)) + " |")
    emit("| 사용한 시드 수 | " + " | ".join(str(n_seed[m_]) for m_ in models if m_ != "naive") + " |")
    emit()
    emit("시드 수가 1인 모델은 시드로 결과가 바뀌지 않는 모델(해당 없음)이거나 해당 시드 산출물이 아직 없는 모델이다. 후자라면 이 표의 값은 "
         "시드 1개 결과이므로 해석에 주의해야 한다.")
    emit()
    emit(f"종목마다 시드 평균 QLIKE로 순위를 매겨 평균했다(작을수록 좋음). 격차는 그 구간 최선 모델 대비 시드 평균 QLIKE 차의 종목 평균이다. "
         f"**A등급은 격차가 {M.TIE} 이하**로, GRU 시드 표준편차에서 가져온 실무 기준이며 통계 검정이 아니다. 통계 검정(DM·MCS)은 26b 보고서에 있다.")
    emit()
    for H in M.HORIZONS_H:
        g = tt[(tt["H"] == H) & (tt["구간"] == "전체") & (tt["기준"] == "보정후")].set_index("모델")
        r = cl0[cl0["H"] == H].groupby("모델")["rank"].mean()
        order = g.sort_values("격차").index
        emit(f"### {M.hlabel(H)}")
        emit()
        emit("| 순위 | 모델 | 계열 | 평균 순위(시드 평균 손실) | 격차(최선 대비) | 등급 | 종목 수 |")
        emit("| ---: | :--- | :--- | ---: | ---: | :--- | ---: |")
        for i, nm in enumerate(order, 1):
            emit(f"| {i} | {nm} | {M.FAMILY[nm]} | {r.get(nm, np.nan):.1f} | {g.loc[nm, '격차']:.4f} | {g.loc[nm, '등급']} | {int(g.loc[nm, '종목수'])} |")
        emit()
    emit("## 4. 계열별 최선과 격차")
    emit()
    emit("각 계열에서 그 구간 최선과의 격차가 가장 작은 모델을 대표로 삼았다. 값이 0.01 이하면 그 구간의 최선과 사실상 동률이다. 계열에 해당 모델이 "
         "없으면 \"-\"로 표시했다.")
    emit()
    fams = ["통계", "하이브리드", "트리", "딥러닝", "커널", "어텐션", "합성곱", "상태공간", "파운데이션"]
    emit("| 계열 | 모델 수 | " + " | ".join(M.hlabel(H) for H in M.HORIZONS_H) + " |")
    emit("| :--- | ---: | " + " | ".join(["---:"] * len(M.HORIZONS_H)) + " |")
    allg = tt[(tt["구간"] == "전체") & (tt["기준"] == "보정후") & (tt["모델"] != "naive")].copy()
    allg["계열"] = allg["모델"].map(M.FAMILY)
    for fam in fams:
        n_f = len([m_ for m_ in models if M.FAMILY.get(m_) == fam])
        cells = []
        for H in M.HORIZONS_H:
            x = allg[(allg["H"] == H) & (allg["계열"] == fam)]
            cells.append("-" if not len(x) else f"{x['격차'].min():.4f} ({x.loc[x['격차'].idxmin(), '모델']})")
        emit(f"| {M.FAM_LABEL.get(fam, fam)} | {n_f} | " + " | ".join(cells) + " |")
    emit()
    M.emit_profile(tt)
    emit("## 5. 신규 모델의 위치(보정후, 전체 구간)")
    emit()
    emit("신규 모델이 26c의 최선 계열과 얼마나 떨어져 있는지 구간별로 적는다. 격차가 양수면 최선보다 손실이 큰 것이다.")
    emit()
    emit("| 모델 | 계열 | " + " | ".join(M.hlabel(H) for H in M.HORIZONS_H) + " |")
    emit("| :--- | :--- | " + " | ".join(["---:"] * len(M.HORIZONS_H)) + " |")
    for nm in NEW_MODELS:
        cells = []
        for H in M.HORIZONS_H:
            x = allg[(allg["H"] == H) & (allg["모델"] == nm)]
            if nm == "TTM" and H in TTM_UNSUPPORTED_H:
                cells.append("제외(공식 지원 해상도 밖)")
            else:
                cells.append("미포함(산출물 없음)" if not len(x) else f"{x['격차'].iloc[0]:.4f} ({x['등급'].iloc[0]})")
        emit(f"| {nm} | {M.FAMILY[nm]} | " + " | ".join(cells) + " |")
    emit()
    emit("## 6. 사전 구간별·분기별 순위와 정지(RV=0) 분해")
    emit()
    rk, size = M.regime_tables(store, models)
    rk.to_csv(RES / f"{STEM}_exante_regime_ranks.csv", index=False)
    size.to_csv(RES / f"{STEM}_exante_regime_sizes.csv", index=False)
    M.quarter_table(store, models).to_csv(RES / f"{STEM}_quarter_ranks.csv", index=False)
    emit("사전 구간(직전 H시간 RV 5분위)별 순위와 달력 분기별 순위는 CSV로 저장했다(`exante_regime_ranks.csv`, `quarter_ranks.csv`). "
         "구간별 통계적 동률은 26b 보고서 3-2절에 있다.")
    emit()
    M.emit_zero_split(store, models)
    (RES / f"{STEM}_report.md").write_text("\n".join(M._LINES), encoding="utf-8")
    print(f"[보고서] {RES / (STEM + '_report.md')}", flush=True)


def merge_jobs(results: list, store: dict, rows: list, rows1: list, store1: dict) -> None:
    for r in results:
        key = (r["ticker"], r["H"])
        rows.extend(r["rows"]); rows1.extend(r["rows1"])
        for st, pk in ((store, "preds"), (store1, "preds1")):
            S = st.setdefault(key, {"preds": {}})
            S["preds"].update(r[pk])
            S.update(act=r["act"], T=r["T"])
        store[key]["pi"] = r["pi"]


def main(argv=None) -> None:
    from report_header import study_universe
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="nf", choices=["nf", "conv", "ssm", "fm-prep", "fm-score", "combine", "report", "pi"])
    ap.add_argument("--elapsed-note", default="")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--n-tickers", type=int, default=20)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--models", default="", help="콤마 목록(기본: NF 전부)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--retries", type=int, default=2, help="실패한 작업 재시도 횟수(워커 1개)")
    a = ap.parse_args(argv)
    selftest()
    M.selftest()
    if a.selftest:
        return
    global SEED, RUN_STEM
    if a.seed:
        os.environ["RUN27_SEED"] = str(a.seed)
        SEED, RUN_STEM = a.seed, f"{STEM}_seed{a.seed}"
    RES.mkdir(parents=True, exist_ok=True)
    tickers, _ = study_universe()
    tickers = tickers[:2] if a.quick else tickers[:a.n_tickers]
    if a.family == "combine":
        combine()
        return
    if a.family == "report":
        write_report27(a.elapsed_note)
        return
    default = {"nf": NF_MODELS, "conv": CONV_MODELS, "ssm": SSM_MODELS, "fm-score": FM_MODELS, "fm-prep": (), "pi": ()}[a.family]
    models = tuple(m_ for m_ in (a.models.split(",") if a.models else default) if m_)
    tag = {"nf": "nf", "conv": "conv", "ssm": "ssm", "fm-score": "fm", "fm-prep": "prep", "pi": "pi"}[a.family]
    if a.family == "nf" and len(models) == 1:
        tag = f"nf-{models[0]}"                    # 알고리즘 하나씩 순차 실행할 때 산출물을 모델별로 나눈다
    jobs = [(tk, H) for tk in tickers for H in M.HORIZONS_H]
    print(f"[시작] {a.family} {models} · 종목 {len(tickers)} × 구간 {M.HORIZONS_H} = {len(jobs)}작업 · 시드 {SEED} · 워커 {a.workers}",
          flush=True)
    t_start = time.time()
    results, fails, metas, zrows = [], [], [], []
    ctx = mp.get_context("spawn")
    if a.family == "pi":
        with ProcessPoolExecutor(max_workers=a.workers, mp_context=ctx) as pool:
            for i, name in enumerate(pool.map(pi_job, [j[0] for j in jobs], [j[1] for j in jobs], [a.quick] * len(jobs)), 1):
                print(f"  [π {i}/{len(jobs)}] {name}", flush=True)
        print(f"[정지 확률 캐시 완료] {(time.time() - t_start) / 60:.1f}분", flush=True)
        return
    if a.family == "fm-prep":
        with ProcessPoolExecutor(max_workers=a.workers, mp_context=ctx) as pool:
            for i, name in enumerate(pool.map(fm_prepare_job, [j[0] for j in jobs], [j[1] for j in jobs],
                                              [a.quick] * len(jobs)), 1):
                print(f"  [{i}/{len(jobs)}] {name}", flush=True)
        print(f"[입력 작성 완료] {(time.time() - t_start) / 60:.1f}분 · {FM_DIR / 'in'}", flush=True)
        return
    fn = run_fm_score_job if a.family == "fm-score" else run_nf_job
    extra = (models,) if a.family == "fm-score" else (models, SEED)
    # 작업 단위 저장·이어하기: 종목×구간 작업이 끝날 때마다 바로 저장하고, 다시 실행하면 끝난 작업은 읽기만 한다.
    # 모델 하나라도 실패한 작업은 완료로 치지 않고 다음 차례에 다시 돌린다(최대 a.retries번, 재시도는 워커 1개로).
    import pickle
    import json
    # 시험 실행(--quick)은 저장 폴더를 따로 쓰고, 설정 지문이 다르면 저장분을 재사용하지 않고 멈춘다(Codex 리뷰 2026-10-07).
    part_dir = RES / "parts" / f"{RUN_STEM}_{tag}{'_quick' if a.quick else ''}"
    part_dir.mkdir(parents=True, exist_ok=True)
    fp = run_fingerprint(a.quick, models, a.family)
    fp_file = part_dir / "config.json"
    if fp_file.exists():
        old = json.loads(fp_file.read_text())
        if old != fp:
            diff = {k: (old.get(k), fp.get(k)) for k in set(old) | set(fp) if old.get(k) != fp.get(k)}
            raise SystemExit(f"[중단] {part_dir.name}의 저장분은 다른 설정으로 만들어졌다: {diff}. 폴더를 옮기거나 지운 뒤 다시 실행하라")
    else:
        fp_file.write_text(json.dumps(fp, ensure_ascii=False, indent=1))
    part = lambda tk, H: part_dir / f"{tk}_{H}.pkl"
    done: dict = {}
    for tk, H in jobs:
        if part(tk, H).exists():
            done[(tk, H)] = pickle.loads(part(tk, H).read_bytes())
    print(f"[이어하기] 저장된 작업 {len(done)}/{len(jobs)}", flush=True)
    for attempt in range(a.retries + 1):
        todo = [j for j in jobs if j not in done or done[j]["fails"]]
        if not todo:
            break
        nw = a.workers if attempt == 0 else 1
        print(f"[{'실행' if attempt == 0 else f'재시도 {attempt}'}] {len(todo)}작업 · 워커 {nw}", flush=True)
        with ProcessPoolExecutor(max_workers=nw, mp_context=ctx) as pool:
            futs = {pool.submit(fn, tk, H, a.quick, *extra): (tk, H) for tk, H in todo}
            for i, fu in enumerate(as_completed(futs), 1):
                tk, H = futs[fu]
                try:
                    r = fu.result()
                except Exception as e:
                    print(f"    ! {tk} H={H} 작업 실패 {type(e).__name__}: {str(e)[:120]}", flush=True)
                    done.setdefault((tk, H), {"fails": [("(작업)", H, type(e).__name__, str(e)[:200])], "_crash": True})
                    continue
                done[(tk, H)] = r
                part(tk, H).write_bytes(pickle.dumps(r))
                print(f"  [{i}/{len(todo)}] {tk} H={H} ({r['elapsed']:.0f}s) 실패 모델 {len(r['fails'])}", flush=True)
    for (tk, H), r in done.items():
        if r.get("_crash"):
            fails.extend({"종목": tk, "모델": f[0], "H": f[1], "예외": f[2], "메시지": f[3]} for f in r["fails"])
            continue
        results.append(r)
        metas.append(r["meta"]); zrows.append(r["zinfo"])
        fails.extend({"종목": tk, "모델": f[0], "H": f[1], "예외": f[2], "메시지": f[3]} for f in r["fails"])
    store, store1, rows, rows1 = {}, {}, [], []
    merge_jobs(results, store, rows, rows1, store1)
    pd.DataFrame(rows).to_csv(RES / f"{RUN_STEM}_{tag}_model_comparison.csv", index=False)
    pd.DataFrame(rows1).to_csv(RES / f"{RUN_STEM}_{tag}_onepart_comparison.csv", index=False)
    pd.DataFrame(fails, columns=["종목", "모델", "H", "예외", "메시지"]).to_csv(RES / f"{RUN_STEM}_{tag}_fit_failures.csv", index=False)
    pd.DataFrame(metas).to_csv(RES / f"{RUN_STEM}_{tag}_meta.csv", index=False)
    M.save_npz(RES / f"{RUN_STEM}_{tag}_test_predictions.npz", store, aux=True)
    M.save_npz(RES / f"{RUN_STEM}_{tag}_onepart_predictions.npz", store1, aux=False)
    pd.DataFrame(zrows).to_csv(RES / f"{RUN_STEM}_{tag}_zero_classifier.csv", index=False)
    print(f"[{a.family} 완료] {(time.time() - t_start) / 60:.1f}분 · {len(rows)}행 · 실패 {len(fails)}", flush=True)


if __name__ == "__main__":
    main()
