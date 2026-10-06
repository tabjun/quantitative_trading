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
FM_MODELS = ("Chronos-Bolt", "TimesFM", "TTM", "Moirai-2", "Sundial", "Time-MoE", "Lag-Llama")
NEW_MODELS = NF_MODELS + CONV_MODELS + FM_MODELS

NEW_FAMILY = {"PatchTST": "어텐션", "iTransformer": "어텐션", "Autoformer": "어텐션", "TimeXer": "어텐션",
              "TCN": "합성곱", "TimesNet": "합성곱", "ModernTCN": "합성곱", **{m_: "파운데이션" for m_ in FM_MODELS}}
NEW_PROCESS = {"PatchTST": "병렬 어텐션(패치)", "iTransformer": "병렬 어텐션(변수 토큰)", "Autoformer": "병렬 어텐션(자기상관)",
               "TimeXer": "병렬 어텐션(패치+변수 토큰)", "TCN": "병렬 합성곱(인과 팽창)", "TimesNet": "병렬 합성곱(2D 주기)",
               "ModernTCN": "병렬 합성곱(순수)", "Chronos-Bolt": "사전학습 zero-shot(인코더-디코더)",
               "TimesFM": "사전학습 zero-shot(디코더)", "TTM": "사전학습 zero-shot(MLP-Mixer)",
               "Moirai-2": "사전학습 zero-shot(디코더)", "Sundial": "사전학습 zero-shot(플로우 매칭)",
               "Time-MoE": "사전학습 zero-shot(전문가 혼합)", "Lag-Llama": "사전학습 zero-shot(lag 디코더)"}
M.FAMILY.update(NEW_FAMILY)
M.PROCESS.update(NEW_PROCESS)
M.FAM_ORDER[:] = ["통계", "하이브리드", "트리", "딥러닝", "커널", "어텐션", "합성곱", "파운데이션"]
M.FAM_COLOR.update({"어텐션": "#4a3aa7", "합성곱": "#e87ba4", "파운데이션": "#6b6b66"})
M.FAM_MARK.update({"어텐션": "X", "합성곱": "v", "파운데이션": "*"})
M.FAM_LABEL.update({"어텐션": "병렬 어텐션", "합성곱": "병렬 합성곱", "파운데이션": "파운데이션(zero-shot)"})
M.SHORT.update({"PatchTST": "PTST", "iTransformer": "iTrans", "Autoformer": "Autof", "TimesNet": "TNet",
                "ModernTCN": "MTCN", "Chronos-Bolt": "Chr", "TimesFM": "TFM", "Moirai-2": "Moi",
                "Sundial": "Sun", "Time-MoE": "TMoE", "Lag-Llama": "LagL"})

INPUT_BLOCKS = {15: 96, 30: 96, 60: 96, 240: 60, 720: 30}   # 입력 길이(블록 수): 하루~며칠치 이력
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


def nf_predict(name: str, B: dict, ymu: float, ysd: float, train_end: int, end: int, H: int, seed: int, quick: bool):
    if name == "ModernTCN":
        return modern_tcn_predict(B, ymu, ysd, train_end, end, H, seed, quick)
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
    pi_va, pi_te, zinfo = M.zero_model(D, HD, M.make_features(D), M.zero_features(D), quick)
    S.set_zero(pi_va, pi_te)
    ymu, ysd = standardize(B, MK["k_in"])
    return dict(D=D, HD=HD, B=B, MK=MK, S=S, ymu=ymu, ysd=ysd, zinfo={"종목": ticker, "H": H, **zinfo})


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
FM_PY = {"Chronos-Bolt": ".venv", "TimesFM": ".venv", "TTM": ".venv", "Moirai-2": ".venvs/moirai_py312_20261004_192438",
         "Sundial": ".venvs/legacy_hf_py312_20261004", "Time-MoE": ".venvs/legacy_hf_py312_20261004",
         "Lag-Llama": ".venvs/lagllama_py312_20261004"}
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
    ap.add_argument("--family", default="nf", choices=["nf", "conv", "fm-prep", "fm-score"])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--n-tickers", type=int, default=20)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--models", default="", help="콤마 목록(기본: NF 전부)")
    ap.add_argument("--seed", type=int, default=0)
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
    default = {"nf": NF_MODELS, "conv": CONV_MODELS, "fm-score": FM_MODELS, "fm-prep": ()}[a.family]
    models = tuple(m_ for m_ in (a.models.split(",") if a.models else default) if m_)
    tag = {"nf": "nf", "conv": "conv", "fm-score": "fm", "fm-prep": "prep"}[a.family]
    jobs = [(tk, H) for tk in tickers for H in M.HORIZONS_H]
    print(f"[시작] {a.family} {models} · 종목 {len(tickers)} × 구간 {M.HORIZONS_H} = {len(jobs)}작업 · 시드 {SEED} · 워커 {a.workers}",
          flush=True)
    t_start = time.time()
    results, fails, metas, zrows = [], [], [], []
    ctx = mp.get_context("spawn")
    if a.family == "fm-prep":
        with ProcessPoolExecutor(max_workers=a.workers, mp_context=ctx) as pool:
            for i, name in enumerate(pool.map(fm_prepare_job, [j[0] for j in jobs], [j[1] for j in jobs],
                                              [a.quick] * len(jobs)), 1):
                print(f"  [{i}/{len(jobs)}] {name}", flush=True)
        print(f"[입력 작성 완료] {(time.time() - t_start) / 60:.1f}분 · {FM_DIR / 'in'}", flush=True)
        return
    fn = run_fm_score_job if a.family == "fm-score" else run_nf_job
    extra = (models,) if a.family == "fm-score" else (models, SEED)
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=ctx) as pool:
        futs = {pool.submit(fn, tk, H, a.quick, *extra): (tk, H) for tk, H in jobs}
        for i, fu in enumerate(as_completed(futs), 1):
            tk, H = futs[fu]
            try:
                r = fu.result()
            except Exception as e:
                fails.append({"종목": tk, "모델": "(작업)", "H": H, "예외": type(e).__name__, "메시지": str(e)[:200]})
                print(f"    ! {tk} H={H} 작업 실패 {type(e).__name__}: {str(e)[:120]}", flush=True)
                continue
            results.append(r)
            metas.append(r["meta"]); zrows.append(r["zinfo"])
            fails.extend({"종목": tk, "모델": f[0], "H": f[1], "예외": f[2], "메시지": f[3]} for f in r["fails"])
            print(f"  [{i}/{len(jobs)}] {tk} H={H} ({r['elapsed']:.0f}s)", flush=True)
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
