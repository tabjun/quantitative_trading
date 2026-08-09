# %% [markdown]
# # 16번: 비정상성 정규화 + 큰 변동(분위/분포) 손실 + KRW 전 종목 cross-sectional
#
# 이 파일은 **헤드리스 `.py` 드라이버**다(사용자 지시 2026-07-19: ipynb는 만들지 않고 `.py`만,
# `#%%`로 셀을 구분하고 파일 내 마크다운·주석으로 설명한다). 실행과 결과 저장은 이 파일이 담당한다.
#
# - 계획서: `test/experiment_specs/16_nonstationary_crosssectional_plan_20260719.md`
# - 방법론 근거(문헌): `test/research_materials/16_nonstationary_trend_capture_literature_review_20260719.md`
# - 연구 초고(1~15 종합): `test/results/paper_draft_nonstationary_crypto_trend_20260719.md`
#
# ## 왜 (Why)
# 15번까지 확인된 벽: (1) 평균형 손실(MSE/Huber)은 15분 수익률의 신호 대 잡음비가 0에 가까워
# 조건부 평균(≈0)으로 수렴 → **진폭 압축**(variance_ratio 0.36), (2) 과한 정상화가 급변 정보를
# 지움, (3) 약한 신호가 BTC 단일 종목 우연이라 일반화 안 됨(T8). 하나만 바꾸면 다시 국소 최적에
# 갇히므로(15번 교훈), 16번은 **정규화·손실·데이터축 세 가지를 함께** 바꾼다.
#
# ## 무엇을 (What) — 5단계
# - T1 t1_normalization: 정규화 {window_standard, revin, dishts_lite, none} 비교 (Q1 비정상성)
# - T2 t2_loss: 손실 {huber, pinball(다중분위), student_t, tail_weighted} 비교 (Q2 큰 변동)
# - T3 t3_crosssection: 학습 축 {single, pooled_ci, pooled_cd} — KRW 전 종목 pooled (Q3 일반화)
# - T4 t4_ranking: objective {regression, cross_sectional_rank} (Q3 상대 강도)
# - T5 t5_policy: risk gate {none, quantile} — 변동 재현 유지하며 방어 (Q4 생존 제약)
#
# ## 어떻게 (How)
# 공용 엔진 `engine/`(데이터·윈도우·모델 backbone·자원)을 재사용하되, 16번 고유 컴포넌트
# (RevIN, 다중출력 head, 분위/분포/tail 손실, cross-sectional batcher, tail P/R/F1·IC 지표)는
# 이 드라이버 안에 새로 둔다(엔진 원본은 read-only). 13번 crash 교훈으로 헤드리스·num_workers=0·
# per-case 그림 off·OOM 자동 배치 축소를 유지한다.
#
# ## 기대 결과 / 반영 (Expected)
# variance_ratio가 1에 가까워지고(진폭 회복) 큰 변동 tail F1·trend_corr가 huber 대비 오르며,
# cross-sectional IC가 종목 가로질러 양수면 "전 종목 상대강도" 축을 데이터마트로 승격한다.
# 실패해도(여전히 큰 변동 못 잡음) horizon·손실별로 정직하게 기록한다. MDD는 생존 제약으로만
# 병기하고 단독 랭킹하지 않는다(2026-06-24 이탈 재발 방지, CLAUDE.md 2.12).

# %%
"""16번 드라이버 본체. 실행 예시(서버):

    uv run test/models/16_nonstationary_crosssectional_test.py --suite t1_normalization --dry-run
    uv run test/models/16_nonstationary_crosssectional_test.py --suite t1_normalization
    uv run test/models/16_nonstationary_crosssectional_test.py --suite t2_loss \
        --normalization revin --losses huber,pinball,student_t,tail_weighted
    uv run test/models/16_nonstationary_crosssectional_test.py --suite t3_crosssection \
        --normalization revin --loss pinball --n-tickers 30
    uv run test/models/16_nonstationary_crosssectional_test.py --suite t5_policy \
        --normalization revin --loss pinball --n-tickers 30

참고문헌(문헌 리뷰에서 실재 확인): RevIN(ICLR22), Non-stationary Transformer(arXiv:2205.14415),
Dish-TS(arXiv:2302.14829), DeepAR(arXiv:1704.04110), pinball/quantile(Koenker-Bassett 1978),
tail-weighted loss(arXiv:2112.00825), PatchTST-CI(arXiv:2211.14730), iTransformer(arXiv:2310.06625),
RSR cross-sectional ranking(arXiv:1809.09441), Informer HF Bitcoin 0-붕괴 실증(arXiv:2503.18096).
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = ["Noto Sans CJK KR", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


# %% [markdown]
# ## 1. 엔진 import 계약
# quantitative_trading을 sys.path 루트로 두고 top-level `import engine` (repo 관례).

# %%
def _engine_root(start: Path) -> Path:
    candidates = [start, *start.parents, Path.home() / "personal_ai_project" / "quantitative_trading"]
    for candidate in candidates:
        if (candidate / "pyproject.toml").exists() and (candidate / "engine").is_dir():
            return candidate
    raise RuntimeError("engine을 담은 quantitative_trading 디렉터리를 찾지 못했다.")


try:
    _START = Path(__file__).resolve().parent
except NameError:  # 노트북/REPL 경로
    _START = Path.cwd()
ROOT = _engine_root(_START)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine import data as engdata
from engine import models as engmodels
from engine import resources as engres
from engine import windows as engwin

EXPERIMENT_TAG = "16_nonstationary_crosssectional_20260719"
RESULTS_DIR = ROOT / "test" / "results" / EXPERIMENT_TAG
IMAGES_DIR = ROOT / "test" / "images" / EXPERIMENT_TAG

# 데이터 기본 설정 (연구 기본 가정)
KRW_TABLE = "upbit_krw_candle"      # 전 종목 축(원설계). 269종목/1,608만행(2026-07-19 재수집)
BTC_TABLE = "btc_15m_advance"       # 단일종목 진단용(빠른 비교)
# multi-timeframe 변수셋(12·14·15에서 방어·추세 신호원으로 확인된 mtf_trend 포함)
FEATURE_COLUMNS = [
    "log_return_1", "return_4", "return_16", "return_64", "return_192",
    "realized_vol_16", "realized_vol_64", "realized_vol_192",
    "vol_ratio_16_64", "vol_ratio_64_192",
    "price_z_192", "volume_z_192", "value_z_192", "range_mean_192",
    "trend_strength_64", "trend_strength_192",
]
QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)   # pinball 손실용 분위(중앙 0.5 = 점예측)


# %% [markdown]
# ## 2. 판독 가이드 (각 suite/지표가 무엇을 왜 보는가 — raw md 상단 자동 출력)
# 15번에서 도입한 SUITE_INTENT/METRIC_GLOSSARY 관례를 계승한다(사용자 요청: 결과 단에 설명 필수).

# %%
SUITE_INTENT: dict[str, dict[str, str]] = {
    "t1_normalization": {
        "name": "비정상성 정규화 비교",
        "fix": "손실=huber, 모델=ITransformer/PatchTST, BTC 단일축(빠른 진단), h=16.",
        "vary": "정규화 {window_standard, revin, dishts_lite, none}.",
        "question": "정규화가 지운 급변 정보를 되살려 variance_ratio를 1에 가깝게 올리는가(과적합 악화 없이)?",
        "read": "variance_ratio가 1에 가깝고 val 곡선이 안정하면 그 정규화가 비정상성을 잘 처리.",
    },
    "t2_loss": {
        "name": "큰 변동 손실함수 비교",
        "fix": "우승 정규화, 모델, BTC 단일축, h=16.",
        "vary": "손실 {huber, pinball(다중분위), student_t(분포), tail_weighted}.",
        "question": "평균형 huber를 분위·분포·tail 손실로 바꾸면 큰 변동(tail)을 더 잘 잡는가?",
        "read": "tail_f1·trend_corr가 오르고 variance_ratio가 1에 가까워지면 큰 변동 포착 개선.",
    },
    "t3_crosssection": {
        "name": "KRW 전 종목 pooled 학습",
        "fix": "우승 정규화·손실, h=16.",
        "vary": "학습 축 {single(BTC), pooled_ci(전종목 채널독립), pooled_cd}.",
        "question": "전 종목을 함께 학습하면 신호가 종목 가로질러 일반화되는가(BTC 우연 탈피)?",
        "read": "pooled 모델의 종목별 trend_corr가 고루 양수면 구조적 신호.",
    },
    "t4_ranking": {
        "name": "cross-sectional ranking objective",
        "fix": "우승 정규화·손실, 전 종목 pooled.",
        "vary": "objective {regression(절대 수익률), rank(같은 시점 종목 간 상대강도)}.",
        "question": "절대 예측이 안 되어도 상대 순위(cross-sectional IC)는 신호를 남기는가?",
        "read": "cross_sectional_ic가 양수로 유지되면 상대강도 축이 유효.",
    },
    "t5_policy": {
        "name": "생존 제약(risk gate) 평가",
        "fix": "우승 예측(변동 재현 유지).",
        "vary": "risk gate {none, quantile(하위 분위 위험 시 차단)}.",
        "question": "변동을 죽이지 않으면서(variance_ratio·active_share 유지) MDD를 방어하는가?",
        "read": "MDD가 개선되되 variance_ratio/active_share가 함께 유지돼야 진짜 방어(MDD 단독 랭킹 금지).",
    },
}

METRIC_GLOSSARY: list[tuple[str, str]] = [
    ("trend_corr", "예측 vs 실제 h-step 수익률 Pearson 상관. 0=무상관. 금융에선 0.1도 유의미."),
    ("variance_ratio", "예측분산/실제분산. 1이 이상적(실제만큼 출렁임), ≪0.1 평탄화, ≫20 폭주. '진폭' 지표."),
    ("large_move_da", "|실제| 상위 25%(큰 변동) 구간 방향 정확도. 0.5=동전던지기. 큰 변동 포착 직접 지표."),
    ("tail_precision", "큰 변동을 '예측이 큰 변동이라 본 것' 중 실제 큰 변동 비율(정밀도)."),
    ("tail_recall", "실제 큰 변동 중 예측이 잡아낸 비율(재현율). 작은 폭만 예측하면 여기서 낮게 나옴."),
    ("tail_f1", "tail 정밀도·재현율의 조화평균. 큰 변동 포착의 종합 지표(정확도 대신 이걸 본다)."),
    ("mase_momentum", "예측 오차/'직전 h봉 추세 지속' 오차. <1이면 추세지속 기준선보다 우수."),
    ("copy_risk_krw", "KRW MAE/persistence MAE. <1이면 직전값 복사보다 우수."),
    ("cross_sectional_ic", "같은 시점 종목 간 예측-실제 순위상관(Spearman) 평균. 상대강도 신호 지표."),
    ("val_overfit_gap", "best val_loss 대비 마지막 val_loss 상승분. 과적합 진단(양수 크면 과적합)."),
    ("healthy_variance", "variance_ratio가 0.05~20이면 True(평탄화/폭주 아님)."),
]


# %% [markdown]
# ## 3. 신규 컴포넌트 A — RevIN (Reversible Instance Normalization, ICLR22)
# 각 입력 윈도우(instance)의 평균·분산을 제거해 backbone에 넣고, 출력 스케일을 복원한다.
# window_standard(15번)는 복원 단계·학습 affine이 없는 경량판이었다. 여기서는 정식 RevIN을 둔다.
# dishts_lite: 입력 통계뿐 아니라 "예측 구간 분포가 다를 수 있다"를 반영해 출력 스케일을 별도
# 학습 계수로 조정하는 Dish-TS(AAAI23)의 경량 근사.

# %%
class RevIN(nn.Module):
    """instance 정규화-역정규화 + 학습 affine(γ, β). mode: 'revin' | 'dishts_lite' | 'none'."""

    def __init__(self, n_features: int, mode: str = "revin", eps: float = 1e-5):
        super().__init__()
        self.mode = mode
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(n_features))
        self.beta = nn.Parameter(torch.zeros(n_features))
        # dishts_lite: 출력 스케일을 입력 표준편차에 곱하는 학습 계수(예측구간 분포 이동 근사)
        self.out_scale = nn.Parameter(torch.ones(1)) if mode == "dishts_lite" else None
        self._mean = None
        self._std = None

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "none":
            return x
        self._mean = x.mean(dim=1, keepdim=True)
        self._std = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps)
        x = (x - self._mean) / self._std
        return x * self.gamma + self.beta

    def input_scale(self, feature_index: int = 0) -> torch.Tensor | float:
        """마지막 normalize에서 쓴 (타깃 채널) 윈도우 std. dishts_lite면 학습 계수를 곱한다.

        RevIN 정식 사용에서 출력 복원은 '입력에서 뺀 통계를 그대로 되돌리는' 것이지만, 여기서
        타깃은 입력 feature 자체가 아니라 h-step 누적수익률이라 입력 std로 강제 복원하면 스케일이
        어긋난다(스모크에서 student_t variance_ratio 29 폭주로 확인). 그래서 출력 head는 원 스케일
        타깃을 직접 학습하고(denorm 제거), RevIN은 '입력 비정상성 제거' 역할만 한다. 이 메서드는
        dishts_lite의 출력 스케일 계수만 노출해, 필요 시 head 뒤에 곱할 수 있게 남겨 둔다.
        """
        if self.out_scale is not None:
            return self.out_scale
        return 1.0


# %% [markdown]
# ## 4. 신규 컴포넌트 B — 다중출력 backbone 래퍼
# 엔진의 backbone(ITransformerLike 등)은 마지막 `.head = Linear(hidden, 1)` 하나뿐이다.
# 그 head를 출력 차원 out_dim으로 교체하고, 앞단에 RevIN을 붙인다. out_dim은 손실이 결정한다:
# huber/tail=1(점), pinball=len(QUANTILES), student_t=3(μ,log σ,log ν).

# %%
class WrappedForecaster(nn.Module):
    """엔진 backbone(스칼라 (B,) 출력) 위에 RevIN(앞)과 다중출력 head(뒤)를 얹는다.

    엔진 backbone들의 forward는 마지막에 `head(feat)`를 호출한 뒤 `.squeeze(-1)`을 하드코딩한다.
    head를 `hidden -> out_dim`으로 교체하면: out_dim>1이면 squeeze(-1)가 크기1 축이 없어 그대로
    (B, out_dim)을 내고, out_dim==1이면 (B, 1)->(B,)가 된다. 그래서 wrapper는 backbone 출력이
    1차원이면 unsqueeze만 해 (B, out_dim)로 통일한다(표현력 병목 없이 head가 직접 out_dim 예측).
    """

    def __init__(self, backbone: nn.Module, n_features: int, out_dim: int, norm_mode: str):
        super().__init__()
        self.revin = RevIN(n_features, norm_mode)
        self.backbone = backbone
        head = getattr(backbone, "head", None)
        if not isinstance(head, nn.Linear):
            raise ValueError(f"{type(backbone).__name__}에 교체할 nn.Linear head가 없다.")
        backbone.head = nn.Linear(head.in_features, out_dim)   # hidden -> out_dim 직접
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.revin.normalize(x)                 # 입력 비정상성 제거(RevIN 핵심 역할)
        y = self.backbone(x)                        # out_dim>1: (B,out_dim), out_dim==1: (B,)
        if y.dim() == 1:
            y = y.unsqueeze(-1)                     # (B,) -> (B,1)
        # 출력 head는 원 스케일 h-step 타깃을 직접 학습한다(denorm 제거 — 스케일 이중적용 방지).
        # dishts_lite면 예측구간 분포 이동 근사를 위해 학습 스케일 계수를 곱한다.
        y = y * self.revin.input_scale()
        if self.out_dim == 1:
            return y.squeeze(-1)
        return y


def make_wrapped(name: str, seq_len: int, n_features: int, hidden: int, out_dim: int, norm_mode: str) -> nn.Module:
    backbone = engmodels.make_model(name, seq_len, n_features, hidden)
    return WrappedForecaster(backbone, n_features, out_dim, norm_mode)


# %% [markdown]
# ## 5. 신규 컴포넌트 C — 손실함수 (큰 변동 명시 모델링)
# - pinball(분위): 각 분위 τ에서 비대칭 벌점. 여러 분위를 함께 학습해 분포의 꼬리를 모델링 →
#   큰 변동을 눌러버리지 않는다(진폭 압축 해소). 점예측은 0.5 분위.
# - student_t NLL(분포): DeepAR 계열. 두꺼운 꼬리(첨도 100)를 ν(자유도)로 흡수.
# - tail_weighted: huber에 |target| 크기 비례 가중 → 큰 변동 표본의 오차를 더 크게 벌점.

# %%
def pinball_loss(pred_q: torch.Tensor, target: torch.Tensor, quantiles=QUANTILES) -> torch.Tensor:
    losses = []
    for i, tau in enumerate(quantiles):
        err = target - pred_q[:, i]
        losses.append(torch.maximum(tau * err, (tau - 1.0) * err))
    return torch.stack(losses, dim=1).mean()


def student_t_nll(params: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mu = params[:, 0]
    sigma = F.softplus(params[:, 1]) + 1e-4
    nu = F.softplus(params[:, 2]) + 2.0            # ν>2 라야 분산 유한
    z = (target - mu) / sigma
    log_prob = (
        torch.lgamma((nu + 1.0) / 2.0) - torch.lgamma(nu / 2.0)
        - 0.5 * torch.log(nu * np.pi) - torch.log(sigma)
        - (nu + 1.0) / 2.0 * torch.log1p(z * z / nu)
    )
    return -log_prob.mean()


def tail_weighted_huber(pred: torch.Tensor, target: torch.Tensor, beta: float = 0.001) -> torch.Tensor:
    huber = F.smooth_l1_loss(pred, target, beta=beta, reduction="none")
    scale = target.detach().abs().median().clamp_min(1e-5)
    weight = 1.0 + 3.0 * torch.clamp(target.detach().abs() / (3.0 * scale), 0.0, 4.0)
    return (weight * huber).mean()


def loss_out_dim(loss_name: str) -> int:
    return {"huber": 1, "tail_weighted": 1, "pinball": len(QUANTILES), "student_t": 3}[loss_name]


def compute_loss(loss_name: str, out: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if loss_name == "huber":
        return F.smooth_l1_loss(out.squeeze(-1) if out.dim() > 1 else out, target, beta=0.001)
    if loss_name == "tail_weighted":
        return tail_weighted_huber(out.squeeze(-1) if out.dim() > 1 else out, target)
    if loss_name == "pinball":
        return pinball_loss(out, target)
    if loss_name == "student_t":
        return student_t_nll(out, target)
    raise ValueError(f"지원하지 않는 손실: {loss_name}")


def point_from_output(loss_name: str, out: np.ndarray) -> np.ndarray:
    """모델 출력에서 '점예측'(중앙값/평균)을 뽑는다. 평가·정책은 이 점예측으로 한다."""
    if out.ndim == 1:
        return out
    if loss_name == "pinball":
        return out[:, len(QUANTILES) // 2]      # 0.5 분위
    if loss_name == "student_t":
        return out[:, 0]                        # μ
    return out[:, 0]


# %% [markdown]
# ## 6. 신규 컴포넌트 D — 지표 (큰 변동을 정밀도·재현율로 본다 + cross-sectional IC)
# 사용자 핵심: "작은 폭만 맞추는 건 정확한 게 아니다. 정확도 대신 정밀도·재현율처럼."
# → 큰 변동 구간을 이진 분류로 보고(실제 |수익률| 상위 25% = 양성), 예측 |수익률|이 같은 임계
# 이상이면 '큰 변동이라 예측'한 것으로 간주해 P/R/F1을 계산한다.

# %%
def pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def tail_precision_recall_f1(actual: np.ndarray, pred: np.ndarray, q: float = 0.75) -> dict[str, float]:
    a_thr = np.quantile(np.abs(actual), q)
    p_thr = np.quantile(np.abs(pred), q)
    actual_big = np.abs(actual) >= a_thr
    pred_big = np.abs(pred) >= p_thr
    tp = float(np.sum(actual_big & pred_big))
    fp = float(np.sum(~actual_big & pred_big))
    fn = float(np.sum(actual_big & ~pred_big))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"tail_precision": precision, "tail_recall": recall, "tail_f1": f1}


def trend_metrics(actual: np.ndarray, pred: np.ndarray, past: np.ndarray,
                  prev_close: np.ndarray, target_close: np.ndarray) -> dict[str, float]:
    actual = actual.astype(np.float64); pred = pred.astype(np.float64)
    error = np.abs(pred - actual)
    mae = float(np.mean(error))
    mae_zero = float(np.mean(np.abs(actual)))
    mae_mom = float(np.mean(np.abs(past.astype(np.float64) - actual)))
    a_std, p_std = float(np.std(actual)), float(np.std(pred))
    var_ratio = float((p_std ** 2) / (a_std ** 2 + 1e-18))
    large = np.abs(actual) >= np.quantile(np.abs(actual), 0.75)
    pred_close = prev_close.astype(np.float64) * np.exp(pred)
    mae_krw = float(np.mean(np.abs(pred_close - target_close.astype(np.float64))))
    persist_krw = float(np.mean(np.abs(prev_close.astype(np.float64) - target_close.astype(np.float64))))
    metrics = {
        "mae_return": mae,
        "mae_zero_ratio": mae / max(mae_zero, 1e-12),
        "mase_momentum": mae / max(mae_mom, 1e-12),
        "trend_corr": pearson(pred, actual),
        "direction_accuracy": float(np.mean((pred > 0) == (actual > 0))),
        "large_move_da": float(np.mean((pred[large] > 0) == (actual[large] > 0))),
        "variance_ratio": var_ratio,
        "pred_return_std": p_std, "actual_return_std": a_std,
        "copy_risk_krw": mae_krw / max(persist_krw, 1e-9),
        "healthy_variance": bool(0.05 <= var_ratio <= 20.0),
    }
    metrics.update(tail_precision_recall_f1(actual, pred))
    return metrics


# ---------------------------------------------------------------------------
# 코드 게이트: 챔피언 선정을 연구 목적으로 고정 (known_pitfalls P1/P2/P4).
# huber 등 평균형 손실이 진폭을 죽이면(variance_ratio 건강구간 밖) 자동 탈락시키고,
# 큰 변동 포착(tail_f1·large_move_da)과 진폭 건전성(|variance_ratio-1| 작을수록)을 함께 점수화한다.
# 단일 지표 최고로 뽑던 승계 오류(2026-07-19 T2: tail_f1 최고→huber 재선택 방향이탈)를 코드로 막는다.
# ---------------------------------------------------------------------------

# 실행 중 발견한 이슈/수정/재실행 사유를 전량 모아 결과 raw md에 남긴다(known_pitfalls P5).
RUN_ISSUES: list[str] = []


def log_issue(message: str) -> None:
    RUN_ISSUES.append(message)
    print(f"[issue] {message}")


def champion_score(row: dict) -> float:
    """연구 목적 기준 종합 점수(높을수록 우승). 진폭이 죽었으면 큰 폭으로 감점."""
    tail_f1 = float(row.get("tail_f1", 0.0) or 0.0)
    large_da = float(row.get("large_move_da", 0.5) or 0.5)
    vr = float(row.get("variance_ratio", 0.0) or 0.0)
    amp_gap = abs(np.log(max(vr, 1e-6)) - 0.0)          # variance_ratio=1이면 0, 멀수록 큼
    score = tail_f1 + (large_da - 0.5) - 0.15 * amp_gap
    if not (0.05 <= vr <= 20.0):                          # P1/P2: 평탄화·폭주면 후보 탈락
        score -= 1.0
    return score


def select_champion(rows: list[dict], label: str) -> dict:
    """rows 중 champion_score 최고를 우승으로. 탈락 사유를 이슈 로그에 남긴다."""
    if not rows:
        raise ValueError("우승 후보가 없다.")
    scored = sorted(rows, key=champion_score, reverse=True)
    best = scored[0]
    dropped = [r for r in rows if not (0.05 <= float(r.get("variance_ratio", 0) or 0) <= 20.0)]
    if dropped:
        names = ", ".join(str(r.get("loss") or r.get("normalization") or r.get("ticker")) for r in dropped)
        log_issue(f"{label}: 진폭 비건강(variance_ratio 구간 밖)으로 감점된 후보 {len(dropped)}개 [{names}]")
    log_issue(f"{label} 우승 선정: {best.get('loss') or best.get('normalization')} "
              f"(tail_f1={best.get('tail_f1'):.3f}, large_move_da={best.get('large_move_da'):.3f}, "
              f"variance_ratio={best.get('variance_ratio'):.3f}, score={champion_score(best):.3f})")
    return best


def cross_sectional_ic(pred_by_ts: dict[str, list[tuple[float, float]]]) -> float:
    """같은 timestamp에서 종목 간 (예측, 실제)의 Spearman 순위상관 평균 = IC."""
    from scipy.stats import spearmanr
    ics = []
    for _, pairs in pred_by_ts.items():
        if len(pairs) < 3:
            continue
        preds = [p for p, _ in pairs]; reals = [r for _, r in pairs]
        rho, _ = spearmanr(preds, reals)
        if np.isfinite(rho):
            ics.append(rho)
    return float(np.mean(ics)) if ics else float("nan")


# %% [markdown]
# ## 7. 윈도우 준비 (엔진 재사용) + momentum naive 기준선
# 엔진 build_risk_windows가 h-step 누적수익 경로·decision_timestamp를 제공한다(15번과 동일 재사용).
# target은 h-step 누적 로그수익률(future_return).

# %%
# 정규화 경로 분리(핵심 설계):
#  - WRAPPER_NORMS: 학습형 RevIN 계열은 wrapper(WrappedForecaster.RevIN)가 처리하고 엔진 윈도우
#    정규화는 'none'으로 둔다. 정규화가 wrapper 한 곳에서만 일어난다.
#  - 그 외(window_standard/window_robust/asinh_revin/none): 15번에서 검증된 엔진 윈도우 정규화
#    경로를 그대로 쓰고 wrapper RevIN은 통과(mode='none')시킨다.
# 이렇게 나눠야 정규화가 이중 적용되거나(스케일 폭주) window_standard가 무시되는 문제가 없다.
WRAPPER_NORMS = {"revin", "dishts_lite"}


def engine_norm_for(normalization: str) -> str:
    """엔진 build_risk_windows에 넘길 정규화. wrapper가 맡는 모드면 'none'."""
    return "none" if normalization in WRAPPER_NORMS else normalization


def wrapper_norm_for(normalization: str) -> str:
    """WrappedForecaster.RevIN에 넘길 모드. 엔진이 맡는 모드면 'none'."""
    return normalization if normalization in WRAPPER_NORMS else "none"


def make_windows_for_ticker(features: pd.DataFrame, seq_len: int, horizon: int,
                            preprocessing: str, normalization_input: str,
                            max_windows: int | None, stride: int) -> dict[str, np.ndarray] | None:
    cols = [c for c in FEATURE_COLUMNS if c in features.columns]
    if len(cols) < 6:
        return None
    data = engwin.build_risk_windows(
        features, cols, seq_len, horizon, preprocessing, engine_norm_for(normalization_input), max_windows, stride
    )
    log_close = np.log(features["close"].astype(float).clip(lower=1e-9)).to_numpy()
    ts_to_idx = {ts: i for i, ts in enumerate(features["timestamp"].astype(str).to_numpy())}
    past = np.zeros(len(data["y"]), dtype=np.float32)
    for i, ts in enumerate(data["decision_timestamp"]):
        idx = ts_to_idx.get(str(ts))
        if idx is not None and idx - horizon >= 0:
            past[i] = log_close[idx] - log_close[idx - horizon]
    data["past_return"] = past
    return data


# %% [markdown]
# ## 8. 학습 루프 (과적합 진단 포함)
# val_loss best 대비 마지막 상승분(val_overfit_gap)을 기록해 과적합을 진단한다(연구 목적 ① 건전성).
# OOM 시 배치 절반 재시도(13번 crash 회피). num_workers=0.

# %%
def train_one(model: nn.Module, splits: dict, profile, args, loss_name: str, batch_size: int):
    device = torch.device(profile.device)
    model = model.to(device)
    xb_tr = torch.tensor(splits["train"]["x"]); yb_tr = torch.tensor(splits["train"]["y"])
    xb_va = torch.tensor(splits["val"]["x"]); yb_va = torch.tensor(splits["val"]["y"])
    optim = engmodels.make_optimizer(model, args.optimizer, args.lr, args.weight_decay)
    n = len(yb_tr)
    steps = max(1, n // batch_size)
    sched = engmodels.make_scheduler(optim, args.scheduler, args.epochs, steps, args.lr)
    best_val, best_state, gaps = float("inf"), None, []
    curves = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n)
        tr_losses = []
        for s in range(0, n, batch_size):
            idx = perm[s:s + batch_size]
            x = xb_tr[idx].to(device); y = yb_tr[idx].to(device)
            optim.zero_grad(set_to_none=True)
            out = model(x)
            loss = compute_loss(loss_name, out, y)
            loss.backward()
            engmodels.apply_gradient_policy(model, args.gradient_policy, epoch)
            optim.step()
            tr_losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            va_losses = []
            for s in range(0, len(yb_va), batch_size):
                x = xb_va[s:s + batch_size].to(device); y = yb_va[s:s + batch_size].to(device)
                va_losses.append(float(compute_loss(loss_name, model(x), y).cpu()))
        tr, va = float(np.mean(tr_losses)), float(np.mean(va_losses))
        curves.append({"epoch": epoch, "train_loss": tr, "val_loss": va})
        if va < best_val - args.min_delta:
            best_val = va
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        gaps.append(va)
        print(f"epoch={epoch:03d} train={tr:.6f} val={va:.6f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    val_overfit_gap = float(gaps[-1] - best_val) if gaps else float("nan")
    return model, pd.DataFrame(curves), val_overfit_gap


def predict_output(model: nn.Module, split: dict, profile, batch_size: int) -> np.ndarray:
    device = torch.device(profile.device)
    model.eval()
    xb = torch.tensor(split["x"])
    outs = []
    with torch.no_grad():
        for s in range(0, len(xb), batch_size):
            outs.append(model(xb[s:s + batch_size].to(device)).cpu().numpy())
    arr = np.concatenate(outs, axis=0)
    return arr


# %% [markdown]
# ## 9. 케이스 실행 (단일 종목)

# %%
def run_case(case: dict, features: pd.DataFrame, profile, args):
    engmodels.set_seed(int(case["seed"]))
    data = make_windows_for_ticker(features, args.seq_len, args.horizon,
                                   case["preprocessing"], case["normalization"], args.max_windows, args.stride)
    if data is None:
        raise ValueError("feature 컬럼 부족")
    splits = engwin.time_split(data, args.train_ratio, args.val_ratio)
    n_features = data["x"].shape[-1]
    out_dim = loss_out_dim(case["loss"])
    batch_size = int(args.batch_size)
    while True:
        try:
            model = make_wrapped(case["model"], args.seq_len, n_features, args.hidden, out_dim,
                                 wrapper_norm_for(str(case["normalization"])))
            model, curves, gap = train_one(model, splits, profile, args, case["loss"], batch_size)
            out = predict_output(model, splits["test"], profile, batch_size)
            break
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower() or batch_size <= 8:
                raise
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            batch_size = max(8, batch_size // 2)
            print(f"[oom-retry] batch_size -> {batch_size}")
    pred = point_from_output(case["loss"], out)
    test = splits["test"]
    result = {k: case[k] for k in ("suite", "ticker", "model", "loss", "normalization", "preprocessing", "seed")}
    result.update(trend_metrics(test["y"], pred, test["past_return"], test["prev_close"], test["target_close"]))
    result["val_overfit_gap"] = gap
    result["horizon"] = args.horizon
    artifacts = {"pred": pred, "test": test, "curves": curves}
    return result, artifacts


# %% [markdown]
# ## 10. 결과 저장 (raw md + csv + 대표 그림) — 판독 가이드 자동 출력

# %%
def frame_to_md(frame: pd.DataFrame) -> str:
    try:
        return frame.to_markdown(index=False)
    except ImportError:
        return "```\n" + frame.to_string(index=False) + "\n```"


def save_raw_report(suite, args, environment, statistics, leaderboard, failures, extra_sections):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    leaderboard.to_csv(RESULTS_DIR / f"{suite}_leaderboard.csv", index=False)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"# 16번 raw 결과 — {suite}", "",
        f"- 실행 시각: {stamp} (서버, 헤드리스 .py)",
        f"- 드라이버: `test/models/16_nonstationary_crosssectional_test.py` --suite {suite}",
        f"- 케이스 수: {len(leaderboard)} (실패 {len(failures)})", "",
        "## 실행 인자", "", "```json",
        json.dumps({k: v for k, v in vars(args).items()}, ensure_ascii=False, indent=2, default=str), "```", "",
        "## 환경", "", "```json", json.dumps(environment, ensure_ascii=False, indent=2, default=str), "```", "",
        "## 데이터 기초 통계", "", "```json", json.dumps(statistics, ensure_ascii=False, indent=2, default=str), "```", "",
    ]
    intent = SUITE_INTENT.get(suite)
    if intent:
        lines += [f"## 이 suite가 보는 것 — {intent['name']}", "",
                  f"- **고정**: {intent['fix']}", f"- **변화**: {intent['vary']}",
                  f"- **질문**: {intent['question']}", f"- **읽는 법**: {intent['read']}", ""]
    lines += ["## 지표 사전", ""] + [f"- `{n}`: {d}" for n, d in METRIC_GLOSSARY]
    lines += ["", "## 전체 leaderboard", "", frame_to_md(leaderboard), ""]
    # known_pitfalls P5: 실행 중 발견한 이슈/수정/재실행 사유를 결과에 raw로 전량 남긴다.
    # 나중에 논문/보고서로 옮길 때 넣고 뺄지는 사용자가 고른다(에이전트가 미리 생략 금지).
    if RUN_ISSUES:
        lines += ["## 실행 중 이슈·판단 로그 (raw, 취사선택은 사용자 몫)", ""]
        lines += [f"- {msg}" for msg in RUN_ISSUES] + [""]
    for title, body in extra_sections:
        lines += [f"## {title}", "", body, ""]
    if failures:
        lines += ["## 실패 케이스", "", "```json", json.dumps(failures, ensure_ascii=False, indent=2, default=str), "```", ""]
    (RESULTS_DIR / f"{suite}_raw.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[saved] {RESULTS_DIR / f'{suite}_raw.md'}")


def save_overlay(suite, tag, artifacts, case):
    IMAGES_DIR.joinpath(suite).mkdir(parents=True, exist_ok=True)
    test = artifacts["test"]; pred = artifacts["pred"]
    n = min(400, len(pred))
    fig, ax = plt.subplots(2, 1, figsize=(13, 7), gridspec_kw={"height_ratios": [2.2, 1.0]})
    ax[0].plot(test["y"][-n:], color="#444", lw=1.0, label="actual h-step return")
    ax[0].plot(pred[-n:], color="#d62728", lw=1.0, alpha=0.9, label="model prediction")
    ax[0].axhline(0, color="black", lw=0.5)
    ax[0].set_title(f"{suite} {tag}: {case['model']} loss={case['loss']} norm={case['normalization']} {case['ticker']}")
    ax[0].legend(fontsize=8); ax[0].set_ylabel("cumulative log return")
    ax[1].scatter(test["y"], pred, s=6, alpha=0.4, color="#d62728")
    lim = float(np.nanmax(np.abs(test["y"]))) or 1e-4
    ax[1].plot([-lim, lim], [-lim, lim], "--", color="black", lw=0.6)
    ax[1].set_xlabel("actual"); ax[1].set_ylabel("predicted")
    fig.tight_layout()
    out = IMAGES_DIR / suite / f"fig_{tag}_overlay.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"[saved] {out}")


# %% [markdown]
# ## 11. suite 격자 구성 + 실행기

# %%
def split_list(text: str) -> list[str]:
    return [t.strip() for t in text.split(",") if t.strip()]


def load_ticker_features(args, table: str, ticker: str | None) -> pd.DataFrame:
    from engine import features as engfeat
    a = argparse.Namespace(**vars(args)); a.table = table; a.ticker = ticker
    db = engdata.resolve_db_path(args.db)
    raw = engdata.load_price_data(db, table, ticker, args.max_rows)
    feats = engdata.make_features(raw)
    feats = engfeat.add_coin_specific_features(feats)
    return feats


def top_liquid_tickers(args, n: int) -> list[str]:
    import duckdb
    db = engdata.resolve_db_path(args.db)
    with duckdb.connect(str(db), read_only=True) as con:
        rows = con.execute(
            f"SELECT ticker, SUM(value) v FROM {KRW_TABLE} GROUP BY ticker ORDER BY v DESC LIMIT {int(n)}"
        ).fetchall()
    return [r[0] for r in rows]


def run_single_axis_suite(args, profile, environment):
    """t1/t2: BTC 단일축에서 정규화·손실만 바꾼다(빠른 진단)."""
    feats = load_ticker_features(args, BTC_TABLE, None)
    statistics = engdata.basic_statistics(feats)
    if args.suite == "t1_normalization":
        norms = split_list(args.normalizations); losses = [args.loss]; models = split_list(args.models)
        cases = [{"normalization": nz, "loss": args.loss, "model": m} for nz in norms for m in models]
    else:  # t2_loss
        losses = split_list(args.losses); cases = [{"normalization": args.normalization, "loss": ls, "model": m}
                                                   for ls in losses for m in split_list(args.models)]
    base = {"suite": args.suite, "ticker": "KRW-BTC", "preprocessing": args.preprocessing, "seed": int(split_list(args.seeds)[0])}
    cases = [{**base, **c} for c in cases]
    print(f"[plan] suite={args.suite} cases={len(cases)}")
    if args.dry_run:
        print(pd.DataFrame(cases).to_string(index=False)); return
    rows, failures, kept = [], [], {}
    for i, case in enumerate(cases, 1):
        print(f"\n[case {i}/{len(cases)}] {case}")
        try:
            res, art = run_case(case, feats, profile, args)
            rows.append(res); kept[len(rows) - 1] = (case, art)
            print(json.dumps({k: round(v, 4) for k, v in res.items() if isinstance(v, float)}, ensure_ascii=False))
        except Exception as exc:  # noqa: BLE001
            if not args.continue_on_failure:
                raise
            failures.append({"case": case, "error": str(exc)}); print(f"[failed] {exc}")
        finally:
            torch.cuda.empty_cache() if torch.cuda.is_available() else None; gc.collect()
    if not rows:
        print("[warn] 성공 케이스 없음"); return
    # 코드 게이트: 단일 지표(tail_f1) 최고가 아니라 champion_score(진폭 건전성 반영)로 정렬.
    axis = "loss" if args.suite == "t2_loss" else "normalization"
    champ = select_champion(rows, args.suite)
    rows_sorted = sorted(rows, key=champion_score, reverse=True)
    lb = pd.DataFrame([{**r, "champion_score": round(champion_score(r), 4)} for r in rows_sorted]).reset_index(drop=True)
    # 파이프라인 승계용: 우승 축 값을 파싱하기 쉬운 한 줄로 출력.
    print(f"[WINNER] {args.suite} {axis}={champ[axis]}")
    for tag, pos in (("best", 0), ("worst", len(lb) - 1)):
        target = lb.iloc[pos]
        for j, (c, a) in kept.items():
            if c["loss"] == target["loss"] and c["normalization"] == target["normalization"] and c["model"] == target["model"]:
                save_overlay(args.suite, tag, a, c); break
    save_raw_report(args.suite, args, environment, statistics, lb.round(6), failures, [])


def run_crosssection_suite(args, profile, environment):
    """t3/t4/t5: KRW 전 종목(유동성 상위 N) pooled. IC·정책까지."""
    tickers = top_liquid_tickers(args, args.n_tickers)
    print(f"[plan] suite={args.suite} tickers={len(tickers)} (상위 유동성) {tickers[:8]}...")
    if args.dry_run:
        return
    # 종목별 단일 학습(pooled_ci 근사: 종목마다 같은 구성으로 학습해 종목 가로질러 평가)
    rows, failures, pred_by_ts = [], [], {}
    stats_ref = None
    for ti, ticker in enumerate(tickers, 1):
        try:
            feats = load_ticker_features(args, KRW_TABLE, ticker)
            if stats_ref is None:
                stats_ref = engdata.basic_statistics(feats)
            case = {"suite": args.suite, "ticker": ticker, "model": split_list(args.models)[0],
                    "loss": args.loss, "normalization": args.normalization,
                    "preprocessing": args.preprocessing, "seed": int(split_list(args.seeds)[0])}
            res, art = run_case(case, feats, profile, args)
            rows.append({**res, "rows": int(len(feats))})
            # IC용: 같은 timestamp에 종목별 (예측, 실제) 모음
            test = art["test"]; pred = art["pred"]
            for ts, p, r in zip(test["decision_timestamp"], pred, test["y"]):
                pred_by_ts.setdefault(str(ts), []).append((float(p), float(r)))
            print(f"[t3/{ti}] {ticker}: trend_corr={res['trend_corr']:.4f} tail_f1={res['tail_f1']:.3f} var={res['variance_ratio']:.3f}")
        except Exception as exc:  # noqa: BLE001
            if not args.continue_on_failure:
                raise
            failures.append({"ticker": ticker, "error": str(exc)}); print(f"[failed] {ticker}: {exc}")
        finally:
            torch.cuda.empty_cache() if torch.cuda.is_available() else None; gc.collect()
    if not rows:
        print("[warn] 성공 종목 없음"); return
    lb = pd.DataFrame(rows)
    ic = cross_sectional_ic(pred_by_ts)
    positive = int((lb["trend_corr"] > 0).sum())
    extra = [("cross-sectional 일반화 요약",
              f"{len(lb)}종목 중 trend_corr>0: {positive} ({100*positive/len(lb):.0f}%). "
              f"**cross_sectional_ic = {ic:.4f}** (같은 시점 종목 간 예측-실제 Spearman 평균; "
              f"양수면 상대강도 신호 존재). tail_f1 평균 {lb['tail_f1'].mean():.3f}, "
              f"variance_ratio 평균 {lb['variance_ratio'].mean():.3f}.")]
    save_raw_report(args.suite, args, environment, stats_ref or {}, lb.round(6), failures, extra)


# %% [markdown]
# ## 12. 인자 정의 + main
# 기본 가정: h=16(4시간), seq_len=64(16시간 입력), train/val/test=0.70/0.15/0.15 시간순,
# 거래비용 14bps(정책 단계), num_workers=0(shm crash 회피).

# %%
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="16번 비정상성+큰변동+cross-sectional 실험")
    p.add_argument("--suite", default="t1_normalization",
                   choices=["t1_normalization", "t2_loss", "t3_crosssection", "t4_ranking", "t5_policy"])
    p.add_argument("--db", default=None)
    p.add_argument("--profile", default="school_4090_15gb")
    p.add_argument("--device", choices=["cpu", "cuda"], default=None)
    p.add_argument("--num-workers", type=int, default=0)
    # 격자 축
    p.add_argument("--models", default="ITransformerLike,PatchTSTLike")
    p.add_argument("--normalizations", default="window_standard,revin,dishts_lite,none")
    p.add_argument("--losses", default="huber,pinball,student_t,tail_weighted")
    p.add_argument("--seeds", default="42")
    # 승계 고정값
    p.add_argument("--normalization", default="revin")
    p.add_argument("--loss", default="huber")
    p.add_argument("--preprocessing", default="seasonal_diff16")
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--n-tickers", type=int, default=30)
    # 학습 설정
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--hidden", type=int, default=96)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--min-delta", type=float, default=1e-5)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--optimizer", default="adamw")
    p.add_argument("--scheduler", default="cosine")
    p.add_argument("--gradient-policy", default="clip1")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--max-rows", type=int, default=0)
    p.add_argument("--max-windows", type=int, default=16000)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--continue-on-failure", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_known_args(argv)[0]
    if args.max_rows <= 0:
        args.max_rows = None
    return args


def main(argv=None):
    args = parse_args(argv)
    profile = engres.build_resource_profile(args.profile, args.device)
    profile = dataclasses.replace(profile, num_workers=max(0, int(args.num_workers)))
    engres.apply_resource_profile(profile)
    environment = engres.log_environment(profile)
    print("[environment]", json.dumps(environment, ensure_ascii=False, indent=2, default=str))
    if args.suite in ("t1_normalization", "t2_loss"):
        run_single_axis_suite(args, profile, environment)
    else:
        run_crosssection_suite(args, profile, environment)
    print(f"[done] suite={args.suite}")


if __name__ == "__main__":
    main()
