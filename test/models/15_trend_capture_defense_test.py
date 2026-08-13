# [FOR COMMIT TRACKING ONLY - DO NOT EXECUTE]
# This file is automatically mirrored from the corresponding .ipynb for git diff purposes.
# Actual research execution should be performed in the Jupyter Notebook (.ipynb)
# or in an approved remote/server environment.

# %% [markdown]
# # 15번: 변동·추세 포착 최적화 + 방어 gate 융합
#
# 이번 실험부터 실행 방식이 노트북 셀 실행이 아니라 **헤드리스 `.py` 드라이버**로 바뀌었다(13번 notebook backup 실패·출력 용량 문제 교훈 + 원격 세션 전환). 이 노트북은 커밋 정책(동명 `.ipynb` 필수)을 지키는 미러이며, 실행과 결과 저장은 `.py`가 담당한다.
#
# - 실행: `python test/models/15_trend_capture_defense_test.py --suite t1_horizon` 등 (아래 셀 docstring 참고)
# - raw 결과: `test/results/15_trend_capture_defense_20260716/*_raw.md` + `*_leaderboard.csv`
# - 이미지: `test/images/15_trend_capture_defense_20260716/<suite>/`
# - 계획서: `test/experiment_specs/15_trend_capture_defense_plan_20260716.md`
#
# 아래 한 셀이 `.py` 미러와 동일하다.

# %%
"""15번: 변동·추세 포착 최적화 + 방어 gate 융합 (헤드리스 .py 드라이버).

배경(계획서: test/experiment_specs/15_trend_capture_defense_plan_20260716.md):
- 8~14번 내내 "다음 15분 수익률" 점예측은 persistence를 못 이겼고, 학습은
  분산 폭주(Linear) 아니면 0 근처 평탄화 사이를 오갔다. 13번의 "변동 없는 예측 그래프"는
  방어 우선 셋업과 평탄화 모드가 결합한 결과다.
- 15번은 target을 h-step 누적수익률(추세)로 바꿔 "변동을 살린 예측"이 가능한지 본다.
  risk gate(11번 계보)는 버리지 않고 T5에서 추세 정책과 융합해 MDD 방어를 함께 평가한다.

engine/ 공용 패키지(12 fusion 결함 ①②③ 교정본)를 재사용한다. 원본 8~14는 read-only.
build_risk_windows가 이미 h-step 누적수익 경로를 제공하므로 이를 회귀 target으로 쓴다.
point/risk가 같은 윈도우와 decision_timestamp를 공유해 정렬(fix①)이 자명해진다.

13번 crash 교훈 반영: 헤드리스 실행(Agg), per-case 그래프 없음, DataLoader num_workers=0,
suite 단계 분리, 모든 결과는 test/results·test/images 아래 md/csv/png로 저장.

실행 예시(서버):
    uv run test/models/15_trend_capture_defense_test.py --suite t1_horizon --dry-run
    uv run test/models/15_trend_capture_defense_test.py --suite t1_horizon
    uv run test/models/15_trend_capture_defense_test.py --suite t2_objective --horizons 16 \
        --models PatchTSTLike,DLinearLike,NLinearLike
    uv run test/models/15_trend_capture_defense_test.py --suite t5_gate_fusion \
        --feature-sets coin_multitimeframe_structure,mtf_plus_momentum --seeds 42,7,123
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
# 서버에 설치된 CJK 폰트로 그림 내 한글 라벨 깨짐(tofu) 방지.
matplotlib.rcParams["font.family"] = ["Noto Sans CJK KR", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def _engine_root(start: Path) -> Path:
    candidates = [start, *start.parents, Path.home() / "personal_ai_project" / "quantitative_trading"]
    for candidate in candidates:
        if (candidate / "pyproject.toml").exists() and (candidate / "engine").is_dir():
            return candidate
    raise RuntimeError("engine을 담은 quantitative_trading 디렉터리를 찾지 못했다.")


try:
    _START = Path(__file__).resolve().parent
except NameError:  # 노트북 셀 실행 경로
    _START = Path.cwd()
ROOT = _engine_root(_START)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine import data as engdata
from engine import features as engfeat
from engine import fusion as engfusion
from engine import models as engmodels
from engine import point as engpoint
from engine import resources as engres
from engine import risk as engrisk
from engine import windows as engwin

EXPERIMENT_TAG = "15_trend_capture_defense_20260716"
RESULTS_DIR = ROOT / "test" / "results" / EXPERIMENT_TAG
IMAGES_DIR = ROOT / "test" / "images" / EXPERIMENT_TAG

# ---------------------------------------------------------------------------
# 판독 가이드: 각 suite가 "무엇을 왜 비교하는가"와 각 지표의 의미.
# raw md 상단에 자동으로 찍혀서, 결과 숫자만 보고도 무슨 실험인지 알 수 있게 한다.
# (사용자 요청 2026-07-16: horizon/objective/nonstationarity 등 축이 뭔지 프롬프트·
#  결과 단에서 설명이 있어야 한다.)
# ---------------------------------------------------------------------------

SUITE_INTENT: dict[str, dict[str, str]] = {
    "t1_horizon": {
        "name": "예측 지평(horizon) 스크린",
        "fix": "변수셋·전처리·정규화·objective를 고정한다.",
        "vary": "예측 대상 기간 h(1·4·16·64봉 = 15분·1시간·4시간·16시간)와 모델을 바꾼다.",
        "question": "몇 봉 앞을 맞히려 할 때 신호가 가장 잘 잡히는가? (짧을수록 랜덤워크에 가까움)",
        "read": "mae_zero_ratio가 1에 가까워질수록, mase_momentum이 작을수록 그 h가 예측 가능.",
    },
    "t2_objective": {
        "name": "손실함수(objective) 스크린",
        "fix": "h=16, 변수셋·전처리·정규화를 고정한다.",
        "vary": "손실함수 종류(huber·방향강조·분산강조·상관강조·tail강조·regime·anti_collapse)를 바꾼다.",
        "question": "어떤 벌점 기준으로 학습해야 추세를 살리면서 평탄화·폭주를 피하는가?",
        "read": "trend_corr가 높고 variance_ratio가 건강(0.05~20) 구간이면 좋은 objective.",
    },
    "t3_nonstationarity": {
        "name": "비정상성(nonstationarity) 처리 스크린",
        "fix": "h=16, 변수셋·objective·모델을 고정한다.",
        "vary": "정규화(window_standard/robust/asinh_revin) × 전처리(seasonal_diff16/winsor_025/none)를 바꾼다.",
        "question": "비정상 시계열을 어떤 안정화 조합으로 넣어야 추세 신호가 안 죽는가?",
        "read": "trend_corr 보존 + variance_ratio 건강이 동시에 되는 조합이 우수.",
    },
    "t4_feature": {
        "name": "multi-timeframe 변수 분해",
        "fix": "h=16, 우승 모델·objective·전처리를 고정한다.",
        "vary": "mtf 변수셋을 하위 블록(returns/volatility/trend/volume_range)과 확장(+momentum 등)으로 쪼갠다.",
        "question": "12·14번에서 확인된 mtf 우위가 실제로 어느 하위 블록에서 나오는가?",
        "read": "특정 블록만으로도 trend_corr·large_move_da가 유지되면 그 블록이 신호원.",
    },
    "t5_gate_fusion": {
        "name": "risk gate 방어 융합",
        "fix": "추세 예측(우승 구성)을 고정한다.",
        "vary": "risk gate 공격성(none/0.45/0.55/0.65)을 얹어 급변 구간 매수를 차단한다.",
        "question": "방어층(risk gate)이 추세 정책 위에서도 MDD를 실제로 줄이는가? 노출 대비 트레이드오프는?",
        "read": "gate를 켤수록 MDD가 0에 가까워지면 방어 작동. 단 active_share/trade_count 하한 통과분만 비교.",
    },
    "t6_signal_boost": {
        "name": "신호 강화 — seed ensemble + 데이터 규모",
        "fix": "우승 h·objective·전처리·정규화·변수셋을 고정한다.",
        "vary": "seed(여러 개) 평균 앙상블 여부와 학습 데이터 규모(max_windows/stride)를 바꾼다.",
        "question": "seed 평균과 더 많은 데이터로 trend_corr/large_move_da가 실제로 오르는가?",
        "read": "ensemble_* 행의 trend_corr가 단일 seed 평균보다 높으면 신호 강화 성공. 안 오르면 예측축 한계.",
    },
    "t7_amplitude": {
        "name": "진폭 교정 — tail 가중 + 분산 보존",
        "fix": "우승 h·정규화·변수셋을 고정한다.",
        "vary": "tail 가중 손실과 분산 보존 objective, 전처리를 바꿔 진폭 과소예측을 교정한다.",
        "question": "variance_ratio를 0.36에서 1 근처로 끌어올리면서 trend_corr를 유지할 수 있는가?",
        "read": "variance_ratio가 1에 가까워지고 large_move_da가 오르면 큰 변동 포착 개선.",
    },
    "t8_multiasset": {
        "name": "다자산 일반화",
        "fix": "우승 구성(모델·objective·전처리·h)을 고정한다.",
        "vary": "대상 종목(KRW-BTC/ETH/XRP/SOL)을 바꾼다.",
        "question": "같은 구성의 신호가 BTC 전용인가, 다른 코인에도 일반화되는가?",
        "read": "여러 종목에서 trend_corr가 함께 양수면 신호가 구조적. 한 종목만이면 우연·과적합 의심.",
    },
    "t9_bridge": {
        "name": "12번 챔피언 브리지 재현 — 진동 폭 축소 원인 분리",
        "fix": "12번 case 41과 완전히 동일한 설정: 1-step(다음 15분) target, Linear(+비교군), "
               "balanced_composite, seasonal_diff16, window_standard, seq_len 64, max_windows 4096, "
               "stride 1, epochs 12, patience 5, batch 48(=school_4090_15gb profile).",
        "vary": "데이터 구간만 바꾼다 — early(DB 앞 40k행 ≈ 12번과 같은 2023~2024 구간) vs "
                "recent(DB 뒤 40k행 ≈ 2025~2026 최근 구간).",
        "question": "예전(12번)에 보이던 '변동을 유지하며 추세를 따라가는 예측'이 같은 설정으로 "
                    "재현되는가? 지금의 낮은 진동 폭이 설정 탓인가, 시장 구간 탓인가, h-step 전환 탓인가?",
        "read": "early에서 variance_ratio·Pearson이 12번 기록(var≈0.1~0.15, Pearson≈0.13)에 근접하면 "
                "설정 재현 성공. recent에서만 죽으면 구간(레짐) 문제. 둘 다 죽으면 데이터/엔진 차이를 "
                "더 파야 한다. 그림은 12번 case 41 진단(return prediction/next-candle/scatter)과 같은 "
                "구성으로 저장하니 직접 나란히 비교한다.",
    },
}

METRIC_GLOSSARY: list[tuple[str, str]] = [
    ("mae_return", "예측 h-step 누적수익률의 평균절대오차. 낮을수록 좋지만 절대값만으로는 판단 불가(아래 비율로 봄)."),
    ("mae_zero_ratio", "예측 오차 ÷ '항상 0(random-walk)으로 예측' 오차. <1이면 랜덤워크 기준선을 이긴 것, ≥1이면 못 이긴 것."),
    ("mase_momentum", "예측 오차 ÷ '직전 h봉 추세가 지속된다고 가정' 오차. <1이면 단순 추세지속 기준선을 이긴 것."),
    ("trend_corr", "예측 vs 실제 h-step 수익률의 Pearson 상관. 0=무상관, 높을수록 추세 방향·크기 동조. 금융 시계열에선 0.1도 유의미."),
    ("r2", "결정계수(설명력). 0 이상이어야 평균 예측보다 나음. 음수면 '그냥 평균/0을 찍는 것보다 못하다'는 뜻."),
    ("direction_accuracy", "전체 구간 방향(부호) 정확도. 0.5=동전던지기."),
    ("large_move_da", "|실제 수익률| 상위 25%(큰 변동) 구간의 방향 정확도. 사용자가 중시하는 '큰 변동 포착력'. 0.5 초과면 우위."),
    ("variance_ratio", "예측분산 ÷ 실제분산. ≪0.1=평탄화(변동 죽음), ≫20=폭주, 0.05~20을 healthy로 판정. 1 근처가 이상적."),
    ("copy_risk_krw", "KRW 스케일 MAE ÷ persistence(직전값) MAE. <1이면 직전값 복사보다 우수."),
    ("healthy_variance", "variance_ratio가 0.05~20 안에 있으면 True. 평탄화/폭주 둘 다 아님."),
    ("qualified", "(T5) active_share≥하한 && trade_count≥하한. '거의 거래 안 해서 MDD가 좋아 보이는' 케이스를 우승에서 제외."),
    ("mdd", "(T5) 최대낙폭. 자산곡선이 고점 대비 가장 많이 빠진 비율. 0에 가까울수록 방어 우수."),
]

# 14번은 Linear 분산 폭주를 "모델 고유 결함"으로 배제했으나, 사후 감사
# (test/results/15_direction_audit_10_to_14_20260716.md)에서 그 폭주가 12번 대비 1/20 규모
# (12k행/2048윈도우) 재실행의 아티팩트일 가능성이 확인됐다. 12번 40k행에서는 같은 Linear가
# copy_risk≈1.06으로 정상이었다. 따라서 기본 스크린에는 Linear를 넣지 않되(폭주 위험 회피),
# T6/T7 신호 강화에서는 CLI로 Linear를 복권해 규모를 맞춘 재검증을 허용한다.
DEFAULT_MODELS = "PatchTSTLike,DLinearLike,NLinearLike,TCN,ModernTCNLike,ITransformerLike"
DEFAULT_OBJECTIVES = (
    "huber,balanced_composite,directional_huber,variance_huber,"
    "correlation_huber,tail_huber,regime_huber,anti_collapse_v2"
)
DEFAULT_FEATURE_SETS = "coin_multitimeframe_structure"

# multi-timeframe 내부 분해 블록(12·14 1위 변수셋의 하위 구조).
MTF_BLOCKS: dict[str, list[str]] = {
    "mtf_returns": ["log_return_1", "return_4", "return_16", "return_64", "return_192"],
    "mtf_volatility": [
        "log_return_1",
        "realized_vol_16",
        "realized_vol_64",
        "realized_vol_192",
        "vol_ratio_16_64",
        "vol_ratio_64_192",
    ],
    "mtf_trend": ["log_return_1", "price_z_192", "trend_strength_64", "trend_strength_192"],
    "mtf_volume_range": ["log_return_1", "volume_z_192", "value_z_192", "range_mean_192", "hl_range_pct"],
}
MTF_EXPANSIONS = {
    "mtf_plus_momentum": "coin_momentum_reversal",
    "mtf_plus_shock": "coin_shock_event",
    "mtf_plus_volregime": "coin_volatility_regime",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="15번 추세 포착 + 방어 융합 실험")
    parser.add_argument("--suite", required=False, default="t1_horizon",
                        choices=["t1_horizon", "t2_objective", "t3_nonstationarity", "t4_feature", "t5_gate_fusion",
                                 "t6_signal_boost", "t7_amplitude", "t8_multiasset", "t9_bridge"])
    parser.add_argument("--db", default=None)
    parser.add_argument("--table", default="upbit_krw_candle")
    parser.add_argument("--ticker", default="KRW-BTC")
    parser.add_argument("--profile", default="school_4090_15gb")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--num-workers", type=int, default=0, help="13번 shm crash 교훈: 기본 0")
    # 격자 축
    parser.add_argument("--horizons", default="1,4,16,64")
    parser.add_argument("--models", default=DEFAULT_MODELS)
    parser.add_argument("--objectives", default=DEFAULT_OBJECTIVES)
    parser.add_argument("--normalizations", default="window_standard,window_robust,asinh_revin")
    parser.add_argument("--preprocessings", default="seasonal_diff16,winsor_025,none")
    parser.add_argument("--feature-sets", default=DEFAULT_FEATURE_SETS)
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--gate-quantiles", default="none,0.45,0.55,0.65")
    # suite 간 승계 고정값 (단계 결과에 따라 CLI로 명시 승계)
    parser.add_argument("--objective", default="balanced_composite")
    parser.add_argument("--normalization", default="window_standard")
    parser.add_argument("--preprocessing", default="seasonal_diff16")
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--risk-model", default="PatchTSTLike")
    parser.add_argument("--risk-event-kind", choices=["absolute_move", "downside"], default="absolute_move")
    parser.add_argument("--event-quantile", type=float, default=0.70)
    # 학습 설정
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--objective-warmup-epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--optimizer", default="adamw")
    parser.add_argument("--scheduler", default="cosine")
    parser.add_argument("--gradient-policy", default="clip1")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    # 데이터/윈도우
    parser.add_argument("--max-rows", type=int, default=0, help="0=전체")
    parser.add_argument("--max-windows", type=int, default=16000)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--cross-tickers", default="")
    # 신호 강화 (T6~T8)
    parser.add_argument("--ensemble-seeds", default="42,7,123,2026",
                        help="T6 seed ensemble에 쓸 seed들. 이 seed들의 예측을 평균해 신호를 강화한다.")
    parser.add_argument("--tickers", default="KRW-BTC,KRW-ETH,KRW-XRP,KRW-SOL",
                        help="T8 다자산 일반화 대상. 각 종목에 우승 구성을 그대로 적용한다.")
    parser.add_argument("--ticker-tables", default="",
                        help="T8용 'KRW-ETH:eth_15m_advance,...' 매핑. 종목별 테이블이 다를 때 지정. "
                             "미지정 종목은 --table을 그대로 쓴다. (다종목 수집: pipelines/rebuild_price_mart.py)")
    # 브리지 (T9): 12번 case 41 동일 조건 재현용
    parser.add_argument("--bridge-rows", type=int, default=40000,
                        help="T9 브리지에서 각 구간(early/recent)에 쓸 행 수. 12번 max_rows=40000과 동일.")
    parser.add_argument("--bridge-periods", default="early,recent",
                        help="T9 구간: early=DB 앞 N행(12번과 같은 2023~2024), recent=DB 뒤 N행(최근).")
    # 정책/gate 평가 (T5)
    parser.add_argument("--cost-bps", type=float, default=14.0)
    parser.add_argument("--policy-entry-bps", type=float, default=20.0)
    parser.add_argument("--min-active-share", type=float, default=0.05)
    parser.add_argument("--min-trade-count", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-failure", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_known_args(argv)[0]


# ---------------------------------------------------------------------------
# feature set 등록
# ---------------------------------------------------------------------------

def register_custom_feature_sets(features: pd.DataFrame, feature_groups: dict[str, list[str]]) -> dict[str, list[str]]:
    """mtf 하위 블록과 확장 조합을 FEATURE_SETS에 등록한다. 없는 컬럼은 큰 소리로 제외."""

    registered: dict[str, list[str]] = {}
    mtf_full = feature_groups.get("coin_multitimeframe_structure", [])
    for name, columns in MTF_BLOCKS.items():
        available = [c for c in columns if c in features.columns]
        missing = sorted(set(columns) - set(available))
        if missing:
            print(f"[feature-set] {name}: 누락 컬럼 제외 {missing}")
        if len(available) >= 3:
            engres.FEATURE_SETS[name] = available
            registered[name] = available
    for name, base_group in MTF_EXPANSIONS.items():
        extra = feature_groups.get(base_group, [])
        if not mtf_full or not extra:
            print(f"[feature-set] {name}: 기반 그룹 부재로 건너뜀 (mtf={bool(mtf_full)}, {base_group}={bool(extra)})")
            continue
        merged = list(dict.fromkeys([*mtf_full, *extra]))
        engres.FEATURE_SETS[name] = merged
        registered[name] = merged
    return registered


# ---------------------------------------------------------------------------
# 윈도우 캐시와 momentum naive
# ---------------------------------------------------------------------------

class WindowCache:
    """같은 (feature_set, preprocessing, normalization, horizon) 설정의 윈도우 재사용.

    build_risk_windows는 윈도우마다 python 루프로 전처리를 적용해 비용이 크다.
    suite 안에서 모델/seed만 바뀌는 경우 다시 만들 이유가 없다.
    """

    def __init__(self, features: pd.DataFrame, args: argparse.Namespace):
        self.features = features
        self.args = args
        self._store: dict[tuple, dict[str, np.ndarray]] = {}
        log_close = np.log(features["close"].astype(float).clip(lower=1e-9)).to_numpy()
        self._log_close = log_close
        self._ts_to_idx = {ts: i for i, ts in enumerate(features["timestamp"].astype(str).to_numpy())}

    def get(self, feature_set: str, preprocessing: str, normalization: str, horizon: int) -> dict[str, np.ndarray]:
        key = (feature_set, preprocessing, normalization, horizon)
        if key in self._store:
            return self._store[key]
        columns = engres.FEATURE_SETS[feature_set]
        started = time.time()
        data = engwin.build_risk_windows(
            self.features,
            columns,
            self.args.seq_len,
            horizon,
            preprocessing,
            normalization,
            self.args.max_windows,
            self.args.stride,
        )
        # momentum naive: 결정 봉 기준 직전 h봉 누적수익. 예측 없이 "추세 지속"만 가정한 기준선.
        past = np.zeros(len(data["y"]), dtype=np.float32)
        for i, ts in enumerate(data["decision_timestamp"]):
            idx = self._ts_to_idx.get(str(ts))
            if idx is not None and idx - horizon >= 0:
                past[i] = self._log_close[idx] - self._log_close[idx - horizon]
        data["past_return"] = past
        print(
            f"[windows] {feature_set} prep={preprocessing} norm={normalization} h={horizon} "
            f"n={len(data['y'])} build={time.time() - started:.1f}s"
        )
        self._store[key] = data
        return self._store[key]


# ---------------------------------------------------------------------------
# 지표
# ---------------------------------------------------------------------------

def pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def trend_metrics(split: dict[str, np.ndarray], pred: np.ndarray) -> dict[str, float]:
    actual = split["y"].astype(np.float64)
    predicted = pred.astype(np.float64)
    past = split["past_return"].astype(np.float64)

    error = np.abs(predicted - actual)
    mae = float(np.mean(error))
    mae_zero = float(np.mean(np.abs(actual)))
    mae_momentum = float(np.mean(np.abs(past - actual)))

    actual_std = float(np.std(actual))
    pred_std = float(np.std(predicted))
    variance_ratio = float((pred_std**2) / (actual_std**2 + 1e-18))
    near_zero_share = float(np.mean(np.abs(predicted) < 0.1 * max(actual_std, 1e-9)))

    large_mask = np.abs(actual) >= np.quantile(np.abs(actual), 0.75)
    direction_accuracy = float(np.mean((predicted > 0) == (actual > 0)))
    large_move_da = float(np.mean((predicted[large_mask] > 0) == (actual[large_mask] > 0)))
    momentum_da = float(np.mean((past > 0) == (actual > 0)))
    momentum_large_da = float(np.mean((past[large_mask] > 0) == (actual[large_mask] > 0)))

    ss_res = float(np.sum((predicted - actual) ** 2))
    ss_tot = float(np.sum((actual - np.mean(actual)) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-18)

    prev_close = split["prev_close"].astype(np.float64)
    target_close = split["target_close"].astype(np.float64)
    pred_close = prev_close * np.exp(predicted)
    mae_krw = float(np.mean(np.abs(pred_close - target_close)))
    persistence_mae_krw = float(np.mean(np.abs(prev_close - target_close)))

    healthy = bool(0.05 <= variance_ratio <= 20.0)
    return {
        "mae_return": mae,
        "mae_zero_ratio": mae / max(mae_zero, 1e-12),
        "mase_momentum": mae / max(mae_momentum, 1e-12),
        "trend_corr": pearson(predicted, actual),
        "r2": float(r2),
        "direction_accuracy": direction_accuracy,
        "large_move_da": large_move_da,
        "momentum_da_baseline": momentum_da,
        "momentum_large_da_baseline": momentum_large_da,
        "variance_ratio": variance_ratio,
        "pred_return_std": pred_std,
        "actual_return_std": actual_std,
        "near_zero_share": near_zero_share,
        "mae_krw": mae_krw,
        "persistence_mae_krw": persistence_mae_krw,
        "copy_risk_krw": mae_krw / max(persistence_mae_krw, 1e-9),
        "healthy_variance": healthy,
        "naive_mae_zero": mae_zero,
        "naive_mae_momentum": mae_momentum,
    }


# ---------------------------------------------------------------------------
# 케이스 실행
# ---------------------------------------------------------------------------

def run_trend_case(case: dict[str, object], cache: WindowCache, profile, args: argparse.Namespace):
    engmodels.set_seed(int(case["seed"]))
    windows_data = cache.get(
        str(case["feature_set"]), str(case["preprocessing"]), str(case["normalization"]), int(case["horizon"])
    )
    splits = engwin.time_split(windows_data, args.train_ratio, args.val_ratio)
    n_features = windows_data["x"].shape[-1]
    batch_size = int(args.batch_size)
    started = time.time()
    while True:
        try:
            model = engmodels.make_model(str(case["model"]), args.seq_len, n_features, args.hidden)
            param_count = sum(p.numel() for p in model.parameters())
            model, curves = engpoint.train_model(model, splits, profile, args, str(case["objective"]), batch_size)
            val_pred = engres.predict(model, splits["val"], profile, batch_size)
            test_pred = engres.predict(model, splits["test"], profile, batch_size)
            break
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower() or batch_size <= 8:
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            batch_size = max(8, batch_size // 2)
            print(f"[oom-retry] batch_size -> {batch_size}")

    result = {**{k: case[k] for k in ("suite", "feature_set", "model", "objective",
                                      "preprocessing", "normalization", "horizon", "seed")}}
    result.update(trend_metrics(splits["test"], test_pred))
    val_metrics = trend_metrics(splits["val"], val_pred)
    result.update({f"val_{k}": v for k, v in val_metrics.items()
                   if k in ("trend_corr", "large_move_da", "mase_momentum", "variance_ratio")})
    result.update(
        {
            "param_count": int(param_count),
            "batch_size": batch_size,
            "epochs_run": int(curves["epoch"].max()) if len(curves) else 0,
            "best_val_loss": float(curves["val_loss"].min()) if len(curves) else float("nan"),
            "final_grad_norm": float(curves["grad_norm_mean"].iloc[-1]) if len(curves) else float("nan"),
            "train_seconds": round(time.time() - started, 1),
        }
    )
    artifacts = {
        "test_pred": test_pred,
        "test_actual": splits["test"]["y"],
        "test_past": splits["test"]["past_return"],
        "test_ts": splits["test"]["decision_timestamp"],
        "curves": curves,
        "model": model,
        "splits": splits,
    }
    return result, artifacts


def run_risk_classifier(feature_set: str, seed: int, cache: WindowCache, profile, args: argparse.Namespace):
    """T5용 위험 gate: 같은 윈도우에서 absolute_move/downside 이벤트 분류기 학습."""

    engmodels.set_seed(seed)
    windows_data = cache.get(feature_set, args.preprocessing, args.normalization, args.horizon)
    splits, threshold = engwin.prepare_event_splits(
        dict(windows_data), args.train_ratio, args.val_ratio, args.risk_event_kind, args.event_quantile
    )
    n_features = windows_data["x"].shape[-1]
    batch_size = int(args.batch_size)
    while True:
        try:
            model = engmodels.make_model(args.risk_model, args.seq_len, n_features, args.hidden)
            model, curves = engrisk.train_event_model(model, splits, profile, args, batch_size)
            val_prob = engrisk.predict_event_probability(model, splits["val"], profile, batch_size)
            test_prob = engrisk.predict_event_probability(model, splits["test"], profile, batch_size)
            break
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower() or batch_size <= 8:
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            batch_size = max(8, batch_size // 2)
            print(f"[oom-retry] risk batch_size -> {batch_size}")
    ap = engrisk.average_precision(splits["test"]["y"], test_prob)
    event_rate = float(np.mean(splits["test"]["y"]))
    return {
        "event_score_threshold": threshold,
        "val_prob": val_prob,
        "test_prob": test_prob,
        "test_event": splits["test"]["y"],
        "average_precision": ap,
        "event_rate": event_rate,
        "ap_lift": ap / event_rate if event_rate > 0 else float("nan"),
    }


def evaluate_policy(
    test_split: dict[str, np.ndarray],
    test_pred: np.ndarray,
    args: argparse.Namespace,
    gate_prob: np.ndarray | None = None,
    gate_cutoff: float | None = None,
) -> dict[str, float]:
    """비중복(h봉 간격) 의사결정 long/flat 정책. h-step target의 겹침 이중계산을 피한다."""

    horizon = int(args.horizon)
    step = max(1, int(np.ceil(horizon / max(1, args.stride))))
    idx = np.arange(0, len(test_pred), step)
    pred_dec = test_pred[idx]
    actual_dec = test_split["y"][idx].astype(np.float64)
    entry = args.policy_entry_bps * 1e-4
    position = (pred_dec > entry).astype(float)
    gated_share = 0.0
    if gate_prob is not None and gate_cutoff is not None:
        blocked = gate_prob[idx] >= gate_cutoff
        gated_share = float(np.mean(blocked & (position > 0)))
        position = position * (~blocked)
    metrics = engfusion.position_metrics(actual_dec, position, args.cost_bps * 1e-4)
    metrics["decision_count"] = int(len(idx))
    metrics["blocked_long_share"] = gated_share
    metrics["qualified"] = bool(
        metrics["active_share"] >= args.min_active_share and metrics["trade_count"] >= args.min_trade_count
    )
    return metrics


# ---------------------------------------------------------------------------
# suite 격자 구성
# ---------------------------------------------------------------------------

def split_list(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def build_suite_cases(args: argparse.Namespace) -> list[dict[str, object]]:
    seeds = [int(s) for s in split_list(args.seeds)]
    base = {
        "suite": args.suite,
        "feature_set": split_list(args.feature_sets)[0],
        "model": split_list(args.models)[0],
        "objective": args.objective,
        "preprocessing": args.preprocessing,
        "normalization": args.normalization,
        "horizon": args.horizon,
        "seed": seeds[0],
    }
    cases: list[dict[str, object]] = []
    if args.suite == "t1_horizon":
        for horizon in [int(h) for h in split_list(args.horizons)]:
            for model in split_list(args.models):
                cases.append({**base, "horizon": horizon, "model": model})
    elif args.suite == "t2_objective":
        for objective in split_list(args.objectives):
            for model in split_list(args.models):
                cases.append({**base, "objective": objective, "model": model})
    elif args.suite == "t3_nonstationarity":
        for normalization in split_list(args.normalizations):
            for preprocessing in split_list(args.preprocessings):
                for model in split_list(args.models):
                    cases.append({**base, "normalization": normalization, "preprocessing": preprocessing, "model": model})
    elif args.suite == "t4_feature":
        for feature_set in split_list(args.feature_sets):
            for model in split_list(args.models):
                for seed in seeds:
                    cases.append({**base, "feature_set": feature_set, "model": model, "seed": seed})
    elif args.suite == "t5_gate_fusion":
        for feature_set in split_list(args.feature_sets):
            for seed in seeds:
                cases.append({**base, "feature_set": feature_set, "seed": seed})
    return cases


# ---------------------------------------------------------------------------
# 결과 저장 (raw md + csv + 그림)
# ---------------------------------------------------------------------------

def frame_to_markdown(frame: pd.DataFrame) -> str:
    try:
        return frame.to_markdown(index=False)
    except ImportError:
        return "```\n" + frame.to_string(index=False) + "\n```"


def save_raw_report(
    suite: str,
    args: argparse.Namespace,
    environment: dict[str, object],
    statistics: dict[str, object],
    leaderboard: pd.DataFrame,
    failures: list[dict[str, object]],
    extra_sections: list[tuple[str, str]],
) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / f"{suite}_leaderboard.csv"
    leaderboard.to_csv(csv_path, index=False)
    md_path = RESULTS_DIR / f"{suite}_raw.md"
    run_stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"# 15번 raw 결과 — {suite}",
        "",
        f"- 실행 시각: {run_stamp} (서버, 헤드리스 .py)",
        f"- 드라이버: `test/models/15_trend_capture_defense_test.py` --suite {suite}",
        f"- 케이스 수: {len(leaderboard)} (실패 {len(failures)})",
        "",
        "## 실행 인자",
        "",
        "```json",
        json.dumps({k: v for k, v in vars(args).items()}, ensure_ascii=False, indent=2, default=str),
        "```",
        "",
        "## 환경",
        "",
        "```json",
        json.dumps(environment, ensure_ascii=False, indent=2, default=str),
        "```",
        "",
        "## 데이터 기초 통계",
        "",
        "```json",
        json.dumps(statistics, ensure_ascii=False, indent=2, default=str),
        "```",
        "",
    ]
    intent = SUITE_INTENT.get(suite)
    if intent:
        lines.extend(
            [
                f"## 이 suite가 보는 것 — {intent['name']}",
                "",
                f"- **고정**: {intent['fix']}",
                f"- **변화**: {intent['vary']}",
                f"- **질문**: {intent['question']}",
                f"- **읽는 법**: {intent['read']}",
                "",
            ]
        )
    lines.extend(
        [
            "## 지표 사전 (각 컬럼이 무엇인가)",
            "",
        ]
    )
    lines.extend([f"- `{name}`: {desc}" for name, desc in METRIC_GLOSSARY])
    lines.extend(
        [
            "",
            "## 전체 leaderboard",
            "",
            frame_to_markdown(leaderboard),
            "",
        ]
    )
    for title, body in extra_sections:
        lines.extend([f"## {title}", "", body, ""])
    if failures:
        lines.extend(["## 실패 케이스", "", "```json", json.dumps(failures, ensure_ascii=False, indent=2, default=str), "```", ""])
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[saved] {md_path}")
    print(f"[saved] {csv_path}")
    return md_path


def image_dir(suite: str) -> Path:
    path = IMAGES_DIR / suite
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_overlay_figure(suite: str, tag: str, artifacts: dict[str, object], case: dict[str, object]) -> None:
    """예측 vs 실제 h-step 수익률 overlay — '예측이 변동을 살렸는가'를 직접 보는 그림."""

    n = min(400, len(artifacts["test_actual"]))
    ts = pd.to_datetime(pd.Series(artifacts["test_ts"][-n:]))
    actual = np.asarray(artifacts["test_actual"][-n:], dtype=float)
    pred = np.asarray(artifacts["test_pred"][-n:], dtype=float)
    past = np.asarray(artifacts["test_past"][-n:], dtype=float)
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=False, gridspec_kw={"height_ratios": [2.2, 1.0]})
    axes[0].plot(ts, actual, label="actual h-step return", color="#444444", linewidth=1.0)
    axes[0].plot(ts, pred, label="model prediction", color="#d62728", linewidth=1.0, alpha=0.9)
    axes[0].plot(ts, past, label="momentum naive (past h return)", color="#1f77b4", linewidth=0.8, alpha=0.5)
    axes[0].axhline(0.0, color="black", linewidth=0.5)
    axes[0].set_title(
        f"{suite} {tag}: {case['model']} h={case['horizon']} {case['feature_set']} "
        f"obj={case['objective']} prep={case['preprocessing']} norm={case['normalization']} seed={case['seed']}"
    )
    axes[0].set_ylabel("cumulative log return")
    axes[0].legend(loc="upper left", fontsize=8)
    axes[1].scatter(actual, pred, s=6, alpha=0.4, color="#d62728")
    lim = float(np.nanmax(np.abs(np.concatenate([actual, pred])))) or 1e-4
    axes[1].plot([-lim, lim], [-lim, lim], color="black", linewidth=0.6, linestyle="--")
    axes[1].set_xlabel("actual")
    axes[1].set_ylabel("predicted")
    fig.tight_layout()
    out = image_dir(suite) / f"fig_{tag}_overlay.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[saved] {out}")


def save_metric_bars(suite: str, leaderboard: pd.DataFrame, group_col: str, metrics: list[str]) -> None:
    for metric in metrics:
        if metric not in leaderboard.columns:
            continue
        pivot = leaderboard.groupby(group_col)[metric].mean().sort_values()
        fig, ax = plt.subplots(figsize=(9, max(3, 0.45 * len(pivot))))
        pivot.plot.barh(ax=ax, color="#1f77b4")
        ax.set_title(f"{suite}: mean {metric} by {group_col}")
        ax.set_xlabel(metric)
        fig.tight_layout()
        out = image_dir(suite) / f"fig_{group_col}_{metric}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"[saved] {out}")


def rank_leaderboard(frame: pd.DataFrame) -> pd.DataFrame:
    ranked = frame.copy()
    ranked["rank_score"] = (
        ranked["large_move_da"].fillna(0.0)
        + ranked["trend_corr"].fillna(0.0)
        - 0.10 * (ranked["mase_momentum"].clip(0, 5) - 1.0)
    )
    ranked.loc[~ranked["healthy_variance"], "rank_score"] = ranked.loc[~ranked["healthy_variance"], "rank_score"] - 1.0
    return ranked.sort_values("rank_score", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# suite 실행기
# ---------------------------------------------------------------------------

def run_screen_suite(args, features, cache, profile, environment, statistics) -> pd.DataFrame:
    cases = build_suite_cases(args)
    print(f"[plan] suite={args.suite} cases={len(cases)}")
    print(pd.DataFrame(cases).to_string(index=False))
    if args.dry_run:
        print("[dry-run] 케이스 구성 확인만 하고 종료.")
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    kept_artifacts: dict[int, tuple[dict[str, object], dict[str, object]]] = {}
    for index, case in enumerate(cases, start=1):
        print(f"\n[case {index}/{len(cases)}] {case}")
        try:
            result, artifacts = run_trend_case(case, cache, profile, args)
            rows.append(result)
            kept_artifacts[len(rows) - 1] = (case, {k: artifacts[k] for k in ("test_pred", "test_actual", "test_past", "test_ts")})
            print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in result.items()
                              if k in ("trend_corr", "large_move_da", "mase_momentum", "mae_zero_ratio",
                                       "variance_ratio", "direction_accuracy")}, ensure_ascii=False))
        except Exception as exc:  # noqa: BLE001 - 케이스 단위 격리
            if not args.continue_on_failure:
                raise
            failures.append({"case": case, "error": str(exc)})
            print(f"[case failed] {exc}")
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    if not rows:
        print("[warn] 성공한 케이스가 없다.")
        return pd.DataFrame()

    leaderboard = rank_leaderboard(pd.DataFrame(rows))
    group_col = {
        "t1_horizon": "horizon",
        "t2_objective": "objective",
        "t3_nonstationarity": "normalization",
        "t4_feature": "feature_set",
    }.get(args.suite, "model")
    save_metric_bars(args.suite, leaderboard, group_col, ["trend_corr", "large_move_da", "mase_momentum", "variance_ratio"])
    save_metric_bars(args.suite, leaderboard, "model", ["trend_corr", "large_move_da"])
    if args.suite == "t3_nonstationarity":
        save_metric_bars(args.suite, leaderboard, "preprocessing", ["trend_corr", "large_move_da", "variance_ratio"])

    order = leaderboard.index.tolist()
    original = pd.DataFrame(rows)
    for tag, row_pos in (("best", 0), ("second", 1), ("worst", len(leaderboard) - 1)):
        if row_pos >= len(leaderboard):
            continue
        target = leaderboard.iloc[row_pos]
        match = original[
            (original["model"] == target["model"]) & (original["horizon"] == target["horizon"])
            & (original["objective"] == target["objective"]) & (original["feature_set"] == target["feature_set"])
            & (original["preprocessing"] == target["preprocessing"])
            & (original["normalization"] == target["normalization"]) & (original["seed"] == target["seed"])
        ]
        if match.empty:
            continue
        case, artifacts = kept_artifacts[int(match.index[0])]
        save_overlay_figure(args.suite, tag, artifacts, case)

    summary_by_group = leaderboard.groupby(group_col)[
        ["trend_corr", "large_move_da", "mase_momentum", "mae_zero_ratio", "variance_ratio", "direction_accuracy"]
    ].mean().round(4)
    extra = [(f"{group_col}별 평균", frame_to_markdown(summary_by_group.reset_index()))]
    save_raw_report(args.suite, args, environment, statistics, leaderboard.round(6), failures, extra)
    return leaderboard


def run_gate_fusion_suite(args, features, cache, profile, environment, statistics) -> pd.DataFrame:
    feature_sets = split_list(args.feature_sets)
    seeds = [int(s) for s in split_list(args.seeds)]
    gates = split_list(args.gate_quantiles)
    print(f"[plan] suite=t5_gate_fusion feature_sets={feature_sets} seeds={seeds} gates={gates}")
    if args.dry_run:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    equity_figs: list[tuple[str, np.ndarray, dict[str, np.ndarray]]] = []
    for feature_set in feature_sets:
        for seed in seeds:
            case = {
                "suite": args.suite,
                "feature_set": feature_set,
                "model": split_list(args.models)[0],
                "objective": args.objective,
                "preprocessing": args.preprocessing,
                "normalization": args.normalization,
                "horizon": args.horizon,
                "seed": seed,
            }
            print(f"\n[t5 config] {feature_set} seed={seed}")
            try:
                trend_result, artifacts = run_trend_case(case, cache, profile, args)
                risk_bundle = run_risk_classifier(feature_set, seed, cache, profile, args)
                test_split = artifacts["splits"]["test"]
                equity_curves: dict[str, np.ndarray] = {}
                for gate in gates:
                    if gate == "none":
                        policy = evaluate_policy(test_split, artifacts["test_pred"], args)
                        cutoff = float("nan")
                    else:
                        quantile = float(gate)
                        cutoff = float(np.quantile(risk_bundle["val_prob"], quantile))
                        policy = evaluate_policy(
                            test_split, artifacts["test_pred"], args,
                            gate_prob=risk_bundle["test_prob"], gate_cutoff=cutoff,
                        )
                    row = {
                        **{k: case[k] for k in ("suite", "feature_set", "model", "objective",
                                                "preprocessing", "normalization", "horizon", "seed")},
                        "gate_quantile": gate,
                        "gate_cutoff": cutoff,
                        "risk_ap": risk_bundle["average_precision"],
                        "risk_ap_lift": risk_bundle["ap_lift"],
                        "risk_event_rate": risk_bundle["event_rate"],
                        "trend_corr": trend_result["trend_corr"],
                        "large_move_da": trend_result["large_move_da"],
                        "mase_momentum": trend_result["mase_momentum"],
                        "variance_ratio": trend_result["variance_ratio"],
                        **policy,
                    }
                    rows.append(row)
                    # equity curve 재계산(그림용)
                    horizon_step = max(1, int(np.ceil(args.horizon / max(1, args.stride))))
                    idx = np.arange(0, len(artifacts["test_pred"]), horizon_step)
                    pred_dec = artifacts["test_pred"][idx]
                    actual_dec = test_split["y"][idx].astype(np.float64)
                    position = (pred_dec > args.policy_entry_bps * 1e-4).astype(float)
                    if gate != "none":
                        position = position * (risk_bundle["test_prob"][idx] < cutoff)
                    turnover = np.abs(np.diff(np.r_[0.0, position]))
                    pnl = position * actual_dec - turnover * args.cost_bps * 1e-4
                    equity_curves[f"gate={gate}"] = np.exp(np.cumsum(pnl))
                buyhold = np.exp(np.cumsum(test_split["y"][np.arange(0, len(artifacts["test_pred"]),
                                                                     max(1, int(np.ceil(args.horizon / max(1, args.stride)))))].astype(np.float64)))
                equity_curves["buy&hold"] = buyhold
                equity_figs.append((f"{feature_set}_seed{seed}", np.asarray(test_split["decision_timestamp"]), equity_curves))
            except Exception as exc:  # noqa: BLE001
                if not args.continue_on_failure:
                    raise
                failures.append({"feature_set": feature_set, "seed": seed, "error": str(exc)})
                print(f"[t5 failed] {exc}")
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

    if not rows:
        print("[warn] 성공한 t5 구성이 없다.")
        return pd.DataFrame()

    leaderboard = pd.DataFrame(rows).sort_values(["feature_set", "seed", "gate_quantile"]).reset_index(drop=True)

    for tag, ts, curves in equity_figs:
        horizon_step = max(1, int(np.ceil(args.horizon / max(1, args.stride))))
        ts_dec = pd.to_datetime(pd.Series(ts[np.arange(0, len(ts), horizon_step)]))
        fig, ax = plt.subplots(figsize=(12, 5))
        for label, curve in curves.items():
            n = min(len(ts_dec), len(curve))
            style = {"linewidth": 1.2}
            if label == "buy&hold":
                style = {"linewidth": 1.0, "linestyle": "--", "color": "#888888"}
            ax.plot(ts_dec[:n], curve[:n], label=label, **style)
        ax.set_title(f"t5_gate_fusion equity ({tag}, h={args.horizon}, 비중복 의사결정, cost {args.cost_bps}bps)")
        ax.set_ylabel("equity (×)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        out = image_dir(args.suite) / f"fig_equity_{tag}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"[saved] {out}")

    fig, ax = plt.subplots(figsize=(8, 6))
    qualified = leaderboard[leaderboard["qualified"]]
    colors = {"none": "#888888", "0.45": "#d62728", "0.55": "#ff7f0e", "0.65": "#1f77b4"}
    for gate in leaderboard["gate_quantile"].unique():
        sub = qualified[qualified["gate_quantile"] == gate]
        ax.scatter(sub["mdd"], sub["cumulative_return"], label=f"gate={gate}",
                   color=colors.get(str(gate), None), s=42, alpha=0.85)
    ax.set_xlabel("MDD (0에 가까울수록 방어 우수)")
    ax.set_ylabel("cumulative return")
    ax.set_title("t5: gate 공격성별 방어-수익 trade-off (활동 하한 통과만)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = image_dir(args.suite) / "fig_tradeoff.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[saved] {out}")

    pivot = leaderboard.pivot_table(index=["feature_set", "seed"], columns="gate_quantile",
                                    values=["mdd", "cumulative_return", "active_share"], aggfunc="first").round(4)
    extra = [("gate별 pivot (mdd / cumulative_return / active_share)", "```\n" + pivot.to_string() + "\n```")]
    save_raw_report(args.suite, args, environment, statistics, leaderboard.round(6), failures, extra)
    return leaderboard


# ---------------------------------------------------------------------------
# 신호 강화 suite (T6~T8): 예측 신호 자체를 끌어올린다.
#   - 사용자 방향 경고(2026-07-16): 예측 튜닝을 무한 반복하지 말 것. 이 세 suite로
#     "예측축을 더 밀면 신호가 오르는가"를 확인하고, 안 오르면 LLM 융합 축으로 넘어간다.
#     (전환 조건은 계획서 15_trend_capture_defense_plan_20260716.md에 명시.)
# ---------------------------------------------------------------------------

def _base_case(args: argparse.Namespace, **override) -> dict[str, object]:
    case = {
        "suite": args.suite,
        "feature_set": split_list(args.feature_sets)[0],
        "model": split_list(args.models)[0],
        "objective": args.objective,
        "preprocessing": args.preprocessing,
        "normalization": args.normalization,
        "horizon": args.horizon,
        "seed": int(split_list(args.seeds)[0]),
    }
    case.update(override)
    return case


def run_signal_boost_suite(args, features, cache, profile, environment, statistics) -> pd.DataFrame:
    """T6: 같은 우승 구성을 여러 seed로 학습해, 단일 seed 대비 seed 평균 앙상블이 신호를 올리는지 본다.

    각 (model) 별로: seed별 단일 결과 + 그 seed 예측을 평균한 ensemble 결과를 나란히 남긴다.
    데이터 규모 효과는 --max-windows/--stride를 CLI로 바꿔 회차를 나눠 비교한다(silent 확장 금지).
    """
    ensemble_seeds = [int(s) for s in split_list(args.ensemble_seeds)]
    models = split_list(args.models)
    print(f"[plan] suite=t6_signal_boost models={models} ensemble_seeds={ensemble_seeds}")
    if args.dry_run:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    overlay_saved = False
    for model in models:
        seed_preds: list[np.ndarray] = []
        test_split_ref: dict[str, np.ndarray] | None = None
        single_metrics: list[dict[str, float]] = []
        for seed in ensemble_seeds:
            case = _base_case(args, model=model, seed=seed)
            print(f"\n[t6 single] {model} seed={seed}")
            try:
                result, artifacts = run_trend_case(case, cache, profile, args)
                rows.append({**result, "member": f"seed{seed}", "is_ensemble": False})
                single_metrics.append(result)
                seed_preds.append(np.asarray(artifacts["test_pred"], dtype=np.float64))
                test_split_ref = artifacts["splits"]["test"]
            except Exception as exc:  # noqa: BLE001
                if not args.continue_on_failure:
                    raise
                failures.append({"model": model, "seed": seed, "error": str(exc)})
                print(f"[t6 failed] {exc}")
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

        if len(seed_preds) >= 2 and test_split_ref is not None:
            ensemble_pred = np.mean(np.stack(seed_preds), axis=0)
            ens = trend_metrics(test_split_ref, ensemble_pred)
            single_corr = float(np.mean([m["trend_corr"] for m in single_metrics]))
            single_lda = float(np.mean([m["large_move_da"] for m in single_metrics]))
            row = {
                **_base_case(args, model=model, seed=-1),
                **ens,
                "member": f"ensemble({len(seed_preds)})",
                "is_ensemble": True,
                "single_mean_trend_corr": single_corr,
                "single_mean_large_move_da": single_lda,
                "ensemble_corr_gain": ens["trend_corr"] - single_corr,
                "ensemble_lda_gain": ens["large_move_da"] - single_lda,
            }
            rows.append(row)
            print(f"[t6 ensemble] {model} corr {single_corr:.4f}->{ens['trend_corr']:.4f} "
                  f"lda {single_lda:.4f}->{ens['large_move_da']:.4f}")
            if not overlay_saved:
                save_overlay_figure(
                    args.suite, f"ensemble_{model}",
                    {"test_pred": ensemble_pred, "test_actual": test_split_ref["y"],
                     "test_past": test_split_ref["past_return"], "test_ts": test_split_ref["decision_timestamp"]},
                    _base_case(args, model=model, seed=-1),
                )
                overlay_saved = True

    if not rows:
        print("[warn] 성공한 t6 구성이 없다.")
        return pd.DataFrame()
    leaderboard = pd.DataFrame(rows)
    save_metric_bars(args.suite, leaderboard, "member", ["trend_corr", "large_move_da", "variance_ratio"])
    ens_only = leaderboard[leaderboard["is_ensemble"]]
    extra = []
    if not ens_only.empty:
        extra.append(("seed ensemble 이득 (음수면 강화 실패)",
                      frame_to_markdown(ens_only[["model", "single_mean_trend_corr", "trend_corr",
                                                  "ensemble_corr_gain", "ensemble_lda_gain"]].round(4))))
    save_raw_report(args.suite, args, environment, statistics, leaderboard.round(6), failures, extra)
    return leaderboard


def run_amplitude_suite(args, features, cache, profile, environment, statistics) -> pd.DataFrame:
    """T7: tail 가중·분산 보존 objective와 전처리를 바꿔 진폭 과소예측(variance_ratio≈0.36)을 교정한다.

    핵심 지표는 variance_ratio(1에 가까워지는가)와 large_move_da(큰 변동 방향을 더 맞히는가)다.
    trend_corr가 함께 유지돼야 '분산만 키운 잡음'이 아니다.
    """
    objectives = split_list(args.objectives)
    preprocessings = split_list(args.preprocessings)
    models = split_list(args.models)
    cases = [
        _base_case(args, model=m, objective=o, preprocessing=p)
        for m in models for o in objectives for p in preprocessings
    ]
    print(f"[plan] suite=t7_amplitude cases={len(cases)}")
    print(pd.DataFrame(cases).to_string(index=False))
    if args.dry_run:
        return pd.DataFrame()

    rows, failures = [], []
    kept: dict[int, tuple[dict[str, object], dict[str, object]]] = {}
    for index, case in enumerate(cases, start=1):
        print(f"\n[t7 {index}/{len(cases)}] {case['model']} obj={case['objective']} prep={case['preprocessing']}")
        try:
            result, artifacts = run_trend_case(case, cache, profile, args)
            rows.append(result)
            kept[len(rows) - 1] = (case, {k: artifacts[k] for k in ("test_pred", "test_actual", "test_past", "test_ts")})
        except Exception as exc:  # noqa: BLE001
            if not args.continue_on_failure:
                raise
            failures.append({"case": case, "error": str(exc)})
            print(f"[t7 failed] {exc}")
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    if not rows:
        print("[warn] 성공한 t7 구성이 없다.")
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    # 진폭 목표: variance_ratio가 1에 가장 가까우면서 trend_corr>0 인 케이스 우선.
    frame["amp_gap"] = (frame["variance_ratio"] - 1.0).abs()
    ranked = frame.sort_values(["amp_gap", "trend_corr"], ascending=[True, False]).reset_index(drop=True)
    save_metric_bars(args.suite, ranked, "objective", ["variance_ratio", "large_move_da", "trend_corr"])
    save_metric_bars(args.suite, ranked, "preprocessing", ["variance_ratio", "large_move_da"])
    for tag, pos in (("best_amp", 0), ("worst_amp", len(ranked) - 1)):
        target = ranked.iloc[pos]
        original = frame[(frame["model"] == target["model"]) & (frame["objective"] == target["objective"])
                         & (frame["preprocessing"] == target["preprocessing"])]
        if not original.empty:
            case, art = kept[int(original.index[0])]
            save_overlay_figure(args.suite, tag, art, case)
    extra = [("진폭 교정 순위 (variance_ratio가 1에 가까울수록 상단)",
              frame_to_markdown(ranked[["model", "objective", "preprocessing", "variance_ratio",
                                        "trend_corr", "large_move_da", "amp_gap"]].head(12).round(4)))]
    save_raw_report(args.suite, args, environment, statistics, ranked.round(6), failures, extra)
    return ranked


def run_multiasset_suite(args, features, cache, profile, environment, statistics) -> pd.DataFrame:
    """T8: 우승 구성을 종목마다 그대로 적용해 신호가 BTC 전용인지 일반적인지 본다.

    각 종목은 독립적으로 데이터 로드→feature→윈도우를 다시 만든다(캐시는 종목별로 분리).
    features/cache 인자는 첫 종목(호출부에서 로드된 것)용이며, 여기서 종목별로 재로딩한다.
    """
    tickers = split_list(args.tickers)
    model = split_list(args.models)[0]
    table_map: dict[str, str] = {}
    for pair in split_list(args.ticker_tables):
        if ":" in pair:
            tkr, tbl = pair.split(":", 1)
            table_map[tkr.strip()] = tbl.strip()
    print(f"[plan] suite=t8_multiasset tickers={tickers} model={model} table_map={table_map}")
    if args.dry_run:
        return pd.DataFrame()

    rows, failures = [], []
    for ticker in tickers:
        table = table_map.get(ticker, args.table)
        print(f"\n[t8 ticker] {ticker} (table={table})")
        try:
            ticker_args = argparse.Namespace(**vars(args))
            ticker_args.ticker = ticker
            ticker_args.table = table
            tfeatures, tgroups, _ = engfeat.load_feature_frame(ticker_args)
            register_custom_feature_sets(tfeatures, tgroups)
            tcache = WindowCache(tfeatures, ticker_args)
            case = _base_case(args, model=model)
            result, artifacts = run_trend_case(case, tcache, profile, ticker_args)
            rows.append({**result, "ticker": ticker, "rows": int(len(tfeatures))})
            save_overlay_figure(
                args.suite, f"{ticker.replace('-', '_')}",
                {k: artifacts[k] for k in ("test_pred", "test_actual", "test_past", "test_ts")}, case,
            )
            print(f"[t8] {ticker}: trend_corr={result['trend_corr']:.4f} large_move_da={result['large_move_da']:.4f}")
        except Exception as exc:  # noqa: BLE001
            if not args.continue_on_failure:
                raise
            failures.append({"ticker": ticker, "error": str(exc)})
            print(f"[t8 failed] {ticker}: {exc}")
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    if not rows:
        print("[warn] 성공한 t8 종목이 없다.")
        return pd.DataFrame()
    leaderboard = pd.DataFrame(rows)
    save_metric_bars(args.suite, leaderboard, "ticker", ["trend_corr", "large_move_da", "variance_ratio", "mase_momentum"])
    positive = int((leaderboard["trend_corr"] > 0).sum())
    extra = [("종목별 일반화 요약",
              f"{len(leaderboard)}종목 중 trend_corr>0 인 종목: {positive}. "
              f"과반이면 신호가 구조적, BTC만이면 종목 특정·과적합 의심.")]
    save_raw_report(args.suite, args, environment, statistics, leaderboard.round(6), failures, extra)
    return leaderboard


# ---------------------------------------------------------------------------
# 브리지 suite (T9): 12번 case 41 동일 조건 재현 — 진동 폭 축소 원인 분리
# ---------------------------------------------------------------------------

def bridge_metrics(split: dict[str, np.ndarray], pred: np.ndarray) -> dict[str, float]:
    """1-step(다음 15분) 예측용 지표. 10/12/14 보고서와 같은 정의로 계산해 직접 비교 가능하게 한다."""

    actual = split["y"].astype(np.float64)
    predicted = pred.astype(np.float64)
    prev_close = split["prev_close"].astype(np.float64)
    target_close = split["target_close"].astype(np.float64)

    pred_close = prev_close * np.exp(predicted)
    mae_krw = float(np.mean(np.abs(pred_close - target_close)))
    persistence_mae_krw = float(np.mean(np.abs(prev_close - target_close)))
    actual_std = float(np.std(actual))
    pred_std = float(np.std(predicted))
    return {
        "mae_krw": mae_krw,
        "persistence_mae_krw": persistence_mae_krw,
        "copy_risk_krw": mae_krw / max(persistence_mae_krw, 1e-9),
        "direction_accuracy": float(np.mean((predicted > 0) == (actual > 0))),
        "variance_ratio": float((pred_std**2) / (actual_std**2 + 1e-18)),
        "pred_return_std": pred_std,
        "actual_return_std": actual_std,
        "pearson": pearson(predicted, actual),
        "near_zero_share": float(np.mean(np.abs(predicted) < 0.1 * max(actual_std, 1e-9))),
    }


def save_bridge_figure(tag: str, split: dict[str, np.ndarray], pred: np.ndarray, title: str) -> None:
    """12번 case 41 진단의 아래 3칸(return prediction / next-candle / calibration scatter)과
    같은 구성으로 저장해, 과거 그림과 직접 나란히 비교할 수 있게 한다."""

    actual = split["y"].astype(np.float64)
    predicted = pred.astype(np.float64)
    prev_close = split["prev_close"].astype(np.float64)
    target_close = split["target_close"].astype(np.float64)
    pred_close = prev_close * np.exp(predicted)

    fig, axes = plt.subplots(1, 3, figsize=(19, 5))
    n = min(200, len(actual))
    axes[0].plot(actual[-n:], label="actual return", color="#1f77b4", linewidth=0.9)
    axes[0].plot(predicted[-n:], label="predicted return", color="#ff7f0e", linewidth=0.9)
    axes[0].axhline(0.0, color="black", linewidth=0.5)
    axes[0].set_title("Return prediction (마지막 200)")
    axes[0].set_xlabel("test time index")
    axes[0].legend(fontsize=8)

    m = min(60, len(actual))
    axes[1].plot(target_close[-m:], label="actual close", color="#2ca02c", linewidth=1.1)
    axes[1].plot(pred_close[-m:], label="predicted close", color="#ff7f0e", linewidth=1.1)
    axes[1].plot(prev_close[-m:], label="persistence close", color="#555555", linewidth=0.8, linestyle="--")
    axes[1].set_title("Next-candle comparison (마지막 60, KRW)")
    axes[1].set_xlabel("candle index")
    axes[1].legend(fontsize=8)

    corr = pearson(predicted, actual)
    axes[2].scatter(actual, predicted, s=6, alpha=0.4)
    lim = float(np.nanmax(np.abs(np.concatenate([actual, predicted])))) or 1e-4
    axes[2].plot([-lim, lim], [-lim, lim], color="black", linewidth=0.6)
    axes[2].set_title(f"Calibration scatter / Pearson={corr:.3f}")
    axes[2].set_xlabel("actual return")
    axes[2].set_ylabel("predicted return")

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    out = image_dir("t9_bridge") / f"fig_{tag}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[saved] {out}")


def run_bridge_suite(args, features, cache, profile, environment, statistics) -> pd.DataFrame:
    """T9: 12번 case 41(1-step Linear+balanced_composite) 동일 조건을 early/recent 구간에 재현.

    - early: DB 앞 N행을 12번과 같은 경로(load_price_data(max_rows)→make_features)로 로드.
    - recent: main이 로드한 전체 features의 뒤 N행 슬라이스(rolling feature는 인과적이라 값 동일).
    비교 기준(12번 기록): case 41 Pearson 0.129, 10번 동계열 copy_risk 1.06~1.13, var_ratio 0.10~0.17.
    """

    periods = split_list(args.bridge_periods)
    models = split_list(args.models)
    seeds = [int(s) for s in split_list(args.seeds)]
    feature_set = split_list(args.feature_sets)[0]
    print(f"[plan] suite=t9_bridge periods={periods} models={models} seeds={seeds} "
          f"rows={args.bridge_rows} max_windows={args.max_windows} stride={args.stride} "
          f"epochs={args.epochs} batch={args.batch_size}")
    if args.dry_run:
        return pd.DataFrame()

    frames: dict[str, pd.DataFrame] = {}
    if "early" in periods:
        db_path = engdata.resolve_db_path(args.db)
        raw_early = engdata.load_price_data(db_path, args.table, args.ticker, args.bridge_rows)
        feats_early = engfeat.add_coin_specific_features(engdata.make_features(raw_early))
        frames["early"] = feats_early
    if "recent" in periods:
        frames["recent"] = features.iloc[-args.bridge_rows:].reset_index(drop=True)

    rows, failures = [], []
    columns = engres.FEATURE_SETS[feature_set]
    for period, feats in frames.items():
        span = f"{feats['timestamp'].min()} ~ {feats['timestamp'].max()}"
        print(f"\n[t9 period={period}] rows={len(feats)} span={span}")
        windows_data = engwin.build_windows(
            feats, columns, args.seq_len, args.preprocessing, args.normalization,
            args.max_windows, args.stride,
        )
        splits = engwin.time_split(windows_data, args.train_ratio, args.val_ratio)
        n_features = windows_data["x"].shape[-1]
        for model_name in models:
            for seed in seeds:
                case_tag = f"{period}_{model_name}_seed{seed}"
                print(f"\n[t9 case] {case_tag}")
                try:
                    engmodels.set_seed(seed)
                    batch = int(args.batch_size)
                    model = engmodels.make_model(model_name, args.seq_len, n_features, args.hidden)
                    model, curves = engpoint.train_model(model, splits, profile, args, args.objective, batch)
                    test_pred = engres.predict(model, splits["test"], profile, batch)
                    metrics = bridge_metrics(splits["test"], test_pred)
                    rows.append({
                        "suite": args.suite, "period": period, "period_span": span,
                        "model": model_name, "objective": args.objective,
                        "preprocessing": args.preprocessing, "normalization": args.normalization,
                        "seed": seed, "target": "1-step(다음 15분)",
                        **metrics,
                        "epochs_run": int(curves["epoch"].max()) if len(curves) else 0,
                    })
                    print(json.dumps({k: round(v, 4) for k, v in metrics.items()}, ensure_ascii=False))
                    save_bridge_figure(
                        case_tag, splits["test"], test_pred,
                        f"t9_bridge {period}({span}) {model_name} 1-step {args.objective} "
                        f"prep={args.preprocessing} seed={seed} — 12번 case41 조건 재현",
                    )
                except Exception as exc:  # noqa: BLE001
                    if not args.continue_on_failure:
                        raise
                    failures.append({"case": case_tag, "error": str(exc)})
                    print(f"[t9 failed] {case_tag}: {exc}")
                finally:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()

    if not rows:
        print("[warn] 성공한 t9 케이스가 없다.")
        return pd.DataFrame()
    leaderboard = pd.DataFrame(rows)
    save_metric_bars(args.suite, leaderboard, "period", ["variance_ratio", "pearson", "copy_risk_krw", "direction_accuracy"])
    reference = (
        "12번 case 41 기록(2026-06-24, 40k행/4096윈도우/epochs12): Pearson 0.129, fusion return +1.79% "
        "(96케이스 중 유일 양수). 10번 동계열(Linear+balanced_composite): copy_risk 1.06~1.13, "
        "variance_ratio 0.10~0.17, DA 51~53%. 14번(12k행/2048윈도우): 같은 구성이 copy_risk 67.6, "
        "variance_ratio 3679로 폭주 — 규모 아티팩트 의혹. 이번 T9가 그 세 기록 사이 어디에 앉는지가 판정 기준."
    )
    extra = [("비교 기준 (과거 기록)", reference)]
    save_raw_report(args.suite, args, environment, statistics, leaderboard.round(6), failures, extra)
    return leaderboard


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    profile = engres.build_resource_profile(args.profile, args.device)
    profile = dataclasses.replace(profile, num_workers=max(0, int(args.num_workers)))
    engres.apply_resource_profile(profile)
    environment = engres.log_environment(profile)

    if args.max_rows <= 0:
        args.max_rows = None  # type: ignore[assignment]
    features, feature_groups, raw = engfeat.load_feature_frame(args)
    registered = register_custom_feature_sets(features, feature_groups)
    statistics = engdata.basic_statistics(features)
    print("[statistics]", json.dumps(statistics, ensure_ascii=False, indent=2, default=str))
    print(f"[feature-sets] engine groups={sorted(feature_groups)} custom={sorted(registered)}")

    cache = WindowCache(features, args)
    dispatch = {
        "t5_gate_fusion": run_gate_fusion_suite,
        "t6_signal_boost": run_signal_boost_suite,
        "t7_amplitude": run_amplitude_suite,
        "t8_multiasset": run_multiasset_suite,
        "t9_bridge": run_bridge_suite,
    }
    runner = dispatch.get(args.suite, run_screen_suite)
    runner(args, features, cache, profile, environment, statistics)
    print(f"[done] suite={args.suite}")


if __name__ == "__main__":
    main()
