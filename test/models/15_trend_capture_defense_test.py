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

# 14번 결론: Linear는 분산 폭주가 모델 고유 결함 -> 배제.
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
                        choices=["t1_horizon", "t2_objective", "t3_nonstationarity", "t4_feature", "t5_gate_fusion"])
    parser.add_argument("--db", default=None)
    parser.add_argument("--table", default="btc_15m_advance")
    parser.add_argument("--ticker", default=None)
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
        "## 지표 읽는 법 (요약)",
        "",
        "- `mae_zero_ratio` < 1 이면 \"0 예측(random-walk)\" 기준선을 이긴 것.",
        "- `mase_momentum` < 1 이면 \"직전 h봉 추세 지속\" 기준선을 이긴 것.",
        "- `trend_corr`: 예측 vs 실제 h-step 수익률 Pearson 상관 (높을수록 추세 포착).",
        "- `large_move_da`: |실제 수익률| 상위 25% 구간의 방향 정확도 (변동 큰 구간 포착력).",
        "- `variance_ratio`: 예측분산/실제분산. ≪0.1 평탄화, ≫10 폭주, 0.05~20만 건강 판정.",
        "- `copy_risk_krw`: KRW 스케일 MAE / persistence MAE. 1 미만이면 persistence보다 우수.",
        "",
        "## 전체 leaderboard",
        "",
        frame_to_markdown(leaderboard),
        "",
    ]
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
    if args.suite == "t5_gate_fusion":
        run_gate_fusion_suite(args, features, cache, profile, environment, statistics)
    else:
        run_screen_suite(args, features, cache, profile, environment, statistics)
    print(f"[done] suite={args.suite}")


if __name__ == "__main__":
    main()
