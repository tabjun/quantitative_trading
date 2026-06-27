# [FOR COMMIT TRACKING ONLY - DO NOT EXECUTE]
# This file is automatically mirrored from the corresponding .ipynb for git diff purposes.
# Actual research execution should be performed in the Jupyter Notebook (.ipynb)
# or in an approved remote/server environment.

# %%
# [FOR COMMIT TRACKING ONLY - DO NOT EXECUTE]
# This file mirrors the corresponding notebook for git diff and server execution.
# Actual research execution should be performed in the Jupyter Notebook (.ipynb)
# or in an approved remote/server environment.

# %% [markdown]
# # 13번 feature·algorithm·resource 공동 최적화 실험
#
# 12번에서 1순위로 나온 multi-timeframe 변수셋을 내부 분해하고,
# 변수 후보군, 알고리즘, 전처리, risk gate, seed 안정성을 함께 넓게 비교합니다.
#
# 12번 완료 노트북은 읽기 전용 source of truth로 유지합니다. 이 파일은 새 번호 실험입니다.

# %%
"""13번: 변수 후보군, 알고리즘, 전처리, risk gate, seed를 함께 넓게 본다.

이 스크립트는 12번의 검증된 point+risk fusion backend를 재사용한다. 13번에서
새로 추가하는 책임은 다음 네 가지다.

1. 12번의 `coin_multitimeframe_structure`를 returns/volatility/trend/volume-range로 분해한다.
2. multi-timeframe core에 momentum, calendar, shock, attention, liquidity, order-flow, optional mart 변수를 붙인다.
3. Linear/PatchTSTLike뿐 아니라 8번 backend의 TCN, ModernTCN, DLinear/NLinear, Transformer,
   Autoformer, iTransformer, Mamba 계열까지 실행 계획에 포함한다.
4. 서버 독점 실행을 전제로 큰 batch size와 OOM batch downshift를 기본 정책으로 둔다.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import gc
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents, Path.home() / "personal_ai_project" / "quantitative_trading"]:
        if (candidate / "pyproject.toml").exists() and (candidate / "test").exists():
            return candidate
    return start


REPO_ROOT = find_repo_root(Path.cwd())
FUSION_BACKEND_PATH = REPO_ROOT / "test" / "models" / "12_feature_guardrail_fusion_test.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"backend load failed: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


fusion12 = load_module(FUSION_BACKEND_PATH, "feature_guardrail_fusion_backend_for_13")
obj = fusion12.obj
risk = fusion12.risk
diag = fusion12.diag
base = fusion12.base


MODEL_ALIASES = {
    "Linear": "Linear",
    "PatchTSTLike": "PatchTSTLike",
    "TCNLike": "TCN",
    "TCN": "TCN",
    "ModernTCNLike": "ModernTCNLike",
    "DLinearLike": "DLinearLike",
    "NLinearLike": "NLinearLike",
    "TransformerLike": "Transformer",
    "Transformer": "Transformer",
    "ITransformerLike": "ITransformerLike",
    "AutoformerLike": "AutoformerLike",
    "MambaLike": "MambaLike",
}

DEFAULT_POINT_MODELS = [
    "Linear",
    "PatchTSTLike",
    "TCNLike",
    "ModernTCNLike",
    "DLinearLike",
    "NLinearLike",
    "TransformerLike",
    "ITransformerLike",
    "AutoformerLike",
    "MambaLike",
]
DEFAULT_RISK_MODELS = ["PatchTSTLike"]
EXPANDED_RISK_MODELS = ["Linear", "TCNLike", "PatchTSTLike"]
DEFAULT_PREPROCESSINGS = [
    "seasonal_diff16",
    "winsor_025",
    "frequency_bandpass",
    "median_residual_5",
    "linear_detrend+asinh_robust",
    "volatility_scale+asinh_robust",
]
DEFAULT_SEEDS = [42, 137, 2026, 777, 4096]
DEFAULT_RISK_HORIZONS = [8, 16, 32]
DEFAULT_RISK_EVENT_KINDS = ["absolute_move", "downside"]
DEFAULT_RISK_ALLOW_QUANTILES = [0.45, 0.55, 0.65]

MTF_DECOMPOSITION_GROUPS = [
    "mtf_returns_only",
    "mtf_volatility_only",
    "mtf_trend_position_only",
    "mtf_volume_range_only",
    "mtf_returns_volatility",
    "mtf_returns_trend",
    "mtf_volatility_trend",
    "mtf_full",
]
MTF_AUGMENTED_GROUPS = [
    "mtf_plus_momentum_reversal",
    "mtf_plus_calendar_cycle",
    "mtf_plus_shock_event",
    "mtf_plus_attention_proxy",
    "mtf_plus_liquidity_micro",
    "mtf_plus_orderflow_proxy",
    "mtf_plus_full_available",
    "mtf_plus_text_context",
    "mtf_plus_cross_market",
    "mtf_plus_macro_proxy",
    "mtf_plus_onchain_proxy",
    "mtf_plus_derivatives_proxy",
]
DEFAULT_FEATURE_GROUPS = MTF_DECOMPOSITION_GROUPS + MTF_AUGMENTED_GROUPS

SUITE_PRESETS = {
    "dry_plan": {
        "feature_groups": DEFAULT_FEATURE_GROUPS,
        "point_models": DEFAULT_POINT_MODELS,
        "risk_models": EXPANDED_RISK_MODELS,
        "preprocessings": DEFAULT_PREPROCESSINGS,
        "seeds": DEFAULT_SEEDS,
        "risk_horizons": DEFAULT_RISK_HORIZONS,
        "risk_event_kinds": DEFAULT_RISK_EVENT_KINDS,
        "risk_allow_quantiles": DEFAULT_RISK_ALLOW_QUANTILES,
        "seq_lens": [64, 128],
    },
    "mtf_decomposition": {
        "feature_groups": MTF_DECOMPOSITION_GROUPS,
        "point_models": ["Linear", "PatchTSTLike"],
        "risk_models": ["PatchTSTLike"],
        "preprocessings": ["seasonal_diff16", "winsor_025"],
        "seeds": [42, 2026],
        "risk_horizons": [16],
        "risk_event_kinds": ["absolute_move"],
        "risk_allow_quantiles": [0.55],
        "seq_lens": [64],
    },
    "algorithm_screen": {
        "feature_groups": ["mtf_full", "mtf_plus_shock_event", "mtf_plus_orderflow_proxy", "mtf_plus_full_available"],
        "point_models": DEFAULT_POINT_MODELS,
        "risk_models": ["PatchTSTLike"],
        "preprocessings": ["seasonal_diff16"],
        "seeds": [42, 2026],
        "risk_horizons": [16],
        "risk_event_kinds": ["absolute_move"],
        "risk_allow_quantiles": [0.55],
        "seq_lens": [64],
    },
    "feature_algorithm_matrix": {
        "feature_groups": DEFAULT_FEATURE_GROUPS,
        "point_models": DEFAULT_POINT_MODELS,
        "risk_models": ["PatchTSTLike"],
        "preprocessings": ["seasonal_diff16", "winsor_025", "linear_detrend+asinh_robust"],
        "seeds": [42, 2026, 777],
        "risk_horizons": [16],
        "risk_event_kinds": ["absolute_move"],
        "risk_allow_quantiles": [0.55],
        "seq_lens": [64],
    },
    "risk_gate_sensitivity": {
        "feature_groups": ["mtf_full", "mtf_plus_shock_event", "mtf_plus_orderflow_proxy", "mtf_plus_full_available"],
        "point_models": ["Linear", "PatchTSTLike", "TCNLike"],
        "risk_models": EXPANDED_RISK_MODELS,
        "preprocessings": ["seasonal_diff16"],
        "seeds": [42, 2026],
        "risk_horizons": DEFAULT_RISK_HORIZONS,
        "risk_event_kinds": DEFAULT_RISK_EVENT_KINDS,
        "risk_allow_quantiles": DEFAULT_RISK_ALLOW_QUANTILES,
        "seq_lens": [64],
    },
    "full_resource": {
        "feature_groups": DEFAULT_FEATURE_GROUPS,
        "point_models": DEFAULT_POINT_MODELS,
        "risk_models": EXPANDED_RISK_MODELS,
        "preprocessings": DEFAULT_PREPROCESSINGS,
        "seeds": DEFAULT_SEEDS,
        "risk_horizons": DEFAULT_RISK_HORIZONS,
        "risk_event_kinds": DEFAULT_RISK_EVENT_KINDS,
        "risk_allow_quantiles": DEFAULT_RISK_ALLOW_QUANTILES,
        "seq_lens": [64, 128],
    },
}

RAW_VARIABLE_ROWS = [
    ("timestamp", "15분 봉 시각", "시계열 정렬, 시간대 파생변수, train/validation/test 시간 분할 기준"),
    ("open/high/low/close", "15분 봉 OHLC 가격", "수익률, range, candle body/wick, 다음 봉 proxy candle 평가에 사용"),
    ("volume", "15분 거래량", "거래량 z-score, shock, signed volume, attention proxy 생성"),
    ("value", "15분 거래대금", "value z-score, liquidity, attention proxy 생성. 없으면 close*volume으로 보완"),
    ("ticker", "선택적 종목 코드", "다중 ticker 테이블에서 cross-market feature 결합에 사용"),
    ("text/macro/onchain/derivatives/social columns", "선택적 외부 mart 컬럼", "컬럼이 실제로 있을 때만 optional group에 자동 결합"),
]

DERIVED_VARIABLE_ROWS = [
    ("15분/1시간/4시간/16시간/2일 수익률", "log_return_1, return_4, return_16, return_64, return_192", "close 로그 차분과 rolling sum"),
    ("4시간/16시간/2일 변동성", "realized_vol_16, realized_vol_64, realized_vol_192", "15분 수익률 rolling std"),
    ("변동성 비율", "vol_ratio_16_64, vol_ratio_64_192", "단기 변동성 / 장기 변동성"),
    ("가격 위치와 추세", "price_z_192, trend_strength_64, trend_strength_192", "rolling z-score와 rolling mean 대비 괴리"),
    ("거래량·거래대금 위치", "volume_z_192, value_z_192", "log1p(volume/value)의 rolling z-score"),
    ("range 구조", "hl_range_pct, range_mean_192, range_z_96", "(high-low)/close와 rolling 평균/z-score"),
    ("momentum/reversal", "rsi_14_scaled, macd_gap_pct, macd_signal_gap_pct, reversal_4", "RSI, MACD, 단기 반전 proxy"),
    ("order-flow proxy", "candle_body_pct, upper/lower_wick_pct, signed_volume/value_proxy", "실제 호가창이 없을 때 캔들 구조와 거래량 방향으로 매수·매도 압력 근사"),
    ("shock/attention proxy", "abs_return_z_96, volume_shock_z_96, value_shock_z_96, attention_pressure_proxy", "급변·거래 폭증·관심도 대리 지표"),
    ("calendar cycle", "hour_sin/cos, day_sin/cos, korea/us_active_session", "24시간 코인 시장의 지역별 유동성 시간대 proxy"),
]


def configure_inline_matplotlib() -> None:
    diag.configure_inline_matplotlib()


def display_table(title: str, frame: pd.DataFrame, max_rows: int = 80) -> None:
    print(f"\n[{title}]")
    try:
        from IPython.display import display

        display(frame.head(max_rows))
    except Exception:
        print(frame.head(max_rows).to_string(index=False))


def display_markdown(text: str) -> None:
    diag.display_markdown(text)


def parse_csv(text: str) -> list[str]:
    return [item.strip() for item in str(text).split(",") if item.strip()]


def parse_int_csv(text: str) -> list[int]:
    return [int(item) for item in parse_csv(text)]


def parse_float_csv(text: str) -> list[float]:
    return [float(item) for item in parse_csv(text)]


def normalize_model_name(name: str) -> str:
    if name not in MODEL_ALIASES:
        raise KeyError(f"unknown model name: {name}. available={sorted(MODEL_ALIASES)}")
    return MODEL_ALIASES[name]


def unique_preserve(items: list[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result


def available_columns(features: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in columns if column in features.columns]


def extend_feature_groups(features: pd.DataFrame, groups12: dict[str, list[str]]) -> tuple[dict[str, list[str]], pd.DataFrame]:
    """Build 13번 feature group definitions on top of 12번 groups."""

    mtf_returns = ["log_return_1", "return_4", "return_16", "return_64", "return_192"]
    mtf_vol = ["realized_vol_16", "realized_vol_64", "realized_vol_192", "vol_ratio_16_64", "vol_ratio_64_192"]
    mtf_trend = ["price_z_192", "trend_strength_64", "trend_strength_192", "ema_gap_16", "ema_gap_64"]
    mtf_volume_range = ["volume_z_96", "value_z_96", "volume_z_192", "value_z_192", "hl_range_pct", "range_mean_192", "range_z_96"]
    required_core = ["log_return_1", "return_4", "realized_vol_16", "volume_z_96"]

    raw_groups = {
        "mtf_returns_only": mtf_returns,
        "mtf_volatility_only": required_core + mtf_vol,
        "mtf_trend_position_only": required_core + mtf_trend,
        "mtf_volume_range_only": ["log_return_1", "return_4", *mtf_volume_range],
        "mtf_returns_volatility": mtf_returns + mtf_vol,
        "mtf_returns_trend": mtf_returns + mtf_trend,
        "mtf_volatility_trend": required_core + mtf_vol + mtf_trend,
        "mtf_full": groups12.get("coin_multitimeframe_structure", mtf_returns + mtf_vol + mtf_trend + mtf_volume_range),
        "mtf_plus_momentum_reversal": groups12.get("coin_multitimeframe_structure", []) + groups12.get("coin_momentum_reversal", []),
        "mtf_plus_calendar_cycle": groups12.get("coin_multitimeframe_structure", []) + groups12.get("coin_calendar_cycle", []),
        "mtf_plus_shock_event": groups12.get("coin_multitimeframe_structure", []) + groups12.get("coin_shock_event", []),
        "mtf_plus_attention_proxy": groups12.get("coin_multitimeframe_structure", []) + groups12.get("coin_attention_proxy", []),
        "mtf_plus_liquidity_micro": groups12.get("coin_multitimeframe_structure", []) + groups12.get("coin_liquidity_micro", []),
        "mtf_plus_orderflow_proxy": groups12.get("coin_multitimeframe_structure", []) + groups12.get("coin_orderflow_proxy", []),
        "mtf_plus_full_available": groups12.get("coin_full_available", []),
        "mtf_plus_text_context": groups12.get("coin_multitimeframe_structure", []) + groups12.get("coin_text_context", []),
        "mtf_plus_cross_market": groups12.get("coin_multitimeframe_structure", []) + groups12.get("coin_cross_market", []),
        "mtf_plus_macro_proxy": groups12.get("coin_multitimeframe_structure", []) + groups12.get("coin_macro_proxy", []),
        "mtf_plus_onchain_proxy": groups12.get("coin_multitimeframe_structure", []) + groups12.get("coin_onchain_proxy", []),
        "mtf_plus_derivatives_proxy": groups12.get("coin_multitimeframe_structure", []) + groups12.get("coin_derivatives_proxy", []),
    }

    rows = []
    groups13 = dict(groups12)
    for name, columns in raw_groups.items():
        unique_columns = unique_preserve(columns)
        present = available_columns(features, unique_columns)
        missing = [column for column in unique_columns if column not in features.columns]
        optional_empty = name in {
            "mtf_plus_cross_market",
            "mtf_plus_macro_proxy",
            "mtf_plus_onchain_proxy",
            "mtf_plus_derivatives_proxy",
        } and len(present) <= len(groups12.get("coin_multitimeframe_structure", []))
        status = "included" if len(present) >= 4 and not optional_empty else "excluded"
        reason = "" if status == "included" else "usable columns too few or optional mart columns absent"
        rows.append(
            {
                "feature_group": name,
                "status": status,
                "n_features": len(present),
                "missing_count": len(missing),
                "reason": reason,
                "columns": ", ".join(present),
                "missing_examples": ", ".join(missing[:12]),
            }
        )
        if status == "included":
            groups13[name] = present
            base.FEATURE_SETS[name] = present
    return groups13, pd.DataFrame(rows)


def raw_variable_table() -> pd.DataFrame:
    return pd.DataFrame(RAW_VARIABLE_ROWS, columns=["raw_variable", "plain_meaning", "used_for"])


def derived_variable_table() -> pd.DataFrame:
    return pd.DataFrame(DERIVED_VARIABLE_ROWS, columns=["variable_family", "example_columns", "construction"])


def suite_config(args: argparse.Namespace) -> dict[str, list]:
    config = {key: list(value) for key, value in SUITE_PRESETS[args.suite].items()}
    if args.feature_groups:
        config["feature_groups"] = parse_csv(args.feature_groups)
    if args.point_models:
        config["point_models"] = parse_csv(args.point_models)
    if args.risk_models:
        config["risk_models"] = parse_csv(args.risk_models)
    if args.preprocessings:
        config["preprocessings"] = parse_csv(args.preprocessings)
    if args.seeds:
        config["seeds"] = parse_int_csv(args.seeds)
    if args.risk_horizons:
        config["risk_horizons"] = parse_int_csv(args.risk_horizons)
    if args.risk_event_kinds:
        config["risk_event_kinds"] = parse_csv(args.risk_event_kinds)
    if args.risk_allow_quantiles:
        config["risk_allow_quantiles"] = parse_float_csv(args.risk_allow_quantiles)
    if args.seq_lens:
        config["seq_lens"] = parse_int_csv(args.seq_lens)
    return config


def build_cases(args: argparse.Namespace, available_groups: dict[str, list[str]], config: dict[str, list]) -> pd.DataFrame:
    rows = []
    for feature_group in config["feature_groups"]:
        if feature_group not in available_groups:
            continue
        for preprocessing in config["preprocessings"]:
            for point_model_label in config["point_models"]:
                point_model = normalize_model_name(point_model_label)
                for risk_model_label in config["risk_models"]:
                    risk_model = normalize_model_name(risk_model_label)
                    for seed in config["seeds"]:
                        for seq_len in config["seq_lens"]:
                            for risk_horizon in config["risk_horizons"]:
                                for risk_event_kind in config["risk_event_kinds"]:
                                    for risk_allow_quantile in config["risk_allow_quantiles"]:
                                        rows.append(
                                            {
                                                "feature_group": feature_group,
                                                "preprocessing": preprocessing,
                                                "point_model_label": point_model_label,
                                                "risk_model_label": risk_model_label,
                                                "point_model": point_model,
                                                "risk_model": risk_model,
                                                "seed": int(seed),
                                                "seq_len": int(seq_len),
                                                "risk_horizon": int(risk_horizon),
                                                "risk_event_kind": risk_event_kind,
                                                "risk_allow_quantile": float(risk_allow_quantile),
                                            }
                                        )
    cases = pd.DataFrame(rows)
    if args.max_cases > 0:
        cases = cases.head(args.max_cases)
    if cases.empty:
        raise ValueError("No executable 13번 cases. Check feature group availability and filters.")
    cases.insert(0, "case_no", np.arange(1, len(cases) + 1))
    return cases


def case_to_backend_args(args: argparse.Namespace, row: pd.Series) -> argparse.Namespace:
    case_args = copy.copy(args)
    case_args.seq_len = int(row["seq_len"])
    case_args.risk_horizon = int(row["risk_horizon"])
    case_args.risk_event_kind = str(row["risk_event_kind"])
    case_args.risk_allow_quantile = float(row["risk_allow_quantile"])
    case_args.feature_set = str(row["feature_group"])
    return case_args


def case_to_backend_case(row: pd.Series) -> dict[str, object]:
    return {
        "feature_group": str(row["feature_group"]),
        "preprocessing": str(row["preprocessing"]),
        "point_model": str(row["point_model"]),
        "risk_model": str(row["risk_model"]),
        "seed": int(row["seed"]),
    }


def active_batch_size(args: argparse.Namespace) -> int:
    if args.suite == "full_resource" and args.large_batch:
        return int(args.large_batch)
    return int(args.batch_size)


def run_one_case(row: pd.Series, features: pd.DataFrame, profile, args: argparse.Namespace) -> dict[str, object]:
    case_args = case_to_backend_args(args, row)
    case_args.batch_size = active_batch_size(args)
    backend_case = case_to_backend_case(row)
    point_result, _val_pred, test_pred, point_splits = fusion12.run_point_branch(backend_case, features, profile, case_args)
    risk_result, _val_prob, test_probability, _risk_splits, risk_gate_cutoff = fusion12.run_risk_branch(
        backend_case,
        features,
        profile,
        case_args,
    )
    fusion_result, policy_frame = fusion12.evaluate_fusion_policy(
        backend_case,
        point_splits["test"],
        test_pred,
        test_probability,
        risk_gate_cutoff,
        case_args,
    )
    policy_by_name = policy_frame.set_index("policy")
    result = {
        **fusion_result,
        "case_no": int(row["case_no"]),
        "point_model_label": row["point_model_label"],
        "risk_model_label": row["risk_model_label"],
        "seq_len": int(row["seq_len"]),
        "risk_horizon": int(row["risk_horizon"]),
        "risk_event_kind": row["risk_event_kind"],
        "risk_allow_quantile": float(row["risk_allow_quantile"]),
        "point_actual_batch_size": int(point_result.get("batch_size", case_args.batch_size)),
        "risk_actual_batch_size": int(risk_result.get("batch_size", case_args.batch_size)),
        "risk_average_precision": risk_result["average_precision"],
        "risk_average_precision_lift": risk_result["average_precision_lift"],
        "risk_roc_auc": risk_result["roc_auc"],
        "risk_brier_skill_score": risk_result["brier_skill_score"],
        "risk_expected_calibration_error": risk_result["expected_calibration_error"],
        "risk_event_rate": risk_result["event_rate"],
        "risk_gate_only_mdd": float(policy_by_name.loc["risk_gate_only", "mdd"]),
        "risk_gate_only_cumulative_return": float(policy_by_name.loc["risk_gate_only", "cumulative_return"]),
        "low_trade_warning": bool(fusion_result["point_plus_risk_gate_trade_count"] < 5 or fusion_result["fusion_signal_share"] < 0.02),
    }
    if args.case_plots:
        fusion12.show_fusion_diagnostics(
            backend_case,
            point_splits["test"],
            test_pred,
            test_probability,
            risk_gate_cutoff,
            policy_frame,
            case_args,
        )
    return result


def summarize_results(summary: pd.DataFrame) -> None:
    if summary.empty:
        print("[summary] no successful case")
        return
    ordered = summary.sort_values(
        ["point_plus_risk_gate_mdd", "point_plus_risk_gate_cumulative_return", "point_copy_risk_ratio"],
        ascending=[False, False, True],
    )
    leaderboard_columns = [
        "case_no",
        "feature_group",
        "preprocessing",
        "point_model_label",
        "risk_model_label",
        "seed",
        "seq_len",
        "risk_horizon",
        "risk_event_kind",
        "risk_allow_quantile",
        "point_copy_risk_ratio",
        "point_direction_accuracy",
        "point_variance_ratio",
        "risk_average_precision_lift",
        "point_only_mdd",
        "risk_gate_only_mdd",
        "point_plus_risk_gate_mdd",
        "point_plus_risk_gate_cumulative_return",
        "fusion_signal_share",
        "point_plus_risk_gate_trade_count",
        "low_trade_warning",
        "point_actual_batch_size",
        "risk_actual_batch_size",
    ]
    print("\n[13번 top 30 leaderboard]")
    print(ordered[[column for column in leaderboard_columns if column in ordered.columns]].head(30).to_string(index=False))

    feature_avg = (
        summary.groupby("feature_group", as_index=False)
        .agg(
            cases=("case_no", "count"),
            mean_fusion_mdd=("point_plus_risk_gate_mdd", "mean"),
            mean_fusion_return=("point_plus_risk_gate_cumulative_return", "mean"),
            mean_copy_risk=("point_copy_risk_ratio", "mean"),
            mean_ap_lift=("risk_average_precision_lift", "mean"),
            mean_signal_share=("fusion_signal_share", "mean"),
            low_trade_warnings=("low_trade_warning", "sum"),
        )
        .sort_values(["mean_fusion_mdd", "mean_fusion_return"], ascending=[False, False])
    )
    model_avg = (
        summary.groupby("point_model_label", as_index=False)
        .agg(
            cases=("case_no", "count"),
            mean_fusion_mdd=("point_plus_risk_gate_mdd", "mean"),
            mean_fusion_return=("point_plus_risk_gate_cumulative_return", "mean"),
            mean_copy_risk=("point_copy_risk_ratio", "mean"),
            mean_direction=("point_direction_accuracy", "mean"),
            mean_variance_ratio=("point_variance_ratio", "mean"),
        )
        .sort_values(["mean_fusion_mdd", "mean_copy_risk"], ascending=[False, True])
    )
    display_table("feature group average summary", feature_avg, max_rows=80)
    display_table("model family average summary", model_avg, max_rows=80)
    show_decision_charts(summary)


def show_decision_charts(summary: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    if summary.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(20, 5), dpi=140)
    feature_avg = (
        summary.groupby("feature_group", as_index=False)
        .agg(mean_mdd=("point_plus_risk_gate_mdd", "mean"), mean_return=("point_plus_risk_gate_cumulative_return", "mean"))
        .sort_values("mean_mdd", ascending=True)
    )
    axes[0].barh(feature_avg["feature_group"], feature_avg["mean_mdd"])
    axes[0].set_title("Feature group average fusion MDD")
    axes[0].set_xlabel("MDD; closer to 0 is better")

    model_avg = (
        summary.groupby("point_model_label", as_index=False)
        .agg(mean_mdd=("point_plus_risk_gate_mdd", "mean"), mean_copy=("point_copy_risk_ratio", "mean"))
        .sort_values("mean_mdd", ascending=True)
    )
    axes[1].barh(model_avg["point_model_label"], model_avg["mean_copy"])
    axes[1].axvline(1.0, color="black", linestyle="--")
    axes[1].set_title("Model average copy-risk")
    axes[1].set_xlabel("MAE / persistence MAE")

    scatter = summary.copy()
    axes[2].scatter(scatter["fusion_signal_share"], scatter["point_plus_risk_gate_cumulative_return"], alpha=0.65)
    axes[2].axhline(0.0, color="black", linestyle="--")
    axes[2].set_title("Final decision chart")
    axes[2].set_xlabel("fusion active share")
    axes[2].set_ylabel("fusion return after cost")
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.suptitle("13번 feature x algorithm x risk-gate decision summary")
    fusion12.show_figure(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="13번 feature algorithm resource 공동 최적화")
    parser.add_argument("--db", default=None)
    parser.add_argument("--table", default="btc_15m_advance")
    parser.add_argument("--ticker", default=None)
    parser.add_argument("--profile", default="school_4090_15gb")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--parallel-slot", choices=sorted(obj.PARALLEL_RESOURCE_SLOTS), default="exclusive")
    parser.add_argument(
        "--suite",
        choices=["dry_plan", "mtf_decomposition", "algorithm_screen", "feature_algorithm_matrix", "risk_gate_sensitivity", "full_resource"],
        default="feature_algorithm_matrix",
    )
    parser.add_argument("--feature-groups", default="")
    parser.add_argument("--point-models", default="")
    parser.add_argument("--risk-models", default="")
    parser.add_argument("--preprocessings", default="")
    parser.add_argument("--seeds", default="")
    parser.add_argument("--risk-horizons", default="")
    parser.add_argument("--risk-event-kinds", default="")
    parser.add_argument("--risk-allow-quantiles", default="")
    parser.add_argument("--seq-lens", default="")
    parser.add_argument("--cross-tickers", default="KRW-ETH,KRW-XRP,KRW-SOL")
    parser.add_argument("--point-objective", default="balanced_composite")
    parser.add_argument("--event-quantile", type=float, default=0.70)
    parser.add_argument("--normalization", default="window_standard")
    parser.add_argument("--optimizer", default="adamw")
    parser.add_argument("--scheduler", default="cosine")
    parser.add_argument("--gradient-policy", default="clip1")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--objective-warmup-epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--large-batch", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--max-windows", type=int, default=0)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--cost-bps", type=float, default=14.0)
    parser.add_argument("--min-signal-bps", type=float, default=3.0)
    # 2026-06-28 안정화: case별 plot은 기본 OFF (출력 폭주로 인한 노트북 저장 실패 방지).
    # 대표 그래프가 필요하면 --case-plots 로 명시적으로 켠다.
    parser.add_argument("--case-plots", action=argparse.BooleanOptionalAction, default=False)
    # 2026-06-28 안정화: DataLoader worker 수 (기본 0).
    # 0이면 worker process를 띄우지 않아 /dev/shm(공유 메모리)을 전혀 쓰지 않으므로
    # "RuntimeError: unable to allocate shared memory(shm)" 를 원천 차단한다.
    # WSL2 등 RAM/shm이 작은 환경에서는 반드시 0을 유지한다.
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-failure", action="store_true")
    return parser.parse_known_args(argv)[0]


def main(argv: list[str] | None = None) -> None:
    configure_inline_matplotlib()
    args = parse_args(argv)
    profile = base.build_resource_profile(args.profile, args.device)
    profile, slot_details = obj.apply_parallel_resource_slot(profile, args.parallel_slot)
    # 2026-06-28 안정화 override: slot이 정한 num_workers(예: exclusive=4)를 사용자 지정값으로 덮는다.
    # base.make_loader 가 profile.num_workers / profile.pin_memory 를 직접 읽으므로
    # 여기서 0으로 낮추면 모든 DataLoader가 공유 메모리를 쓰지 않는다.
    if args.num_workers != profile.num_workers:
        profile = dataclasses.replace(
            profile,
            num_workers=max(0, int(args.num_workers)),
            pin_memory=profile.pin_memory and int(args.num_workers) > 0,
        )
        slot_details["num_workers_override"] = int(args.num_workers)
        slot_details["pin_memory"] = profile.pin_memory
    base.apply_resource_profile(profile)
    environment = base.log_environment(profile)
    print("[environment]", json.dumps(environment, ensure_ascii=False, indent=2))
    print("[slot]", json.dumps(slot_details, ensure_ascii=False, indent=2))
    print(f"[batch-policy] requested={active_batch_size(args)} oom_retry=2048->1024->512->256->... actual is logged per branch")

    features, groups12, raw = fusion12.load_feature_frame(args)
    feature_groups, availability = extend_feature_groups(features, groups12)
    config = suite_config(args)
    cases = build_cases(args, feature_groups, config)

    display_markdown(
        "# 13번 실행 개요\n\n"
        "12번의 point+risk fusion 계보를 유지하면서, multi-timeframe 변수 분해와 알고리즘 확장을 함께 확인한다.\n"
        "완료된 12번 노트북은 수정하지 않고 읽기 전용 backend로만 사용한다."
    )
    display_table("RAW variable explanation", raw_variable_table())
    display_table("derived variable explanation", derived_variable_table())
    display_table("13 feature group availability", availability, max_rows=80)
    display_table("available feature group definition", fusion12.feature_group_table({key: feature_groups[key] for key in feature_groups if key in DEFAULT_FEATURE_GROUPS}), max_rows=120)
    display_table("missingness audit", obj.build_missingness_audit(raw, features, "13 feature-algorithm-resource labels"))
    print(f"\n[plan] suite={args.suite} executable_cases={len(cases)}")
    print(cases.head(120).to_string(index=False))
    print("[case-count-by-feature]")
    print(cases.groupby("feature_group").size().sort_values(ascending=False).to_string())
    print("[case-count-by-model]")
    print(cases.groupby("point_model_label").size().sort_values(ascending=False).to_string())

    if args.suite == "dry_plan" or args.dry_run:
        print("[dry-plan] 계획, 변수 정의, optional group 제외 사유만 출력하고 학습은 실행하지 않는다.")
        return

    base.apply_window_preprocessing = diag.apply_preprocessing_pipeline
    rows: list[dict[str, object]] = []
    for _, row in cases.iterrows():
        print(
            f"\n[case {int(row['case_no'])}/{len(cases)}] "
            f"{row['feature_group']} / {row['preprocessing']} / "
            f"{row['point_model_label']} + {row['risk_model_label']} / "
            f"seed{int(row['seed'])} / seq{int(row['seq_len'])} / "
            f"{row['risk_event_kind']} h{int(row['risk_horizon'])} q{float(row['risk_allow_quantile']):.2f}"
        )
        try:
            rows.append(run_one_case(row, features, profile, args))
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and int(args.hidden) > 96:
                print(f"[hidden-fallback] hidden {args.hidden} -> 96 after OOM")
                retry_args = copy.copy(args)
                retry_args.hidden = 96
                rows.append(run_one_case(row, features, profile, retry_args))
                continue
            if not args.continue_on_failure:
                raise
            print(f"[case failed] {row.to_dict()}: {exc}")
        except Exception as exc:
            if not args.continue_on_failure:
                raise
            print(f"[case failed] {row.to_dict()}: {exc}")
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    summary = pd.DataFrame(rows)
    summarize_results(summary)


if __name__ == "__main__":
    main()
