"""점예측 branch + 위험 gate fusion 정책과 진단.

12번에서 옮겼다. `run_point_branch`/`run_risk_branch`/`evaluate_fusion_policy`/
`show_fusion_diagnostics`/`show_feature_summary`/`position_metrics`/`max_drawdown`/
`make_case_args`/`build_cases`/`build_inline_report`를 담는다.

12번의 indirection 연결:
- `obj.run_case` -> engine.point.run_case
- `base.*` -> engine.models / engine.resources
- `risk.build_risk_windows`/`risk.prepare_event_splits` -> engine.windows
  (엔진에서는 윈도우 빌더가 windows 모듈에 있다)
- `risk.train_event_model` 등 분류 함수 -> engine.risk
- `feature_group_table` -> engine.features
- `show_figure`/`display_table`/`display_markdown` -> engine.display
"""

from __future__ import annotations

import argparse
import copy
import json

import numpy as np
import pandas as pd
import torch

from . import display, features as features_module, models, point, resources, risk, windows


def make_case_args(args: argparse.Namespace, feature_group: str) -> argparse.Namespace:
    case_args = copy.copy(args)
    case_args.feature_set = feature_group
    return case_args


def build_cases(args: argparse.Namespace, available_groups: dict[str, list[str]]) -> list[dict[str, object]]:
    requested_groups = [item.strip() for item in args.feature_groups.split(",") if item.strip()]
    groups = [group for group in requested_groups if group in available_groups]
    if not groups:
        raise ValueError(f"사용 가능한 feature group이 없다. requested={requested_groups}, available={sorted(available_groups)}")
    point_models = [item.strip() for item in args.point_models.split(",") if item.strip()]
    risk_models = [item.strip() for item in args.risk_models.split(",") if item.strip()]
    preprocessings = [item.strip() for item in args.preprocessings.split(",") if item.strip()]
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]

    cases: list[dict[str, object]] = []
    for feature_group in groups:
        for preprocessing in preprocessings:
            for point_model in point_models:
                for risk_model in risk_models:
                    for seed in seeds:
                        cases.append(
                            {
                                "feature_group": feature_group,
                                "preprocessing": preprocessing,
                                "point_model": point_model,
                                "risk_model": risk_model,
                                "seed": seed,
                            }
                        )
    return cases[: args.max_cases] if args.max_cases > 0 else cases


def run_point_branch(case: dict[str, object], features: pd.DataFrame, profile, args: argparse.Namespace):
    point_case = {
        "model": case["point_model"],
        "preprocessing": case["preprocessing"],
        "objective": args.point_objective,
        "seed": case["seed"],
    }
    point_args = make_case_args(args, str(case["feature_group"]))
    result, val_pred, test_pred, splits = point.run_case(point_case, features, profile, point_args)
    result["branch"] = "point"
    result["feature_group"] = case["feature_group"]
    result["risk_model"] = case["risk_model"]
    return result, val_pred, test_pred, splits


def run_risk_branch(case: dict[str, object], features: pd.DataFrame, profile, args: argparse.Namespace):
    models.set_seed(int(case["seed"]))
    feature_columns = resources.FEATURE_SETS[str(case["feature_group"])]
    risk_args = make_case_args(args, str(case["feature_group"]))
    data = windows.build_risk_windows(
        features,
        feature_columns,
        args.seq_len,
        args.risk_horizon,
        str(case["preprocessing"]),
        args.normalization,
        args.max_windows,
        args.stride,
    )
    splits, event_score_threshold = windows.prepare_event_splits(
        data,
        args.train_ratio,
        args.val_ratio,
        args.risk_event_kind,
        args.event_quantile,
    )
    batch_size = args.batch_size or profile.batch_size or 32
    while True:
        try:
            model = models.make_model(
                str(case["risk_model"]),
                args.seq_len,
                len(feature_columns),
                int(args.hidden),
            )
            param_count = sum(parameter.numel() for parameter in model.parameters())
            model, curves = risk.train_event_model(model, splits, profile, risk_args, batch_size)
            val_probability = risk.predict_event_probability(model, splits["val"], profile, batch_size)
            test_probability = risk.predict_event_probability(model, splits["test"], profile, batch_size)
            break
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower() or batch_size <= 4:
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            next_batch = max(4, batch_size // 2)
            print(f"[oom-retry] risk batch_size {batch_size} -> {next_batch}")
            batch_size = next_batch

    validation_threshold = risk.select_validation_threshold(splits["val"]["y"], val_probability)
    gate_cutoff = float(np.quantile(val_probability, args.risk_allow_quantile))
    metrics = risk.classification_metrics(splits["test"]["y"], test_probability, validation_threshold)
    calibration = risk.calibration_table(splits["test"]["y"], test_probability)
    selective = risk.selective_risk_curve(splits["test"]["y"], test_probability)
    train_event_rate = float(np.mean(splits["train"]["y"]))
    climatology_brier = float(np.mean((train_event_rate - splits["test"]["y"]) ** 2))
    brier_skill = 1.0 - metrics["brier_score"] / max(1e-12, climatology_brier)
    average_precision_lift = (
        metrics["average_precision"] / metrics["event_rate"]
        if metrics["event_rate"] > 0 and np.isfinite(metrics["average_precision"])
        else float("nan")
    )
    result = {
        "case_id": (
            f"{case['risk_model']}__{case['feature_group']}__{case['preprocessing']}"
            f"__{args.risk_event_kind}__horizon{args.risk_horizon}__seed{case['seed']}"
        ),
        "branch": "risk",
        "feature_group": case["feature_group"],
        "preprocessing": case["preprocessing"],
        "risk_model": case["risk_model"],
        "point_model": case["point_model"],
        "seed": int(case["seed"]),
        "risk_event_kind": args.risk_event_kind,
        "risk_horizon": int(args.risk_horizon),
        "event_score_threshold": event_score_threshold,
        "validation_probability_threshold": validation_threshold,
        "risk_gate_cutoff": gate_cutoff,
        "train_event_rate": train_event_rate,
        "param_count": int(param_count),
        "batch_size": int(batch_size),
        "expected_calibration_error": risk.expected_calibration_error(calibration),
        "climatology_brier_score": climatology_brier,
        "brier_skill_score": brier_skill,
        "average_precision_lift": average_precision_lift,
        "selective_error_at_50pct": float(
            selective.iloc[(selective["coverage"] - 0.5).abs().argmin()]["classification_error"]
        ),
        **metrics,
    }
    if args.case_plots:
        risk.show_event_diagnostics(
            curves,
            splits["test"],
            test_probability,
            validation_threshold,
            str(result["case_id"]),
        )
    return result, val_probability, test_probability, splits, gate_cutoff


def max_drawdown(equity: np.ndarray) -> float:
    if len(equity) == 0:
        return 0.0
    running_max = np.maximum.accumulate(equity)
    drawdown = equity / np.where(running_max == 0, 1.0, running_max) - 1.0
    return float(np.min(drawdown))


def position_metrics(actual_return: np.ndarray, position: np.ndarray, cost: float) -> dict[str, float]:
    position = position.astype(float)
    turnover = np.abs(np.diff(np.r_[0.0, position]))
    pnl = position * actual_return - turnover * cost
    equity = np.exp(np.cumsum(pnl))
    active_share = float(np.mean(position > 0))
    trade_count = int(np.sum(turnover > 0))
    mean_return = float(np.mean(pnl))
    downside = pnl[pnl < 0]
    sortino = mean_return / max(1e-12, float(np.std(downside)) if len(downside) else 1e-12)
    return {
        "cumulative_return": float(equity[-1] - 1.0) if len(equity) else 0.0,
        "mdd": max_drawdown(equity),
        "sortino_proxy": sortino,
        "active_share": active_share,
        "trade_count": trade_count,
        "turnover": float(np.sum(turnover)),
        "mean_step_pnl": mean_return,
    }


def align_point_risk(point_split, point_pred, risk_probability, risk_split):
    """point/risk를 decision_timestamp(진입 봉)로 inner-join 정렬한다.

    fix①: 기존 `[-n:]` 꼬리 절단은 horizon만큼 적고 시점이 밀린 risk를 다른 결정시점의
    point와 짝지었다. point 진입 봉(end)과 risk 진입 봉(end-1)이 같은 close[T]를 가리킬 때만
    join한다. evaluate와 진단 그래프가 이 함수를 공유해 같은 정렬을 쓴다(fix②-diag).

    dup-guard(fix③): decision_timestamp가 중복이면 dict 매핑이 마지막 행만 남겨 many-to-one
    오정렬을 조용히 만든다. 그래서 join 전에 양쪽 유니크를 강제하고, 아니면 큰 소리로 실패한다.
    """
    if risk_split is None or "decision_timestamp" not in risk_split:
        raise ValueError(
            "fusion 정렬에는 risk_split(test)와 그 decision_timestamp가 필요하다. "
            "run_risk_branch가 돌려준 splits['test']를 넘겨라."
        )
    point_ts = np.asarray(point_split["decision_timestamp"]).astype(str)
    risk_ts = np.asarray(risk_split["decision_timestamp"]).astype(str)
    if len(np.unique(point_ts)) != len(point_ts):
        raise ValueError("point decision_timestamp에 중복이 있다 — ticker 필터/유니크 timestamp를 보장하라.")
    if len(np.unique(risk_ts)) != len(risk_ts):
        raise ValueError("risk decision_timestamp에 중복이 있다 — ticker 필터/유니크 timestamp를 보장하라.")
    risk_pos = {ts: i for i, ts in enumerate(risk_ts)}
    point_idx = [i for i, ts in enumerate(point_ts) if ts in risk_pos]
    risk_idx = [risk_pos[point_ts[i]] for i in point_idx]
    n = len(point_idx)
    if n == 0:
        raise ValueError(
            f"fusion 정렬 겹침이 0이다 (point={len(point_ts)}, risk={len(risk_ts)}). 이 케이스는 점수화하지 않는다."
        )
    point_idx_arr = np.asarray(point_idx, dtype=int)
    risk_idx_arr = np.asarray(risk_idx, dtype=int)
    aligned_point_split = {key: np.asarray(value)[point_idx_arr] for key, value in point_split.items()}
    pred = np.asarray(point_pred)[point_idx_arr]
    risk_prob = np.asarray(risk_probability)[risk_idx_arr]
    assert len(aligned_point_split["y"]) == len(pred) == len(risk_prob) == n
    info = {
        "n_aligned": n,
        "n_point_dropped": int(len(point_ts) - n),
        "n_risk_dropped": int(len(risk_ts) - n),
        "low_overlap_warning": bool(n < 0.5 * len(point_ts)),
    }
    return aligned_point_split, pred, risk_prob, info


def evaluate_fusion_policy(
    case: dict[str, object],
    point_split,
    point_pred: np.ndarray,
    risk_probability: np.ndarray,
    risk_gate_cutoff: float,
    args: argparse.Namespace,
    risk_split,
) -> tuple[dict[str, object], pd.DataFrame]:
    # risk_split은 필수다(정렬에 risk decision_timestamp가 필요). 선택적처럼 보이지 않도록 기본값을 두지 않는다.
    aligned_point_split, pred, risk_prob, info = align_point_risk(
        point_split, point_pred, risk_probability, risk_split
    )
    n = info["n_aligned"]
    n_point_dropped = info["n_point_dropped"]
    n_risk_dropped = info["n_risk_dropped"]
    low_overlap_warning = info["low_overlap_warning"]
    if n_point_dropped or n_risk_dropped:
        print(
            f"[fusion-align] {case.get('feature_group')}/{case.get('preprocessing')}: "
            f"aligned={n}, point_dropped={n_point_dropped}, risk_dropped={n_risk_dropped}"
        )
    actual = aligned_point_split["y"]
    cost = args.cost_bps / 10000.0
    signal_threshold = max(args.min_signal_bps / 10000.0, cost)
    risk_allowed = risk_prob <= risk_gate_cutoff
    point_long = pred > signal_threshold
    risk_only_hold = risk_allowed
    fusion_long = point_long & risk_allowed

    policy_rows = []
    for policy_name, position in [
        ("buy_hold", np.ones(n, dtype=bool)),
        ("point_only", point_long),
        ("risk_gate_only", risk_only_hold),
        ("point_plus_risk_gate", fusion_long),
        ("no_trade", np.zeros(n, dtype=bool)),
    ]:
        metrics = position_metrics(actual, position, cost)
        policy_rows.append(
            {
                "policy": policy_name,
                **metrics,
            }
        )
    policy_frame = pd.DataFrame(policy_rows)
    point_metrics = resources.evaluate_predictions(aligned_point_split, pred)
    fusion_metrics = {
        "case_id": (
            f"{case['feature_group']}__{case['preprocessing']}__{case['point_model']}"
            f"__{case['risk_model']}__seed{case['seed']}"
        ),
        **case,
        "n_aligned": int(n),
        "n_point_dropped": n_point_dropped,
        "n_risk_dropped": n_risk_dropped,
        "low_overlap_warning": low_overlap_warning,
        "risk_gate_cutoff": float(risk_gate_cutoff),
        "risk_allowed_share": float(np.mean(risk_allowed)),
        "point_signal_share": float(np.mean(point_long)),
        "fusion_signal_share": float(np.mean(fusion_long)),
        "point_mae_krw": point_metrics["mae_krw"],
        "point_copy_risk_ratio": point_metrics["copy_risk_ratio"],
        "point_direction_accuracy": point_metrics["direction_accuracy"],
        "point_variance_ratio": point_metrics["variance_ratio"],
    }
    for _, row in policy_frame.iterrows():
        prefix = str(row["policy"])
        for metric in ["cumulative_return", "mdd", "sortino_proxy", "active_share", "trade_count", "turnover"]:
            fusion_metrics[f"{prefix}_{metric}"] = float(row[metric])
    return fusion_metrics, policy_frame


def show_fusion_diagnostics(
    case: dict[str, object],
    point_split,
    point_pred: np.ndarray,
    risk_probability: np.ndarray,
    risk_gate_cutoff: float,
    policy_frame: pd.DataFrame,
    args: argparse.Namespace,
    risk_split,
) -> None:
    import matplotlib.pyplot as plt

    # fix②-diag: 그래프도 지표와 동일하게 decision_timestamp로 정렬한 샘플로 그린다.
    # 기존 `[-n:]` 꼬리 절단은 horizon>1에서 point/risk를 다른 시점으로 어긋나게 그렸다.
    aligned_point_split, pred_all, risk_prob_all, _ = align_point_risk(
        point_split, point_pred, risk_probability, risk_split
    )
    actual_all = aligned_point_split["y"]
    n = min(320, len(pred_all))
    actual = actual_all[-n:]
    pred = pred_all[-n:]
    risk_prob = risk_prob_all[-n:]
    cost = args.cost_bps / 10000.0
    signal_threshold = max(args.min_signal_bps / 10000.0, cost)
    risk_allowed = risk_prob <= risk_gate_cutoff
    point_long = pred > signal_threshold
    fusion_long = point_long & risk_allowed
    x = np.arange(n)

    fig, axes = plt.subplots(3, 2, figsize=(17, 12), dpi=140)
    axes[0, 0].plot(x, actual, label="actual next return")
    axes[0, 0].plot(x, pred, label="point prediction")
    axes[0, 0].axhline(0.0, color="black", linewidth=0.8)
    axes[0, 0].axhline(signal_threshold, color="tab:orange", linestyle="--", label="cost/signal threshold")
    axes[0, 0].set_xlabel("test time index")
    axes[0, 0].set_ylabel("next log return")
    axes[0, 0].set_title("Point forecast from 10번 branch")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(x, risk_prob, label="absolute-move risk probability")
    axes[0, 1].axhline(risk_gate_cutoff, color="black", linestyle="--", label="risk gate cutoff")
    axes[0, 1].fill_between(x, 0, 1, where=risk_allowed, alpha=0.12, label="allowed zone")
    axes[0, 1].set_xlabel("test time index")
    axes[0, 1].set_ylabel("risk probability")
    axes[0, 1].set_title("Risk guardrail from 11번 branch")
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].plot(x, point_long.astype(float), label="point-only long")
    axes[1, 0].plot(x, fusion_long.astype(float), label="point + risk gate long")
    axes[1, 0].set_xlabel("test time index")
    axes[1, 0].set_ylabel("position")
    axes[1, 0].set_title("Decision mask")
    axes[1, 0].legend(fontsize=8)

    for policy_name, position in [
        ("buy_hold", np.ones(n, dtype=bool)),
        ("point_only", point_long),
        ("risk_gate_only", risk_allowed),
        ("point_plus_risk_gate", fusion_long),
    ]:
        turnover = np.abs(np.diff(np.r_[0.0, position.astype(float)]))
        pnl = position.astype(float) * actual - turnover * cost
        axes[1, 1].plot(x, np.exp(np.cumsum(pnl)) - 1.0, label=policy_name)
    axes[1, 1].set_xlabel("test time index")
    axes[1, 1].set_ylabel("cumulative return proxy")
    axes[1, 1].set_title("Transaction-cost-aware policy comparison")
    axes[1, 1].legend(fontsize=8)

    axes[2, 0].scatter(risk_prob, np.abs(actual), s=15, alpha=0.45)
    axes[2, 0].axvline(risk_gate_cutoff, color="black", linestyle="--")
    axes[2, 0].set_xlabel("predicted risk probability")
    axes[2, 0].set_ylabel("realized absolute return")
    axes[2, 0].set_title("Risk probability versus realized movement")

    mdd_rows = policy_frame[policy_frame["policy"] != "no_trade"].copy()
    axes[2, 1].barh(mdd_rows["policy"], mdd_rows["mdd"])
    axes[2, 1].set_xlabel("MDD")
    axes[2, 1].set_title("MDD by policy")
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    fig.suptitle(
        f"{case['feature_group']} / {case['preprocessing']} / "
        f"{case['point_model']} + {case['risk_model']} / seed{case['seed']}"
    )
    display.show_figure(fig)
    display.display_table("policy preview", policy_frame)


def show_feature_summary(
    summary: pd.DataFrame,
    min_active_share: float = 0.05,
    min_trade_count: int = 5,
) -> None:
    import matplotlib.pyplot as plt

    if summary.empty:
        return
    # fix③: 거래를 거의 안 한 케이스는 MDD가 0에 가까워 방어 1등으로 둔갑한다.
    # 활동 하한(active share / trade count)을 통과한 케이스만 우승 후보로 줄세우고,
    # 미달 케이스는 별도 표에 "비활동(우승 자격 없음)"으로 분리해 보여준다.
    qualified_mask = (
        (summary["point_plus_risk_gate_active_share"] >= min_active_share)
        & (summary["point_plus_risk_gate_trade_count"] >= min_trade_count)
    )
    qualified = summary[qualified_mask]
    disqualified = summary[~qualified_mask]
    columns = [
        "feature_group",
        "preprocessing",
        "point_model",
        "risk_model",
        "seed",
        "point_copy_risk_ratio",
        "point_direction_accuracy",
        "point_variance_ratio",
        "risk_allowed_share",
        "fusion_signal_share",
        "point_plus_risk_gate_active_share",
        "point_plus_risk_gate_trade_count",
        "point_only_mdd",
        "point_plus_risk_gate_mdd",
        "point_only_cumulative_return",
        "point_plus_risk_gate_cumulative_return",
    ]
    print(
        f"\n[12번 feature guardrail leaderboard] "
        f"활동 하한 active_share>={min_active_share}, trade_count>={min_trade_count} 통과 "
        f"{len(qualified)}/{len(summary)}건만 우승 후보."
    )
    if qualified.empty:
        print("  (경고) 활동 하한을 통과한 케이스가 없다 — gate가 사실상 거래를 막고 있다. 우승자를 뽑지 않는다.")
    else:
        ordered = qualified.sort_values(
            ["point_plus_risk_gate_mdd", "point_copy_risk_ratio", "point_plus_risk_gate_cumulative_return"],
            ascending=[False, True, False],
        )
        print(ordered[columns].head(40).to_string(index=False))
    if not disqualified.empty:
        flagged = disqualified.sort_values("point_plus_risk_gate_active_share")
        display.display_table(
            "비활동/저거래 케이스 (우승 자격 없음 — MDD≈0은 방어가 아니라 미거래)",
            flagged[columns],
            max_rows=40,
        )

    grouped = (
        summary.groupby("feature_group", as_index=False)
        .agg(
            mean_copy_risk=("point_copy_risk_ratio", "mean"),
            mean_direction=("point_direction_accuracy", "mean"),
            mean_variance_ratio=("point_variance_ratio", "mean"),
            mean_point_mdd=("point_only_mdd", "mean"),
            mean_fusion_mdd=("point_plus_risk_gate_mdd", "mean"),
            mean_fusion_return=("point_plus_risk_gate_cumulative_return", "mean"),
            mean_signal_share=("fusion_signal_share", "mean"),
        )
        .sort_values(["mean_fusion_mdd", "mean_copy_risk"], ascending=[False, True])
    )
    display.display_table("feature group aggregate", grouped, max_rows=80)

    fig, axes = plt.subplots(1, 3, figsize=(19, 5), dpi=140)
    axes[0].barh(grouped["feature_group"], grouped["mean_copy_risk"])
    axes[0].axvline(1.0, color="black", linestyle="--")
    axes[0].set_xlabel("MAE / persistence MAE")
    axes[0].set_title("Point branch copy-risk by feature group")

    axes[1].barh(grouped["feature_group"], grouped["mean_fusion_mdd"] - grouped["mean_point_mdd"])
    axes[1].axvline(0.0, color="black", linestyle="--")
    axes[1].set_xlabel("fusion MDD - point-only MDD")
    axes[1].set_title("Risk gate MDD improvement")

    axes[2].scatter(
        grouped["mean_signal_share"],
        grouped["mean_fusion_return"],
        s=80,
        alpha=0.75,
    )
    for _, row in grouped.iterrows():
        axes[2].annotate(row["feature_group"], (row["mean_signal_share"], row["mean_fusion_return"]), fontsize=8)
    axes[2].set_xlabel("fusion active share")
    axes[2].set_ylabel("fusion cumulative return proxy")
    axes[2].set_title("Opportunity versus return proxy")
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.suptitle("12번 feature group decision summary")
    display.show_figure(fig)


def build_inline_report(args: argparse.Namespace, summary: pd.DataFrame, feature_groups: dict[str, list[str]]) -> str:
    lines = [
        "# 12번 feature guardrail fusion 실행 요약",
        "",
        "## 목적",
        "",
        "10번의 `balanced_composite` 점예측을 기본 토대로 쓰되, 11번의 `absolute_move` 위험 확률을 guardrail로 붙인다.",
        "이번 단계의 핵심은 모델 구조 추가가 아니라 코인 독립변수 조합을 바꾸었을 때 collapse와 MDD가 개선되는지 확인하는 것이다.",
        "",
        "## 왜 이 방향인가",
        "",
        "- 10번: 점예측은 아직 persistence를 이기지 못했지만, Huber보다 평탄화가 줄어든 안정 objective를 찾았다.",
        "- 11번: absolute-move 위험 확률은 AP/Brier 기준선보다 나아, 진입 제한이나 포지션 축소 guardrail로 쓸 수 있다.",
        "- 따라서 12번은 `feature group -> point branch -> risk branch -> decision policy` 순서로 어느 독립변수군을 데이터마트에 올릴지 고른다.",
        "",
        "## 실행 설정",
        "",
        "```json",
        json.dumps(vars(args), ensure_ascii=False, indent=2),
        "```",
        "",
        "## 사용 가능한 feature group",
        "",
        "```text",
        features_module.feature_group_table(feature_groups).to_string(index=False),
        "```",
        "",
    ]
    if not summary.empty:
        # fix③: 보고서 상위 후보도 활동 하한을 통과한 케이스에서만 뽑는다(미거래 1등 방지).
        eligible = summary[
            (summary["point_plus_risk_gate_active_share"] >= 0.05)
            & (summary["point_plus_risk_gate_trade_count"] >= 5)
        ]
        rank_source = eligible if not eligible.empty else summary
        best = rank_source.sort_values(
            ["point_plus_risk_gate_mdd", "point_copy_risk_ratio"],
            ascending=[False, True],
        ).head(15)
        lines.extend(
            [
                "## 상위 후보",
                "",
                "```text",
                best.to_string(index=False),
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "## 해석 기준",
            "",
            "- `point_copy_risk_ratio < 1`이면 persistence보다 나은 점예측이다. 1보다 크더라도 10번 대비 낮아지면 독립변수 추가 효과가 있다.",
            "- `point_plus_risk_gate_mdd`가 `point_only_mdd`보다 덜 음수이면 위험 gate가 하방 방어에 도움을 준 것이다.",
            "- `fusion_signal_share`가 거의 0이면 MDD가 좋아도 거래를 안 해서 좋아진 착시일 수 있다.",
            "- 좋은 feature group은 copy-risk를 낮추거나, 같은 copy-risk에서도 risk gate 결합 후 MDD를 줄여야 한다.",
            "- 이 조건을 통과한 feature group만 이후 데이터마트 정식 스키마 후보로 승격한다.",
            "",
        ]
    )
    return "\n".join(lines)
