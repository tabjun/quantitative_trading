# [FOR COMMIT TRACKING ONLY - DO NOT EXECUTE]
# This file is automatically mirrored from the corresponding .ipynb for git diff purposes.
# Actual research execution should be performed in the Jupyter Notebook (.ipynb)
# or in an approved remote/server environment.

# %% [markdown]
# # 14번: 정렬·활동하한 교정 fusion 재실행
#
# engine/ 공용 패키지(12 결함 ①②③ 교정본)를 import해 fusion 비교를 다시 돌린다. 원본 8~13은 read-only. 아래 한 셀이 `.py` 미러와 동일하다. 실행은 셀 하단 `if __name__` 가드 대신 `main([...])`을 직접 호출하거나 `.py`를 커널/터미널에서 돌린다.

# %%
"""14번: 정렬·활동하한이 교정된 fusion 재실행 드라이버.

12번 fusion의 결함을 `engine/` 공용 패키지에서 고친 뒤, 그 engine을 import해 fusion 비교를
다시 돌린다. 원본 8~13 노트북은 건드리지 않는다(read-only). 교정 포인트:
- ① point/risk를 decision_timestamp(진입 봉)로 inner-join 정렬 (기존 꼬리절단 폐기)
- ② 텍스트 컬럼 없으면 coin_text_context를 텍스트 그룹으로 둔갑시키지 않음
- ③ 활동 하한(active_share/trade_count)을 통과한 케이스만 우승 후보로 랭킹

이 재실행의 목적: 정렬·활동하한이 적용된 새 리더보드를 확보해, 어떤 변수셋이 진짜 1순위인지
다시 보고 13번 방향을 결정한다.

import 계약: quantitative_trading을 sys.path 루트로 두고 top-level `import engine`
(repo의 contexts/marts/analysis와 동일 관례).

실행 예시(서버 커널/터미널):
    uv run test/models/14_fusion_alignment_rerun_test.py \
        --feature-groups ohlcv_core,coin_multitimeframe_structure \
        --point-models Linear --risk-models Linear --preprocessings seasonal_diff16 \
        --seeds 42 --epochs 6 --max-rows 12000 --max-windows 2048 --batch-size 256 --no-case-plots
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless 실행: plt.show()는 no-op, 리더보드는 stdout으로 본다.

import pandas as pd
import torch


def _engine_root(start: Path) -> Path:
    candidates = [start, *start.parents, Path.home() / "personal_ai_project" / "quantitative_trading"]
    for candidate in candidates:
        if (candidate / "pyproject.toml").exists() and (candidate / "engine").is_dir():
            return candidate
    raise RuntimeError("engine을 담은 quantitative_trading 디렉터리를 찾지 못했다. 직접 sys.path에 넣어라.")


try:
    _START = Path(__file__).resolve().parent
except NameError:  # 노트북 셀 실행 경로
    _START = Path.cwd()
ROOT = _engine_root(_START)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine import data, display, features as feat, fusion, preprocessing, resources

DEFAULT_FEATURE_GROUPS = "ohlcv_core,coin_multitimeframe_structure,coin_volatility_regime"
DEFAULT_POINT_MODELS = "Linear"
DEFAULT_RISK_MODELS = "Linear"
DEFAULT_PREPROCESSINGS = "seasonal_diff16"
DEFAULT_SEEDS = "42"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="14번 교정 fusion 재실행 (engine 기반)")
    parser.add_argument("--db", default=None)
    parser.add_argument("--table", default="btc_15m_advance")
    parser.add_argument("--ticker", default=None)
    parser.add_argument("--profile", default="school_4090_15gb")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--feature-groups", default=DEFAULT_FEATURE_GROUPS)
    parser.add_argument("--point-models", default=DEFAULT_POINT_MODELS)
    parser.add_argument("--risk-models", default=DEFAULT_RISK_MODELS)
    parser.add_argument("--preprocessings", default=DEFAULT_PREPROCESSINGS)
    parser.add_argument("--seeds", default=DEFAULT_SEEDS)
    parser.add_argument("--cross-tickers", default="KRW-ETH,KRW-XRP,KRW-SOL")
    parser.add_argument("--point-objective", default="balanced_composite")
    parser.add_argument("--risk-event-kind", choices=["absolute_move", "downside"], default="absolute_move")
    parser.add_argument("--risk-horizon", type=int, default=16)
    parser.add_argument("--risk-allow-quantile", type=float, default=0.55)
    parser.add_argument("--event-quantile", type=float, default=0.70)
    parser.add_argument("--normalization", default="window_standard")
    parser.add_argument("--optimizer", default="adamw")
    parser.add_argument("--scheduler", default="cosine")
    parser.add_argument("--gradient-policy", default="clip1")
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--objective-warmup-epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-rows", type=int, default=12000)
    parser.add_argument("--max-windows", type=int, default=2048)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--cost-bps", type=float, default=14.0)
    parser.add_argument("--min-signal-bps", type=float, default=3.0)
    parser.add_argument("--min-active-share", type=float, default=0.05)
    parser.add_argument("--min-trade-count", type=int, default=5)
    parser.add_argument("--case-plots", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-failure", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_known_args(argv)[0]


def main(argv: list[str] | None = None) -> None:
    preprocessing.configure_inline_matplotlib()
    args = parse_args(argv)
    profile = resources.build_resource_profile(args.profile, args.device)
    resources.apply_resource_profile(profile)
    print("[environment]", json.dumps(resources.log_environment(profile), ensure_ascii=False, indent=2))

    features, feature_groups, raw = feat.load_feature_frame(args)
    cases = fusion.build_cases(args, feature_groups)
    display.display_table("feature groups", feat.feature_group_table(feature_groups), max_rows=100)
    print(f"[plan] cases={len(cases)}")
    print(pd.DataFrame(cases).head(80).to_string(index=False))
    print("[statistics]", json.dumps(data.basic_statistics(features), ensure_ascii=False, indent=2))

    if args.dry_run:
        print("[dry-run] 데이터 로드와 feature group 등록까지만 확인하고 종료한다.")
        return

    rows: list[dict[str, object]] = []
    for index, case in enumerate(cases, start=1):
        print(f"\n[case {index}/{len(cases)}] {case}")
        try:
            point_result, val_pred, test_pred, point_splits = fusion.run_point_branch(case, features, profile, args)
            risk_result, val_prob, test_prob, risk_splits, gate_cutoff = fusion.run_risk_branch(
                case, features, profile, args
            )
            fusion_result, policy_frame = fusion.evaluate_fusion_policy(
                case,
                point_splits["test"],
                test_pred,
                test_prob,
                gate_cutoff,
                args,
                risk_split=risk_splits["test"],
            )
            fusion_result.update(
                {
                    "risk_average_precision": risk_result["average_precision"],
                    "risk_average_precision_lift": risk_result["average_precision_lift"],
                    "risk_brier_skill_score": risk_result["brier_skill_score"],
                    "risk_expected_calibration_error": risk_result["expected_calibration_error"],
                    "risk_event_rate": risk_result["event_rate"],
                    "risk_train_event_rate": risk_result["train_event_rate"],
                }
            )
            rows.append(fusion_result)
            if args.case_plots:
                fusion.show_fusion_diagnostics(
                    case,
                    point_splits["test"],
                    test_pred,
                    test_prob,
                    gate_cutoff,
                    policy_frame,
                    args,
                    risk_split=risk_splits["test"],
                )
        except Exception as exc:  # noqa: BLE001 - 케이스 단위 격리
            if not args.continue_on_failure:
                raise
            print(f"[case failed] {case}: {exc}")
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    summary = pd.DataFrame(rows)
    fusion.show_feature_summary(summary, min_active_share=args.min_active_share, min_trade_count=args.min_trade_count)
    display.display_markdown(fusion.build_inline_report(args, summary, feature_groups))
    print(f"[done] corrected fusion re-run complete. scored_cases={len(summary)}")


if __name__ == "__main__":
    main()
