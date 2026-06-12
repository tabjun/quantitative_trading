# [FOR COMMIT TRACKING ONLY - DO NOT EXECUTE]
# This file is automatically mirrored from the corresponding .ipynb for git diff purposes.
# Actual research execution should be performed in the Jupyter Notebook (.ipynb)
# or in an approved remote/server environment.

# %% [markdown]
# # 6. 최적화 안정화 후속 실험
#
# [연구용 스크립트 - Codex 로컬 세션에서 자동 실행 금지]
# 5번 실험은 대표 알고리즘과 목적함수를 작게 비교해 shortcut collapse를 진단하는 단계였다.
# 6번 실험은 5번에서 드러난 문제를 한 번에 하나씩 수정해, 어떤 처방이 실제로 효과가 있는지 검증하는 단계다.
#
# 실행 원칙:
# - 기본 실행은 stage 계획만 출력한다.
# - 실제 학습은 학교 서버 또는 승인된 원격 환경에서만 수행한다.
# - 한 번에 하나의 요소만 수정한다.
# - 각 stage 후 보고서에는 "좋은 그림 기준"과 "이번 그림 해석"을 함께 남긴다.

# %%
from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


def bootstrap_repo_root() -> Path:
    candidates = [Path.cwd(), *Path.cwd().parents]
    for candidate in candidates:
        if (candidate / "database" / "paths.py").exists():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            return candidate
    raise ModuleNotFoundError("Run this notebook from inside the quantitative_trading repository.")


REPO_ROOT = bootstrap_repo_root()


@dataclass(frozen=True)
class StabilizationStage:
    stage: int
    name: str
    purpose: str
    command: list[str]
    good_plot: str
    decision_rule: str


STAGES = [
    StabilizationStage(
        stage=0,
        name="reproduce_5_quick_probe",
        purpose="5번 현재 결과가 승인된 실행 환경에서도 재현되는지 확인한다.",
        command=["uv", "run", "test/models/5_optimization_diagnostics_test.py", "--suite", "quick_probe", "--max-windows", "1024", "--epochs", "30"],
        good_plot="15개 케이스의 순위와 persistence/collapse 방향이 현재 보고서와 크게 다르지 않아야 한다.",
        decision_rule="재현이 안 되면 stage 1로 가지 않고 seed, split, scaler, 데이터 기간부터 확인한다.",
    ),
    StabilizationStage(
        stage=1,
        name="return_target_gate",
        purpose="가격 레벨 직접 회귀를 기본 후보에서 제외하고 return target 중심으로 재확인한다.",
        command=["uv", "run", "test/models/5_optimization_diagnostics_test.py", "--suite", "objective_probe", "--max-windows", "1024", "--epochs", "40"],
        good_plot="return 계열에서 검증선이 같이 내려가고 persistence gap이 기준선 쪽으로 내려와야 한다.",
        decision_rule="level target보다 return target이 확실히 낫지 않으면 target 정의부터 다시 수정한다.",
    ),
    StabilizationStage(
        stage=2,
        name="normalization_ablation",
        purpose="rolling/robust/RevIN-like normalization 후보를 하나씩 비교한다.",
        command=["uv", "run", "test/models/5_optimization_diagnostics_test.py", "--suite", "architecture_probe", "--feature-set", "optimization_probe", "--max-windows", "1024", "--epochs", "40"],
        good_plot="normalization 변경 후 gradient가 덜 튀고 validation gap이 줄어야 한다.",
        decision_rule="정규화만 바꿨는데 persistence gap이 악화되면 해당 scaling은 제외한다.",
    ),
    StabilizationStage(
        stage=3,
        name="loss_ablation",
        purpose="directional hybrid를 기준으로 volatility-weighted, asymmetric, baseline-margin loss를 순차 비교한다.",
        command=["uv", "run", "test/models/5_optimization_diagnostics_test.py", "--suite", "full_matrix", "--feature-set", "optimization_probe", "--max-windows", "1536", "--epochs", "50"],
        good_plot="loss 변경 후 sign agreement가 올라가면서 collapse score가 함께 악화되지 않아야 한다.",
        decision_rule="방향성만 오르고 persistence가 망가지면 실전 후보에서 제외한다.",
    ),
    StabilizationStage(
        stage=4,
        name="collapse_aware_selection",
        purpose="validation loss 최저가 아니라 persistence와 collapse를 함께 통과한 epoch를 선택한다.",
        command=["uv", "run", "test/models/5_optimization_diagnostics_test.py", "--suite", "full_matrix", "--feature-set", "optimization_probe", "--max-windows", "2048", "--epochs", "60", "--save-artifacts", "--save-csv"],
        good_plot="best epoch 주변에서 persistence gap이 기준선 아래에 머물고 zero-share가 과도하게 높지 않아야 한다.",
        decision_rule="loss 최저 epoch와 진단 최저 epoch가 다르면 collapse-aware selection을 채택한다.",
    ),
]

# %%
def print_plan() -> None:
    print("# 6. optimization stabilization stage plan")
    for item in STAGES:
        print(f"\n## Stage {item.stage}: {item.name}")
        print(f"- purpose: {item.purpose}")
        print(f"- command: {' '.join(item.command)}")
        print(f"- good plot: {item.good_plot}")
        print(f"- decision rule: {item.decision_rule}")


def run_stage(stage: int) -> int:
    matches = [item for item in STAGES if item.stage == stage]
    if not matches:
        raise ValueError(f"Unknown stage: {stage}")
    item = matches[0]
    print(f"[stage {item.stage}] {item.name}")
    print(" ".join(item.command))
    completed = subprocess.run(item.command, cwd=REPO_ROOT)
    return int(completed.returncode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan or run staged optimization stabilization checks.")
    parser.add_argument("--run-stage", type=int, default=None, help="Run one stage in an approved remote/server environment.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.run_stage is None:
        print_plan()
        return 0
    return run_stage(args.run_stage)


if __name__ == "__main__":
    raise SystemExit(main())
