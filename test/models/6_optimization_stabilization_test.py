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
# - GPU 학습 stage는 병렬 실행하지 않고 순차 실행한다. 병렬 처리는 CSV/이미지 후처리에서만 고려한다.
# - CUDA OOM이 감지되면 같은 stage를 더 작은 batch size로 재시도한다.
# - 각 stage 후 보고서에는 "좋은 그림 기준"과 "이번 그림 해석"을 함께 남긴다.

# %%
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


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
DIAGNOSTIC_SCRIPT = "test/models/5_optimization_diagnostics_test.py"

OOM_PATTERNS = (
    "out of memory",
    "cuda error: out of memory",
    "cublas_status_alloc_failed",
    "cudnn_status_alloc_failed",
    "memoryerror",
)


@dataclass(frozen=True)
class ResourceProfile:
    name: str
    max_windows: int
    epochs: int
    batch_size: int
    min_batch_size: int
    num_workers: int
    explanation: str


RESOURCE_PROFILES = {
    "server_1024": ResourceProfile(
        name="server_1024",
        max_windows=1024,
        epochs=40,
        batch_size=64,
        min_batch_size=16,
        num_workers=0,
        explanation="빠른 원인 분해용. 5번 quick_probe와 비슷한 창 수로 먼저 안정화 축만 확인한다.",
    ),
    "server_2048": ResourceProfile(
        name="server_2048",
        max_windows=2048,
        epochs=60,
        batch_size=64,
        min_batch_size=16,
        num_workers=0,
        explanation="교수님 보고 전 본 실행 후보. 3번 실험에서 2048 수준이 가능했던 기억을 반영하되 batch는 보수적으로 둔다.",
    ),
    "server_light": ResourceProfile(
        name="server_light",
        max_windows=768,
        epochs=25,
        batch_size=32,
        min_batch_size=8,
        num_workers=0,
        explanation="커널 상태가 불안정할 때 쓰는 복구용. 전체 결론용이 아니라 코드/출력 구조 확인용이다.",
    ),
}


@dataclass(frozen=True)
class StageCommand:
    label: str
    argv: list[str]


@dataclass(frozen=True)
class StabilizationStage:
    stage: int
    name: str
    purpose: str
    why_now: str
    changed_knob: str
    commands: list[StageCommand]
    expected_artifacts: str
    good_plot: str
    failure_signal: str
    decision_rule: str


def build_base_command(profile: ResourceProfile, *, suite: str, normalization: str, epochs: int) -> list[str]:
    return [
        "uv",
        "run",
        DIAGNOSTIC_SCRIPT,
        "--suite",
        suite,
        "--feature-set",
        "optimization_probe",
        "--normalization",
        normalization,
        "--max-windows",
        str(profile.max_windows),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(profile.batch_size),
        "--num-workers",
        str(profile.num_workers),
        "--save-csv",
    ]


def build_stages(profile: ResourceProfile) -> list[StabilizationStage]:
    quick_epochs = min(30, profile.epochs)
    mid_epochs = min(50, profile.epochs)

    return [
        StabilizationStage(
            stage=0,
            name="reproduce_5_quick_probe",
            purpose="5번 현재 결과가 승인된 실행 환경에서도 재현되는지 확인한다.",
            why_now="재현이 흔들리면 이후 정규화/손실함수 비교가 전부 해석 불가능해진다.",
            changed_knob="수정 없음. 5번 quick_probe를 같은 기준선으로 다시 실행한다.",
            commands=[
                StageCommand(
                    label="quick_probe_standard",
                    argv=build_base_command(profile, suite="quick_probe", normalization="standard", epochs=quick_epochs),
                )
            ],
            expected_artifacts="quick_probe summary CSV, curve CSV, notebook/report 출력 비교용 figure",
            good_plot="15개 케이스의 train/validation 방향, persistence gap, collapse score 순위가 5번 보고서와 크게 다르지 않아야 한다.",
            failure_signal="seed, split, scaler, 로컬 노트북 출력과 서버 출력이 다르게 움직인다.",
            decision_rule="재현이 안 되면 stage 1로 가지 않고 데이터 기간, split, scaler, seed부터 고정한다.",
        ),
        StabilizationStage(
            stage=1,
            name="target_objective_gate",
            purpose="가격 레벨 직접 회귀를 기본 후보에서 제외할 근거를 return target 중심으로 다시 확인한다.",
            why_now="level target이 복사형 쉬운 해를 만들면 독립변수를 추가해도 실제 개선인지 구분할 수 없다.",
            changed_knob="target/objective만 확장한다. 모델군과 feature set은 그대로 둔다.",
            commands=[
                StageCommand(
                    label="objective_probe_standard",
                    argv=build_base_command(
                        profile,
                        suite="objective_probe",
                        normalization="standard",
                        epochs=profile.epochs,
                    ),
                )
            ],
            expected_artifacts="level_mse, return_mse, return_huber, return_directional_hybrid 비교 CSV",
            good_plot="return 계열에서 validation loss가 같이 내려가고 persistence gap이 0 아래로 내려가는 후보가 생겨야 한다.",
            failure_signal="return target으로 바꿔도 persistence gap이 계속 양수이고 variance ratio가 0 근처로 붙는다.",
            decision_rule="return target도 실패하면 target을 volatility-scaled return, direction/neutral-band label로 다시 정의한다.",
        ),
        StabilizationStage(
            stage=2,
            name="normalization_ablation",
            purpose="비정상성 완화가 실제로 gradient와 validation gap을 안정화하는지 확인한다.",
            why_now="비정상 시계열은 평균/분산이 구간마다 바뀌므로, 모델이 먼저 scale 변화에 끌려 쉬운 해를 찾을 수 있다.",
            changed_knob="normalization만 `robust`, `window_standard`, `identity`로 바꾼다.",
            commands=[
                StageCommand(
                    label="architecture_probe_robust",
                    argv=build_base_command(profile, suite="architecture_probe", normalization="robust", epochs=profile.epochs),
                ),
                StageCommand(
                    label="architecture_probe_window_standard",
                    argv=build_base_command(
                        profile,
                        suite="architecture_probe",
                        normalization="window_standard",
                        epochs=profile.epochs,
                    ),
                ),
                StageCommand(
                    label="architecture_probe_identity_control",
                    argv=build_base_command(profile, suite="architecture_probe", normalization="identity", epochs=quick_epochs),
                ),
            ],
            expected_artifacts="정규화 방식별 architecture_probe CSV 3묶음",
            good_plot="좋은 정규화는 gradient norm 튐을 줄이고 validation-train gap과 persistence gap을 동시에 낮춘다.",
            failure_signal="정규화만 바꿨는데 sign agreement가 무작위 수준이고 zero_share가 더 높아진다.",
            decision_rule="robust/window_standard 중 persistence gap과 collapse score가 함께 개선되는 쪽만 다음 loss ablation에 남긴다.",
        ),
        StabilizationStage(
            stage=3,
            name="loss_ablation",
            purpose="Huber, directional hybrid, volatility-weighted, tail-focus loss 중 붕괴를 덜 만드는 후보를 고른다.",
            why_now="손실함수가 바뀌면 모델이 찾는 쉬운 답도 바뀐다. loss 최저가 아니라 baseline을 이기는 방향으로 골라야 한다.",
            changed_knob="loss/objective만 바꾼다. 대표 알고리즘 5개는 유지한다.",
            commands=[
                StageCommand(
                    label="loss_probe_standard",
                    argv=build_base_command(
                        profile,
                        suite="stabilization_loss_probe",
                        normalization="standard",
                        epochs=mid_epochs,
                    ),
                ),
                StageCommand(
                    label="loss_probe_window_standard",
                    argv=build_base_command(
                        profile,
                        suite="stabilization_loss_probe",
                        normalization="window_standard",
                        epochs=mid_epochs,
                    ),
                ),
            ],
            expected_artifacts="5개 대표군 x 4개 return 손실함수 x 정규화 2종 비교 CSV",
            good_plot="sign agreement가 올라가면서 persistence gap이 0 아래로 내려가고, variance ratio가 0에 붙지 않아야 한다.",
            failure_signal="방향성 지표만 좋아지고 KRW MAE 또는 persistence gap이 악화된다.",
            decision_rule="방향성, baseline 돌파, collapse score를 함께 만족한 조합만 최종 stage로 보낸다.",
        ),
        StabilizationStage(
            stage=4,
            name="resource_scale_check",
            purpose="창 수를 2048 수준까지 키웠을 때도 같은 결론이 유지되는지 확인한다.",
            why_now="작은 창에서만 좋아 보이면 독립변수/데이터마트 확장 전에 다시 무너질 수 있다.",
            changed_knob="max_windows와 epochs를 본 실행 후보 수준으로 키운다. batch는 OOM retry로 자동 축소한다.",
            commands=[
                StageCommand(
                    label="full_matrix_window_standard_scale",
                    argv=[
                        *build_base_command(
                            profile,
                            suite="full_matrix",
                            normalization="window_standard",
                            epochs=profile.epochs,
                        ),
                        "--save-artifacts",
                    ],
                )
            ],
            expected_artifacts="full_matrix report markdown, summary CSV, curve CSV, case별 PNG",
            good_plot="창 수를 늘려도 best 후보의 persistence gap 방향과 collapse 지표가 stage 3과 같은 방향이어야 한다.",
            failure_signal="창 수를 늘리자 validation loss는 내려가는데 persistence gap이 다시 양수로 돌아온다.",
            decision_rule="확장 시 무너지면 모델 복잡도/seq_len/feature set을 줄인 뒤 stage 2부터 반복한다.",
        ),
        StabilizationStage(
            stage=5,
            name="independent_variable_gate",
            purpose="텍스트 독립변수, historical flow mart, 온체인/유동성 변수 연결 전 최소 안정성 기준을 확정한다.",
            why_now="최적화 문제가 남아 있으면 새 변수를 붙였을 때 성능 변화의 원인을 해석할 수 없다.",
            changed_knob="feature set만 `market_only`, `text_aware`로 바꿀 준비를 한다. 기본 명령은 안전상 출력만 남긴다.",
            commands=[],
            expected_artifacts="다음 7번 또는 독립변수 실험으로 넘길 통과 기준 문서",
            good_plot="최소 한 조합이 baseline을 지속적으로 이기고, flat/zero-return 붕괴 신호가 완화되어야 한다.",
            failure_signal="어떤 조합도 baseline을 못 이기거나, 작은 조정마다 결론이 계속 뒤집힌다.",
            decision_rule="통과 전에는 독립변수 확장보다 target/loss/normalization 재설계를 우선한다.",
        ),
    ]


def get_arg_value(argv: list[str], flag: str) -> str | None:
    try:
        idx = argv.index(flag)
    except ValueError:
        return None
    if idx + 1 >= len(argv):
        return None
    return argv[idx + 1]


def replace_arg_value(argv: list[str], flag: str, value: str) -> list[str]:
    patched = list(argv)
    try:
        idx = patched.index(flag)
    except ValueError:
        return [*patched, flag, value]
    if idx + 1 >= len(patched):
        return [*patched, value]
    patched[idx + 1] = value
    return patched


def is_oom_failure(output: str) -> bool:
    lowered = output.lower()
    return any(pattern in lowered for pattern in OOM_PATTERNS)


def print_completed_output(completed: subprocess.CompletedProcess[str]) -> None:
    if completed.stdout:
        print(completed.stdout)
    if completed.stderr:
        print(completed.stderr, file=sys.stderr)


def run_with_oom_retry(argv: list[str], min_batch_size: int, dry_run: bool = False) -> int:
    current = int(get_arg_value(argv, "--batch-size") or "64")
    command = list(argv)

    while True:
        print(f"[run] {' '.join(command)}")
        if dry_run:
            return 0

        env = os.environ.copy()
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
        )
        print_completed_output(completed)
        if completed.returncode == 0:
            return 0

        combined_output = f"{completed.stdout}\n{completed.stderr}"
        if not is_oom_failure(combined_output):
            return int(completed.returncode)

        if current <= min_batch_size:
            print(
                f"[oom] batch_size={current} already reached min_batch_size={min_batch_size}. "
                "Stop this stage and inspect model/window size."
            )
            return int(completed.returncode)

        next_batch = max(min_batch_size, current // 2)
        print(f"[oom] CUDA/memory failure detected. Retrying with batch_size={next_batch}.")
        current = next_batch
        command = replace_arg_value(command, "--batch-size", str(current))


def print_plan(profile: ResourceProfile, stages: Iterable[StabilizationStage]) -> None:
    print("# 6. optimization stabilization stage plan")
    print(f"\nprofile: {profile.name}")
    print(f"- max_windows: {profile.max_windows}")
    print(f"- epochs: {profile.epochs}")
    print(f"- batch_size: {profile.batch_size}")
    print(f"- min_batch_size: {profile.min_batch_size}")
    print(f"- num_workers: {profile.num_workers}")
    print(f"- reason: {profile.explanation}")

    for item in stages:
        print(f"\n## Stage {item.stage}: {item.name}")
        print(f"- purpose: {item.purpose}")
        print(f"- why now: {item.why_now}")
        print(f"- changed knob: {item.changed_knob}")
        if item.commands:
            for command in item.commands:
                print(f"- command [{command.label}]: {' '.join(command.argv)}")
        else:
            print("- command: no training command; this is a documentation/decision gate.")
        print(f"- expected artifacts: {item.expected_artifacts}")
        print(f"- good plot: {item.good_plot}")
        print(f"- failure signal: {item.failure_signal}")
        print(f"- decision rule: {item.decision_rule}")


def run_stage(
    stage_number: int,
    stages: list[StabilizationStage],
    profile: ResourceProfile,
    dry_run: bool,
    continue_on_failure: bool,
) -> int:
    matches = [item for item in stages if item.stage == stage_number]
    if not matches:
        raise ValueError(f"Unknown stage: {stage_number}")

    item = matches[0]
    print(f"[stage {item.stage}] {item.name}")
    if not item.commands:
        print("[stage] No executable command. Use this gate for report/update decisions.")
        return 0

    for command in item.commands:
        print(f"[stage {item.stage}] command={command.label}")
        code = run_with_oom_retry(command.argv, min_batch_size=profile.min_batch_size, dry_run=dry_run)
        if code != 0 and not continue_on_failure:
            return code
    return 0


def run_all(
    stages: list[StabilizationStage],
    profile: ResourceProfile,
    dry_run: bool,
    continue_on_failure: bool,
) -> int:
    for item in stages:
        code = run_stage(
            stage_number=item.stage,
            stages=stages,
            profile=profile,
            dry_run=dry_run,
            continue_on_failure=continue_on_failure,
        )
        if code != 0 and not continue_on_failure:
            return code
    return 0


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan or run staged optimization stabilization checks.")
    parser.add_argument("--profile", choices=sorted(RESOURCE_PROFILES.keys()), default="server_2048")
    parser.add_argument("--run-stage", type=int, default=None, help="Run one stage in an approved remote/server environment.")
    parser.add_argument("--run-all", action="store_true", help="Run every executable stage sequentially in an approved environment.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    parser.add_argument("--continue-on-failure", action="store_true", help="Continue later stages even if a stage fails.")
    if argv is None:
        if "ipykernel" in sys.modules:
            args, _unknown = parser.parse_known_args()
            return args
        return parser.parse_args()
    return parser.parse_args(list(argv))


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    profile = RESOURCE_PROFILES[args.profile]
    stages = build_stages(profile)

    if args.run_stage is None and not args.run_all:
        print_plan(profile, stages)
        return 0
    if args.run_all:
        return run_all(stages, profile, dry_run=args.dry_run, continue_on_failure=args.continue_on_failure)
    return run_stage(
        stage_number=args.run_stage,
        stages=stages,
        profile=profile,
        dry_run=args.dry_run,
        continue_on_failure=args.continue_on_failure,
    )


if __name__ == "__main__":
    if "ipykernel" in sys.modules:
        main()
    else:
        raise SystemExit(main())
