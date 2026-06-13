# [FOR COMMIT TRACKING ONLY - DO NOT EXECUTE]
# This file is automatically mirrored from the corresponding .ipynb for git diff purposes.
# Actual research execution should be performed in the Jupyter Notebook (.ipynb)
# or in an approved remote/server environment.

# %% [markdown]
# # 7. 성능 폭 확장 후속 실험
#
# [연구용 스크립트 - Codex 로컬 세션에서 자동 실행 금지]
# 7번 실험은 5번 quick probe와 6번 안정화 오케스트레이터를 보존한 채,
# 더 넓은 모델군과 앙상블군을 한 번에 비교하기 위한 후속 오케스트레이터다.
#
# 실행 원칙:
# - 기본 실행은 breadth/ensemble 실험 계획만 출력한다.
# - 실제 학습은 학교 서버 또는 승인된 원격 환경에서만 수행한다.
# - 7번은 6번을 덮어쓰지 않고, 6번 결과를 반영한 다음 번호 실험으로 남긴다.
# - 각 suite 결과 보고서는 독립 문서로 작성하고 방법론 설명을 다시 적는다.

# %%
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ResourceProfile:
    name: str
    max_windows: int
    epochs: int
    batch_size: int
    num_workers: int
    explanation: str


RESOURCE_PROFILES = {
    "server_1024": ResourceProfile(
        name="server_1024",
        max_windows=1024,
        epochs=35,
        batch_size=48,
        num_workers=0,
        explanation="\ube60\ub978 breadth_probe 1\ucc28 \ud655\uc778\uc6a9 \ud504\ub85c\ud544.",
    ),
    "server_2048": ResourceProfile(
        name="server_2048",
        max_windows=2048,
        epochs=60,
        batch_size=48,
        num_workers=0,
        explanation="\uad50\uc218\ub2d8 \ubcf4\uace0 \uc804 \ubcf8 \uc2e4\ud589 \ud6c4\ubcf4.",
    ),
    "server_light": ResourceProfile(
        name="server_light",
        max_windows=768,
        epochs=25,
        batch_size=32,
        num_workers=0,
        explanation="\ucee4\ub110 \uc0c1\ud0dc \ud655\uc778\uacfc \ubcf4\uace0\uc11c \uad6c\uc870 \uc810\uac80\uc6a9 \uacbd\ub7c9 \ud504\ub85c\ud544.",
    ),
}


@dataclass(frozen=True)
class ModelFamily:
    name: str
    category: str
    purpose: str
    why_now: str


MODEL_FAMILIES = [
    ModelFamily("Linear", "baseline", "\uac00\uc7a5 \ub2e8\uc21c\ud55c \ud1b5\uc81c\uad70", "\ubcf5\uc7a1\ud55c \ubaa8\ub378\uc774 baseline\ubcf4\ub2e4 \uc2e4\uc81c\ub85c \ub098\uc740\uc9c0 \ud655\uc778\ud55c\ub2e4."),
    ModelFamily("LSTM", "recurrent", "\uae30\uc5b5 \uae30\ubc18 recurrent \ube44\uad50\uad70", "5\ubc88 \uae30\uc900\uc120\uc744 \uc720\uc9c0\ud558\uba74\uc11c \ub354 \ub113\uc740 breadth\uc5d0 \ud3ec\ud568\ud55c\ub2e4."),
    ModelFamily("GRU", "recurrent", "\uacbd\ub7c9 recurrent \ube44\uad50\uad70", "LSTM\ubcf4\ub2e4 \ube60\ub974\uac8c \uc548\uc815\ud654\ub418\ub294\uc9c0 \ud655\uc778\ud55c\ub2e4."),
    ModelFamily("TCN", "convolutional", "dilated convolution \ube44\uad50\uad70", "recurrent\uc640 \ub2e4\ub978 smoothing \uacbd\ud5a5\uc744 \ubcf8\ub2e4."),
    ModelFamily("Transformer", "attention", "attention \ube44\uad50\uad70", "\ubcf5\uc7a1\ud55c \uc0c1\ud638\uc791\uc6a9 \uad6c\uc870\uac00 \uc2e4\uc81c\ub85c \ub3c4\uc6c0\uc774 \ub418\ub294\uc9c0 \ubcf8\ub2e4."),
    ModelFamily("Autoformer-like", "decomposition", "trend/seasonal decomposition \uacc4\uc5f4 \ub300\ub9ac\uad70", "3\ubc88\uc758 Autoformer \ucc29\uc2dc\ub97c \ub354 \ub113\uc740 \uae30\uc900\uc5d0\uc11c \ub2e4\uc2dc \ud655\uc778\ud55c\ub2e4."),
    ModelFamily("PatchTST-like", "patch_attention", "patch token \uae30\ubc18 \ube44\uad50\uad70", "\uc785\ub825 patching\uc774 collapse\ub97c \uc904\uc774\ub294\uc9c0 \ubcf8\ub2e4."),
    ModelFamily("DLinear-like", "linear_decomposition", "decomposition linear \uacc4\uc5f4", "\ub2e8\uc21c \uad6c\uc870\uac00 \uc624\ud788\ub824 \uc548\uc815\uc801\uc778\uc9c0 \ud655\uc778\ud55c\ub2e4."),
    ModelFamily("NLinear-like", "linear_residual", "residual linear \uacc4\uc5f4", "\ud3c9\uade0 \ubcf5\uc0ac\ud615 \uc26c\uc6b4 \ud574\ub97c \uc5bc\ub9c8\ub098 \uc904\uc774\ub294\uc9c0 \ubcf8\ub2e4."),
    ModelFamily("TimesNet-like", "periodic_block", "\uc8fc\uae30 \ube14\ub85d \uae30\ubc18 \ube44\uad50\uad70", "\uc8fc\uae30 \uad6c\uc870\uac00 \ube44\uc815\uc0c1 \uc2dc\uacc4\uc5f4\uc5d0\uc11c \uacfc\uc5f0 \uc720\uc9c0\ub418\ub294\uc9c0 \ubcf8\ub2e4."),
    ModelFamily("TimeXer-like", "exogenous_attention", "\uc678\uc0dd\ubcc0\uc218 \uce5c\ud654 \uacc4\uc5f4", "\ub2e4\uc74c \ub3c5\ub9bd\ubcc0\uc218 \ud655\uc7a5 \uc804 \uae30\ubc18 \ud6c4\ubcf4\ub85c \uc720\uc9c0\ud560 \uac00\uce58\uac00 \uc788\ub294\uc9c0 \ubcf8\ub2e4."),
    ModelFamily("iTransformer-like", "inverted_attention", "\ubcc0\uc218 \ud1a0\ud070 \uad00\uc810 \ube44\uad50\uad70", "\ubcc0\uc218 \ucc28\uc6d0\uc744 \uc911\uc2ec\uc73c\ub85c \ubcfc \ub54c \uc548\uc815\ud654\uac00 \ub2ec\ub77c\uc9c0\ub294\uc9c0 \ubcf8\ub2e4."),
    ModelFamily("ModernTCN-like", "modern_convolutional", "\ucd5c\uadfc convolutional \uacc4\uc5f4 \ube44\uad50\uad70", "TCN \ud655\uc7a5\ud615\uc774 \ub354 \ub113\uc740 \ucc3d\uc5d0\uc11c \uc720\uc9c0\ub418\ub294\uc9c0 \ubcf8\ub2e4."),
    ModelFamily("Mamba-like", "state_space", "state space \uacc4\uc5f4 \ube44\uad50\uad70", "\ucd5c\uadfc sequence family\uac00 \uae08\uc735 \ube44\uc815\uc0c1 \uad6c\uac04\uc5d0\uc11c \uc5b4\ub5a4\uc9c0 \ubcf8\ub2e4."),
]


@dataclass(frozen=True)
class EnsembleFamily:
    name: str
    members: tuple[str, ...]
    method: str
    purpose: str


ENSEMBLE_FAMILIES = [
    EnsembleFamily("LSTM + Autoformer", ("LSTM", "Autoformer-like"), "soft_average", "recurrent \uae30\uc5b5\uacfc decomposition \uacc4\uc5f4\uc758 \ubcf4\uc644 \uc5ec\ubd80\ub97c \ubcf8\ub2e4."),
    EnsembleFamily("TCN + Transformer", ("TCN", "Transformer"), "soft_average", "convolutional smoothing\uacfc attention \uc0c1\ud638\uc791\uc6a9\uc758 \ubcf4\uc644 \uc5ec\ubd80\ub97c \ubcf8\ub2e4."),
    EnsembleFamily("Linear + sequence residual stack", ("Linear", "PatchTST-like"), "residual_stack", "\ub2e8\uc21c baseline\uacfc sequence model\uc744 \uc794\ucc28 \uad00\uc810\uc73c로 결합한다."),
    EnsembleFamily("validation weighted top-k ensemble", ("top_k_from_breadth_probe",), "validation_weighted", "\ub2e8\uc77c 후보의 우연성보다 검증 기준 가중 평균이 더 안정적인지 본다."),
]


@dataclass(frozen=True)
class BreadthSuite:
    name: str
    purpose: str
    changed_knob: str
    model_names: tuple[str, ...]
    ensemble_names: tuple[str, ...]
    expected_artifacts: str
    good_plot: str
    failure_signal: str
    decision_rule: str


SUITES = {
    "breadth_probe": BreadthSuite(
        name="breadth_probe",
        purpose="\ub113\uc740 \ub2e8\uc77c \ubaa8\ub378\uad70을 \uc57d\uc2dd으로 훑어 collapse 방향과 baseline 돌파 가능성을 본다.",
        changed_knob="\ubaa8\ub378 family 폭을 넓힌다. target/loss/normalization 기본값은 6번 gate를 반영한다.",
        model_names=tuple(model.name for model in MODEL_FAMILIES),
        ensemble_names=(),
        expected_artifacts="summary CSV, curve CSV, collapse figure, model family별 상세 figure",
        good_plot="확장 모델군 중 최소 몇 개는 persistence_gap이 0 아래로 내려가고 zero_share가 과도하지 않아야 한다.",
        failure_signal="모델군만 늘었는데 대부분이 baseline을 못 이기고 분산도 0에 가까워진다.",
        decision_rule="breadth_probe 상위권만 다음 normalization/loss/ensemble 단계로 보낸다.",
    ),
    "ensemble_probe": BreadthSuite(
        name="ensemble_probe",
        purpose="혼합 앙상블군이 단일 모델보다 더 안정적으로 baseline을 이기는지 본다.",
        changed_knob="모델 조합과 결합 방식만 바꾼다.",
        model_names=(),
        ensemble_names=tuple(item.name for item in ENSEMBLE_FAMILIES),
        expected_artifacts="ensemble summary CSV, 단일 모델 대비 비교표, ensemble 상세 figure",
        good_plot="앙상블이 sign_agreement와 persistence_gap을 함께 개선해야 한다.",
        failure_signal="앙상블이 loss만 좋아지고 KRW MAE 또는 persistence_gap은 오히려 악화된다.",
        decision_rule="앙상블은 단일 모델 대비 명확한 개선이 있을 때만 후속 후보로 남긴다.",
    ),
    "normalization_cross_check": BreadthSuite(
        name="normalization_cross_check",
        purpose="6번에서 남긴 normalization 후보가 넓은 모델군에서도 같은 방향으로 유효한지 본다.",
        changed_knob="normalization을 standard, robust, window_standard로만 바꾼다.",
        model_names=("LSTM", "TCN", "Transformer", "PatchTST-like", "ModernTCN-like"),
        ensemble_names=("LSTM + Autoformer", "TCN + Transformer"),
        expected_artifacts="normalization별 breadth summary CSV와 비교 그래프",
        good_plot="window_standard 또는 robust가 여러 family에서 공통으로 persistence_gap과 collapse_score를 개선한다.",
        failure_signal="정규화에 따라 결론이 전부 뒤집혀 일관된 후보가 사라진다.",
        decision_rule="일관된 개선을 보이는 normalization만 7번 후반 stage에 남긴다.",
    ),
    "loss_cross_check": BreadthSuite(
        name="loss_cross_check",
        purpose="6번에서 남긴 loss 후보가 넓은 모델군과 앙상블군에서도 붕괴를 줄이는지 본다.",
        changed_knob="return_huber, directional_hybrid, volatility_weighted, tail_focus만 바꾼다.",
        model_names=("LSTM", "TCN", "Transformer", "PatchTST-like", "Mamba-like"),
        ensemble_names=("validation weighted top-k ensemble",),
        expected_artifacts="loss별 비교 CSV, family별 상세 곡선, collapse figure",
        good_plot="방향성 지표와 baseline 돌파가 함께 좋아지는 loss 후보가 생겨야 한다.",
        failure_signal="방향성만 좋아지거나 zero_share만 낮아지고 실제 KRW 오차는 더 나빠진다.",
        decision_rule="loss는 baseline 돌파와 collapse 완화가 동시에 확인된 경우만 채택한다.",
    ),
    "scale_confirmation": BreadthSuite(
        name="scale_confirmation",
        purpose="넓은 breadth와 앙상블 후보를 더 큰 창 수와 epoch에서 다시 확인한다.",
        changed_knob="max_windows, epochs, batch retry 정책만 확장한다.",
        model_names=("top_candidates_from_breadth_probe",),
        ensemble_names=("top_candidates_from_ensemble_probe",),
        expected_artifacts="확장 실행 summary CSV, 최종 후보 보고서용 figure",
        good_plot="작은 창에서 좋았던 후보가 2048 수준에서도 같은 방향으로 유지되어야 한다.",
        failure_signal="창 수를 늘리자 baseline 대비 우위가 사라지고 collapse 신호가 다시 커진다.",
        decision_rule="확장 시 무너지면 8번 독립변수 확장보다 7번 내부 재설계를 우선한다.",
    ),
}


def print_profile(profile: ResourceProfile) -> None:
    print(f"profile: {profile.name}")
    print(f"- max_windows: {profile.max_windows}")
    print(f"- epochs: {profile.epochs}")
    print(f"- batch_size: {profile.batch_size}")
    print(f"- num_workers: {profile.num_workers}")
    print(f"- reason: {profile.explanation}")


def print_models(model_names: Iterable[str]) -> None:
    for name in model_names:
        item = next((model for model in MODEL_FAMILIES if model.name == name), None)
        if item is None:
            print(f"- {name}: future candidate placeholder")
            continue
        print(f"- {item.name} [{item.category}]: {item.purpose} / {item.why_now}")


def print_ensembles(ensemble_names: Iterable[str]) -> None:
    for name in ensemble_names:
        item = next((ensemble for ensemble in ENSEMBLE_FAMILIES if ensemble.name == name), None)
        if item is None:
            print(f"- {name}: future candidate placeholder")
            continue
        members = ", ".join(item.members)
        print(f"- {item.name} [{item.method}]: members=({members}) / {item.purpose}")


def print_suite(name: str, profile: ResourceProfile) -> None:
    suite = SUITES[name]
    print(f"## suite: {suite.name}")
    print(f"- purpose: {suite.purpose}")
    print(f"- changed knob: {suite.changed_knob}")
    print(f"- expected artifacts: {suite.expected_artifacts}")
    print(f"- good plot: {suite.good_plot}")
    print(f"- failure signal: {suite.failure_signal}")
    print(f"- decision rule: {suite.decision_rule}")
    print("")
    print("### model families")
    print_models(suite.model_names)
    print("")
    print("### ensemble families")
    if suite.ensemble_names:
        print_ensembles(suite.ensemble_names)
    else:
        print("- none")
    print("")
    print("### recommended remote command skeleton")
    print(
        "uv run test/models/7_optimization_breadth_expansion_test.py "
        f"--suite {suite.name} --profile {profile.name}"
    )


def print_plan(profile: ResourceProfile) -> None:
    print("# 7. optimization breadth expansion stage plan")
    print_profile(profile)
    print("")
    print("\uc774 \uc2e4\ud5d8\uc740 6\ubc88 gate\ub97c \uc720\uc9c0\ud55c \ucc44 \ub354 \ub113\uc740 \ubaa8\ub378\uad70\uacfc \uc559\uc0c1\ube14\uad70\uc744 \ud55c \ubc88\uc5d0 \ube44\uad50\ud558\uae30 \uc704\ud55c \ud6c4\uc18d \uc624\ucf00\uc2a4\ud2b8\ub808\uc774\ud130\ub2e4.")
    print("\uc2e4\uc81c \ud559\uc2b5 \uacb0\uacfc\ub294 \uc2b9\uc778\ub41c \uc11c\ubc84 \ud658\uacbd\uc5d0\uc11c \uac01 suite\ub97c \uc21c\ucc28 \uc2e4\ud589\ud574 \ubcc4\ub3c4 CSV/PNG/Markdown\uc73c\ub85c \uc800\uc7a5\ud55c\ub2e4.")
    print("")
    for name in SUITES:
        print_suite(name, profile)
        print("")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Describe the 7th breadth expansion experiment suites.")
    parser.add_argument("--profile", choices=sorted(RESOURCE_PROFILES.keys()), default="server_2048")
    parser.add_argument("--suite", choices=sorted(SUITES.keys()), default=None, help="Print one suite only.")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    profile = RESOURCE_PROFILES[args.profile]
    if args.suite is not None:
        print_profile(profile)
        print("")
        print_suite(args.suite, profile)
        return 0
    print_plan(profile)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
