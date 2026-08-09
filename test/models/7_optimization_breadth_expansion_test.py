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
# - 기본 실행은 breadth/ensemble 실험 계획과 서버 자원 적용값을 출력한다.
# - 실제 학습은 학교 서버 또는 승인된 원격 환경에서만 수행한다.
# - 7번은 6번에서 고른 target, normalization, loss, model selection gate를 확장 단계에 반영한다.
# - 확장 과정에서도 shortcut collapse, 직전가 복사, near-zero return 붕괴가 재발하는지 계속 점검한다.
# - GPU 학습과 CPU 기반 통계/진단 후처리는 분리해서 본다.

# %%
from __future__ import annotations

import argparse
import ctypes
import multiprocessing
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


def bootstrap_repo_root() -> Path:
    candidates = [Path.cwd(), *Path.cwd().parents]
    try:
        current_file = Path(__file__).resolve()
        candidates.extend([current_file.parent, *current_file.parents])
    except NameError:
        pass

    for candidate in candidates:
        if (candidate / "database" / "paths.py").exists():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            return candidate
    raise ModuleNotFoundError("Run this notebook from inside the quantitative_trading repository.")


REPO_ROOT = bootstrap_repo_root()
DIAGNOSTIC_SCRIPT = "test/models/5_optimization_diagnostics_test.py"

BLAS_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


@dataclass(frozen=True)
class RuntimeResources:
    python_executable: str
    python_version: str
    cpu_logical: int | None
    cpu_physical: int | None
    memory_gb: float | None
    available_memory_gb: float | None
    torch_version: str | None
    torch_cuda_version: str | None
    cuda_available: bool
    gpu_name: str | None
    gpu_memory_gb: float | None


@dataclass(frozen=True)
class ResourceProfile:
    name: str
    device: str
    cpu_logical: int
    cpu_physical: int
    memory_gb: float
    available_memory_gb: float | None
    gpu_name: str | None
    gpu_memory_gb: float | None
    n_jobs: int
    max_workers: int
    num_workers: int
    torch_num_threads: int
    torch_interop_threads: int
    optuna_n_jobs: int
    batch_size: int | None
    pin_memory: bool
    max_windows: int
    epochs: int
    feature_set: str
    normalization: str
    explanation: str


RESOURCE_PROFILES = {
    "school_4090_15gb": ResourceProfile(
        name="school_4090_15gb",
        device="cuda",
        cpu_logical=32,
        cpu_physical=16,
        memory_gb=15.0,
        available_memory_gb=9.0,
        gpu_name="NVIDIA GeForce RTX 4090",
        gpu_memory_gb=24.0,
        n_jobs=16,
        max_workers=16,
        num_workers=4,
        torch_num_threads=16,
        torch_interop_threads=4,
        optuna_n_jobs=1,
        batch_size=48,
        pin_memory=True,
        max_windows=2048,
        epochs=60,
        feature_set="optimization_probe",
        normalization="window_standard",
        explanation="학교 서버 RTX 4090 / 가용 RAM 약 9GiB 기준의 기본 확장 프로필. GPU 학습은 순차 실행, CPU 후처리는 제한 병렬로 둔다.",
    ),
    "server_1024": ResourceProfile(
        name="server_1024",
        device="cuda",
        cpu_logical=16,
        cpu_physical=8,
        memory_gb=12.0,
        available_memory_gb=8.0,
        gpu_name="GPU autodetect",
        gpu_memory_gb=None,
        n_jobs=12,
        max_workers=12,
        num_workers=4,
        torch_num_threads=8,
        torch_interop_threads=4,
        optuna_n_jobs=1,
        batch_size=32,
        pin_memory=True,
        max_windows=1024,
        epochs=35,
        feature_set="optimization_probe",
        normalization="window_standard",
        explanation="빠른 breadth_probe 1차 확인용. GPU는 쓰되 CPU 병렬과 batch를 보수적으로 묶는다.",
    ),
    "server_2048": ResourceProfile(
        name="server_2048",
        device="cuda",
        cpu_logical=16,
        cpu_physical=8,
        memory_gb=12.0,
        available_memory_gb=8.0,
        gpu_name="GPU autodetect",
        gpu_memory_gb=None,
        n_jobs=16,
        max_workers=16,
        num_workers=4,
        torch_num_threads=12,
        torch_interop_threads=4,
        optuna_n_jobs=1,
        batch_size=48,
        pin_memory=True,
        max_windows=2048,
        epochs=60,
        feature_set="optimization_probe",
        normalization="window_standard",
        explanation="기본 본 실행 후보. 6번 안정화 이후 breadth expansion을 서버 자원 한도 안에서 순차 실행한다.",
    ),
    "server_light": ResourceProfile(
        name="server_light",
        device="auto",
        cpu_logical=8,
        cpu_physical=4,
        memory_gb=8.0,
        available_memory_gb=6.0,
        gpu_name=None,
        gpu_memory_gb=None,
        n_jobs=8,
        max_workers=8,
        num_workers=2,
        torch_num_threads=8,
        torch_interop_threads=2,
        optuna_n_jobs=1,
        batch_size=24,
        pin_memory=False,
        max_windows=768,
        epochs=25,
        feature_set="optimization_probe",
        normalization="window_standard",
        explanation="커널 복구와 계획 출력 점검용 경량 프로필. 성능 결론용이 아니라 연결 상태 확인용이다.",
    ),
}


@dataclass(frozen=True)
class ModelFamily:
    name: str
    category: str
    purpose: str
    why_now: str


MODEL_FAMILIES = [
    ModelFamily("Linear", "baseline", "가장 단순한 통제군", "복잡한 모델이 baseline보다 실제로 나은지 확인한다."),
    ModelFamily("LSTM", "recurrent", "기억 기반 recurrent 비교군", "5번 기준선을 유지하면서 더 넓은 breadth에 포함한다."),
    ModelFamily("GRU", "recurrent", "경량 recurrent 비교군", "LSTM보다 빠르게 안정화되는지 확인한다."),
    ModelFamily("TCN", "convolutional", "dilated convolution 비교군", "recurrent와 다른 smoothing 경향을 본다."),
    ModelFamily("Transformer", "attention", "attention 비교군", "복잡한 상호작용 구조가 실제로 도움이 되는지 본다."),
    ModelFamily("Autoformer-like", "decomposition", "trend/seasonal decomposition 계열 대리군", "3번 Autoformer 착시를 더 넓은 기준에서 다시 본다."),
    ModelFamily("PatchTST-like", "patch_attention", "patch token 기반 비교군", "입력 patching이 collapse를 줄이는지 본다."),
    ModelFamily("DLinear-like", "linear_decomposition", "decomposition linear 계열", "단순 구조가 오히려 안정적인지 확인한다."),
    ModelFamily("NLinear-like", "linear_residual", "residual linear 계열", "평균 복사형 쉬운 해를 얼마나 줄이는지 본다."),
    ModelFamily("TimesNet-like", "periodic_block", "주기 블록 기반 비교군", "주기 구조가 비정상 시계열에서도 유지되는지 본다."),
    ModelFamily("TimeXer-like", "exogenous_attention", "외생변수 친화 계열", "다음 독립변수 확장 전 기반 후보 가치가 있는지 본다."),
    ModelFamily("iTransformer-like", "inverted_attention", "변수 토큰 관점 비교군", "변수 차원을 중심으로 볼 때 안정화가 달라지는지 본다."),
    ModelFamily("ModernTCN-like", "modern_convolutional", "최근 convolutional 계열 비교군", "TCN 확장형이 더 넓은 창에서 유지되는지 본다."),
    ModelFamily("Mamba-like", "state_space", "state space 계열 비교군", "최근 sequence family가 금융 비정상 구간에서 어떤지 본다."),
]


@dataclass(frozen=True)
class EnsembleFamily:
    name: str
    members: tuple[str, ...]
    method: str
    purpose: str


ENSEMBLE_FAMILIES = [
    EnsembleFamily("LSTM + Autoformer", ("LSTM", "Autoformer-like"), "soft_average", "recurrent 기억과 decomposition 계열의 보완 여부를 본다."),
    EnsembleFamily("TCN + Transformer", ("TCN", "Transformer"), "soft_average", "convolutional smoothing과 attention 상호작용의 보완 여부를 본다."),
    EnsembleFamily("Linear + sequence residual stack", ("Linear", "PatchTST-like"), "residual_stack", "단순 baseline과 sequence model을 잔차 관점으로 결합한다."),
    EnsembleFamily("validation weighted top-k ensemble", ("top_k_from_breadth_probe",), "validation_weighted", "단일 후보의 우연성보다 검증 기준 가중 평균이 더 안정적인지 본다."),
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
        purpose="넓은 단일 모델군을 약식으로 훑어 collapse 방향과 baseline 돌파 가능성을 본다.",
        changed_knob="모델 family 폭을 넓힌다. target/loss/normalization 기본값은 6번 gate를 반영한다.",
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


def detect_cpu_physical() -> int | None:
    try:
        return multiprocessing.cpu_count() if platform.system() == "Windows" else None
    except NotImplementedError:
        return None


def detect_cpu_physical_fallback() -> int | None:
    system = platform.system()
    try:
        if system == "Linux":
            result = subprocess.run(
                ["bash", "-lc", "lscpu -p=Core | grep -v '^#' | sort -u | wc -l"],
                capture_output=True,
                text=True,
                check=True,
            )
            value = int(result.stdout.strip())
            return value if value > 0 else None
        if system == "Windows":
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor | Measure-Object -Property NumberOfCores -Sum).Sum"],
                capture_output=True,
                text=True,
                check=True,
            )
            value = int(result.stdout.strip())
            return value if value > 0 else None
    except Exception:
        return None
    return None


def detect_memory_gb() -> tuple[float | None, float | None]:
    system = platform.system()
    try:
        if system == "Linux":
            values: dict[str, int] = {}
            for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
                if ":" not in line:
                    continue
                key, raw = line.split(":", 1)
                number = raw.strip().split()[0]
                values[key] = int(number)
            total = values.get("MemTotal")
            available = values.get("MemAvailable")
            if total is not None:
                total_gb = round(total / 1024 / 1024, 2)
                available_gb = round(available / 1024 / 1024, 2) if available is not None else None
                return total_gb, available_gb
        if system == "Windows":
            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatus()
            status.dwLength = ctypes.sizeof(MemoryStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            total_gb = round(status.ullTotalPhys / 1024**3, 2)
            available_gb = round(status.ullAvailPhys / 1024**3, 2)
            return total_gb, available_gb
    except Exception:
        return None, None
    return None, None


def detect_runtime_resources() -> RuntimeResources:
    cpu_logical = os.cpu_count()
    cpu_physical = detect_cpu_physical_fallback() or detect_cpu_physical()
    memory_gb, available_memory_gb = detect_memory_gb()

    torch_version = None
    torch_cuda_version = None
    cuda_available = False
    gpu_name = None
    gpu_memory_gb = None
    try:
        import torch

        torch_version = torch.__version__
        torch_cuda_version = torch.version.cuda
        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory_gb = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
    except Exception:
        pass

    return RuntimeResources(
        python_executable=sys.executable,
        python_version=sys.version.replace("\n", " "),
        cpu_logical=cpu_logical,
        cpu_physical=cpu_physical,
        memory_gb=memory_gb,
        available_memory_gb=available_memory_gb,
        torch_version=torch_version,
        torch_cuda_version=torch_cuda_version,
        cuda_available=cuda_available,
        gpu_name=gpu_name,
        gpu_memory_gb=gpu_memory_gb,
    )


def apply_parallel_env(profile: ResourceProfile) -> None:
    for env_name in BLAS_ENV_VARS:
        os.environ[env_name] = str(profile.n_jobs)
    os.environ["QT_MAX_WORKERS"] = str(profile.max_workers)
    os.environ["QT_OPTUNA_N_JOBS"] = str(profile.optuna_n_jobs)
    os.environ["QT_SELECTED_RESOURCE_PROFILE"] = profile.name
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def apply_torch_thread_settings(profile: ResourceProfile) -> None:
    try:
        import torch
    except Exception:
        return

    try:
        torch.set_num_threads(profile.torch_num_threads)
    except Exception as exc:
        print(f"[resource] torch.set_num_threads failed: {exc}")

    try:
        torch.set_num_interop_threads(profile.torch_interop_threads)
    except RuntimeError as exc:
        print(f"[resource] torch.set_num_interop_threads skipped: {exc}")
    except Exception as exc:
        print(f"[resource] torch.set_num_interop_threads failed: {exc}")


def print_runtime_summary(profile: ResourceProfile, runtime: RuntimeResources) -> None:
    print("# runtime detection")
    print(f"- python executable: {runtime.python_executable}")
    print(f"- python version: {runtime.python_version}")
    print(f"- cpu logical / physical: {runtime.cpu_logical} / {runtime.cpu_physical}")
    print(f"- memory total / available (GiB): {runtime.memory_gb} / {runtime.available_memory_gb}")
    print(f"- torch version: {runtime.torch_version}")
    print(f"- torch cuda version: {runtime.torch_cuda_version}")
    print(f"- cuda available: {runtime.cuda_available}")
    print(f"- gpu name: {runtime.gpu_name}")
    print(f"- gpu memory (GiB): {runtime.gpu_memory_gb}")
    print(f"- selected resource profile: {profile.name}")
    print(f"- applied device: {profile.device}")
    print(f"- applied n_jobs: {profile.n_jobs}")
    print(f"- applied max_workers: {profile.max_workers}")
    print(f"- applied num_workers: {profile.num_workers}")
    print(f"- applied torch threads: intra={profile.torch_num_threads}, interop={profile.torch_interop_threads}")
    print(f"- applied optuna_n_jobs: {profile.optuna_n_jobs}")
    print(f"- applied batch_size: {profile.batch_size}")
    print(f"- applied pin_memory: {profile.pin_memory}")


def print_profile(profile: ResourceProfile) -> None:
    print(f"profile: {profile.name}")
    print(f"- device: {profile.device}")
    print(f"- cpu logical / physical: {profile.cpu_logical} / {profile.cpu_physical}")
    print(f"- memory total / available (GiB): {profile.memory_gb} / {profile.available_memory_gb}")
    print(f"- gpu name / memory (GiB): {profile.gpu_name} / {profile.gpu_memory_gb}")
    print(f"- n_jobs: {profile.n_jobs}")
    print(f"- max_workers: {profile.max_workers}")
    print(f"- num_workers: {profile.num_workers}")
    print(f"- torch threads: intra={profile.torch_num_threads}, interop={profile.torch_interop_threads}")
    print(f"- optuna_n_jobs: {profile.optuna_n_jobs}")
    print(f"- batch_size: {profile.batch_size}")
    print(f"- pin_memory: {profile.pin_memory}")
    print(f"- max_windows: {profile.max_windows}")
    print(f"- epochs: {profile.epochs}")
    print(f"- feature_set: {profile.feature_set}")
    print(f"- normalization: {profile.normalization}")
    print(f"- reason: {profile.explanation}")


def profile_env_prefix(profile: ResourceProfile) -> str:
    return " ".join(
        [
            f"OMP_NUM_THREADS={profile.n_jobs}",
            f"MKL_NUM_THREADS={profile.n_jobs}",
            f"OPENBLAS_NUM_THREADS={profile.n_jobs}",
            f"NUMEXPR_NUM_THREADS={profile.n_jobs}",
            f"QT_MAX_WORKERS={profile.max_workers}",
            f"QT_OPTUNA_N_JOBS={profile.optuna_n_jobs}",
            f"PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        ]
    )


def backend_command_for_suite(profile: ResourceProfile, suite_name: str) -> tuple[str | None, str]:
    batch_size = profile.batch_size if profile.batch_size is not None else 48
    suite_mapping = {
        "breadth_probe": "full_matrix",
        "normalization_cross_check": "architecture_probe",
        "loss_cross_check": "stabilization_loss_probe",
        "scale_confirmation": "full_matrix",
    }
    backend_suite = suite_mapping.get(suite_name)
    if backend_suite is None:
        return None, (
            "현재 저장소에는 이 suite를 직접 학습하는 전용 backend가 아직 없다. "
            "즉, expanded family ensemble 엔진을 별도 구현한 뒤 연결해야 한다."
        )

    command = (
        f"{profile_env_prefix(profile)} uv run {DIAGNOSTIC_SCRIPT} "
        f"--suite {backend_suite} "
        f"--feature-set {profile.feature_set} "
        f"--normalization {profile.normalization} "
        f"--max-windows {profile.max_windows} "
        f"--epochs {profile.epochs} "
        f"--batch-size {batch_size} "
        f"--num-workers {profile.num_workers} "
        f"--device {profile.device} "
        f"--save-csv"
    )
    note = (
        f"현재 5번 엔진에서 실행 가능한 representative pre-check는 `{backend_suite}`다. "
        "즉, 7번의 전체 breadth family를 모두 학습하는 완전 backend는 아니고, 6번 안정화 기준이 서버 자원 하에서 유지되는지 먼저 보는 용도다."
    )
    return command, note


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
    print("### resource-aware remote command skeleton")
    command, note = backend_command_for_suite(profile, suite.name)
    if command is None:
        print("- backend pending")
    else:
        print(command)
    print("")
    print("### execution note")
    print(f"- {note}")


def print_plan(profile: ResourceProfile, runtime: RuntimeResources) -> None:
    print("# 7. optimization breadth expansion stage plan")
    print_runtime_summary(profile, runtime)
    print("")
    print_profile(profile)
    print("")
    print("이 실험은 6번 gate를 유지한 채 더 넓은 모델군과 앙상블군을 한 번에 비교하기 위한 후속 오케스트레이터다.")
    print("현재 단계의 핵심은 서버 자원 한도 안에서 GPU 학습과 CPU 진단을 분리하고, 각 suite를 순차 실행해 collapse 재발 여부를 추적하는 것이다.")
    print("7번은 CPU-only 실행 스크립트가 아니다. 다만 지금 파일은 실제 딥러닝 breadth engine 전체를 담은 학습기라기보다, 6번 이후 확장을 위한 자원-인식 실행 계획 레이어다.")
    print("")
    for name in SUITES:
        print_suite(name, profile)
        print("")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Describe the 7th breadth expansion experiment suites.")
    parser.add_argument("--profile", choices=sorted(RESOURCE_PROFILES.keys()), default="school_4090_15gb")
    parser.add_argument("--suite", choices=sorted(SUITES.keys()), default=None, help="Print one suite only.")
    if argv is None:
        if "ipykernel" in sys.modules:
            args, _unknown = parser.parse_known_args()
            return args
        return parser.parse_args()
    return parser.parse_args(list(argv))


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    profile = RESOURCE_PROFILES[args.profile]
    apply_parallel_env(profile)
    apply_torch_thread_settings(profile)
    runtime = detect_runtime_resources()

    if args.suite is not None:
        print_runtime_summary(profile, runtime)
        print("")
        print_suite(args.suite, profile)
        return 0

    print_plan(profile, runtime)
    return 0


if __name__ == "__main__":
    if "ipykernel" in sys.modules:
        main()
    else:
        raise SystemExit(main())
