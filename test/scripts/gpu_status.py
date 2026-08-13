"""GPU 상태 조회 — NVML이 깨진 환경에서도 동작한다.

## 왜 이 도구가 필요한가 (2026-08-14)

이 서버의 `nvidia-smi`는 다음 오류로 동작하지 않는다.

    Failed to initialize NVML: Driver/library version mismatch
    NVML library version: 595.84

원인은 **유저스페이스 라이브러리와 커널 모듈의 버전 불일치**다.

- 설치된 NVIDIA 패키지: 전부 `595.84`(2026-06-30 설치) — 이미 최신
- 실행 중인 커널 모듈: `595.71.05` (`/proc/driver/nvidia/version`)

즉 패키지 업그레이드 후 재부팅을 하지 않아 구버전 모듈이 계속 돌고 있는 상태다.
**유저스페이스 패키지로는 고칠 수 없다** — `nvidia-ml-py`(파이썬 래퍼)를 설치해도 같은
`libnvidia-ml.so`를 감싸므로 `NVMLError_LibRmVersionMismatch: RM has detected an NVML/RM
version mismatch`로 동일하게 실패한다(실측 확인). 근본 해결은 재부팅 또는 커널 모듈 재적재이며
둘 다 root 권한이 필요하다(이 계정은 sudo 불가).

**다행히 계산은 전혀 영향받지 않는다.** torch는 NVML이 아니라 CUDA 드라이버 API(`libcuda.so`)를
쓰고, 그쪽은 버전이 맞아 정상 동작한다(RTX 4090 실연산 검증). 그래서 이 도구는 드라이버 API로
얻을 수 있는 것(장치명·메모리)을 먼저 보여주고, NVML이 필요한 항목(사용률·온도·전력)은
가능할 때만 덧붙인다. 재부팅 이후에는 NVML 항목도 자동으로 함께 나온다.

## 사용법

    python test/scripts/gpu_status.py               # 1회 조회
    python test/scripts/gpu_status.py --watch 10    # 10초 간격 연속 조회(학습 중 감시)
    python test/scripts/gpu_status.py --json        # 기계 판독용
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def kernel_module_version() -> str | None:
    """`/proc/driver/nvidia/version`에서 실제로 돌고 있는 커널 모듈 버전을 읽는다."""
    path = Path("/proc/driver/nvidia/version")
    if not path.exists():
        return None
    for token in path.read_text(encoding="utf-8", errors="replace").split():
        # "NVRM version: ... 595.71.05 Release Build ..." 형태에서 x.y.z 토큰을 집는다
        parts = token.split(".")
        if len(parts) >= 2 and all(p.isdigit() for p in parts):
            return token
    return None


def nvml_metrics(index: int) -> tuple[dict, str | None]:
    """NVML로만 얻을 수 있는 항목. 실패하면 (빈dict, 이유)를 돌려준다."""
    try:
        import pynvml
    except ImportError:
        return {}, "nvidia-ml-py 미설치 (uv pip install nvidia-ml-py)"
    try:
        pynvml.nvmlInit()
    except Exception as exc:  # noqa: BLE001
        return {}, f"{type(exc).__name__}: {exc}"
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        rates = pynvml.nvmlDeviceGetUtilizationRates(handle)
        return {
            "driver_version": pynvml.nvmlSystemGetDriverVersion(),
            "gpu_utilization_pct": rates.gpu,
            "memory_utilization_pct": rates.memory,
            "temperature_c": pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU),
            "power_draw_w": pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0,
        }, None
    except Exception as exc:  # noqa: BLE001
        return {}, f"{type(exc).__name__}: {exc}"
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def collect(index: int = 0) -> dict:
    try:
        import torch
    except ImportError:
        return {"error": "torch 미설치"}

    if not torch.cuda.is_available():
        return {"error": "torch.cuda.is_available() == False", "kernel_module": kernel_module_version()}

    free, total = torch.cuda.mem_get_info(index)          # 드라이버 API — NVML 불필요
    status = {
        "device_name": torch.cuda.get_device_name(index),
        "device_count": torch.cuda.device_count(),
        "memory_total_gb": total / 1e9,
        "memory_free_gb": free / 1e9,
        "memory_used_gb": (total - free) / 1e9,
        "memory_used_pct": 100.0 * (total - free) / total,
        # 아래 두 항목은 이 파이썬 프로세스가 잡은 양(다른 프로세스는 포함 안 됨)
        "torch_allocated_gb": torch.cuda.memory_allocated(index) / 1e9,
        "torch_reserved_gb": torch.cuda.memory_reserved(index) / 1e9,
        "kernel_module": kernel_module_version(),
        "torch_cuda_build": torch.version.cuda,
    }
    extra, reason = nvml_metrics(index)
    status.update(extra)
    if reason:
        status["nvml_unavailable"] = reason
    return status


def render(status: dict) -> str:
    if "error" in status:
        return f"[gpu] 사용 불가: {status['error']}"
    lines = [
        f"[gpu] {status['device_name']} (장치 {status['device_count']}개, "
        f"torch cuda 빌드 {status['torch_cuda_build']}, 커널 모듈 {status['kernel_module']})",
        f"  메모리: {status['memory_used_gb']:.2f} / {status['memory_total_gb']:.2f} GB "
        f"({status['memory_used_pct']:.1f}% 사용, 여유 {status['memory_free_gb']:.2f} GB)",
        f"  이 프로세스 torch: allocated {status['torch_allocated_gb']:.2f} GB / "
        f"reserved {status['torch_reserved_gb']:.2f} GB",
    ]
    if "gpu_utilization_pct" in status:
        lines.append(
            f"  사용률 {status['gpu_utilization_pct']}% · 온도 {status['temperature_c']}C · "
            f"전력 {status['power_draw_w']:.0f}W · 드라이버 {status['driver_version']}"
        )
    else:
        lines.append(f"  (사용률·온도·전력은 NVML 필요 — 현재 불가: {status['nvml_unavailable']})")
        lines.append("   근본 해결은 재부팅(커널 모듈을 설치된 패키지 버전으로 재적재). 계산에는 영향 없음.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="NVML이 깨진 환경에서도 동작하는 GPU 상태 조회")
    parser.add_argument("--index", type=int, default=0, help="장치 번호")
    parser.add_argument("--watch", type=float, default=0, help="초 단위 반복 간격(0이면 1회)")
    parser.add_argument("--json", action="store_true", help="JSON으로 출력")
    args = parser.parse_args(argv)

    while True:
        status = collect(args.index)
        print(json.dumps(status, ensure_ascii=False, indent=2) if args.json else render(status), flush=True)
        if args.watch <= 0:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main(sys.argv[1:])
