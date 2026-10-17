"""서비스 공용 예외와 일부 실패 허용 반복 도우미.

원칙: 문제가 생긴 자리에서 아래 예외를 던지고, 잡는 곳은 경계 두 군데뿐이다.
- 항목 반복(종목·뉴스 출처): `each`가 예상한 예외만 잡아 실패 목록에 남기고 나머지를 계속 처리한다.
- 파이프라인 단계(`pipelines/service_pipeline.py`): `ServiceError`는 짧게, 그 밖의 예외는 추적 정보까지 기록한다.
예상하지 못한 예외(코드 결함)는 `each`에서 잡지 않고 그대로 올려 드러나게 한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Generic, Iterable, TypeVar

K = TypeVar("K")
T = TypeVar("T")


class ServiceError(Exception):
    """서비스에서 예상한 실패의 공통 상위 예외."""


class ResourceUnavailable(ServiceError):
    """공유 서버 여유선(AGENTS.md 2.14) 부족이나 GPU 상태를 읽을 수 없음."""


class DataSourceError(ServiceError):
    """외부 데이터(업비트 API, 뉴스 RSS 등) 수집 실패."""


class DataIntegrityError(ServiceError):
    """저장된 입력이 없거나 기대한 형식이 아님."""


class ModelFitError(ServiceError):
    """통계 모형 적합 실패."""


class LlmError(ServiceError):
    """로컬 LLM 서버 시작·응답 실패."""


@dataclass
class BatchResult(Generic[K, T]):
    ok: dict[K, T] = field(default_factory=dict)
    failed: dict[K, str] = field(default_factory=dict)


def each(keys: Iterable[K], fn: Callable[[K], T], expected: tuple[type[BaseException], ...] = (ServiceError,)) -> BatchResult[K, T]:
    """keys마다 fn을 부른다. 예상한 예외만 실패로 모으고 계속한다. 그 밖의 예외는 그대로 올라간다."""
    out: BatchResult[K, T] = BatchResult()
    for k in keys:
        try:
            out.ok[k] = fn(k)
        except expected as exc:
            out.failed[k] = f"{type(exc).__name__}: {exc}"
    return out
