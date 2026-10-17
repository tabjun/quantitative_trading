"""서비스 공용 로깅 설정.

모든 서비스 모듈은 `logging.getLogger(__name__)`으로 로거를 받아 쓰고, 설정은 진입점(`main.py`·자체 시험)이
`configure_logging`을 한 번 불러 정한다. 파일에는 INFO 이상을 남기고(크기가 넘으면 교체), 화면에는 WARNING 이상만 낸다.
형식: `시각 수준 모듈 [실행 번호] 메시지`. 실행 번호는 파이프라인이 `set_run_id`로 넣는다.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s [%(run_id)s] %(message)s"
_run_id = "-"


class _RunIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = _run_id
        return True


def set_run_id(run_id: str) -> None:
    global _run_id
    _run_id = run_id


def configure_logging(log_dir: Path = Path("logs/service"), level: int = logging.INFO, console_level: int = logging.WARNING) -> Path:
    """서비스 루트 로거를 한 번만 설정하고 로그 파일 경로를 돌려준다. 다시 불러도 핸들러를 겹쳐 달지 않는다."""
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / "service.log"
    root = logging.getLogger()
    if any(getattr(h, "_service_handler", False) for h in root.handlers):
        return path
    fmt = logging.Formatter(LOG_FORMAT)
    file_h = RotatingFileHandler(path, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8")
    console_h = logging.StreamHandler()
    for h, lv in ((file_h, level), (console_h, console_level)):
        h.setLevel(lv)
        h.setFormatter(fmt)
        h.addFilter(_RunIdFilter())
        h._service_handler = True
        root.addHandler(h)
    root.setLevel(level)
    for noisy in ("urllib3", "matplotlib", "numba"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return path
