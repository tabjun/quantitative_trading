from __future__ import annotations

from pathlib import Path


DEFAULT_DB_PATH = Path("data") / "upbit_data.db"
LEGACY_DB_PATH = Path("upbit_data.db")
# 서비스 트랙이 쓰는 테이블(뉴스·텍스트 특징·유사 흐름 마트·판단 로그)은 여기에만 쓴다.
# 시세 DB(upbit_data*.db)는 연구 실행이 함께 읽으므로 서비스 코드는 read_only로만 연다.
SERVICE_DB_PATH = Path("data") / "llm_agent.db"


def ensure_data_dir() -> Path:
    data_dir = DEFAULT_DB_PATH.parent
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def resolve_db_path(db_path: str | Path | None = None) -> str:
    if db_path:
        return str(Path(db_path))
    if DEFAULT_DB_PATH.exists():
        return str(DEFAULT_DB_PATH)
    if LEGACY_DB_PATH.exists():
        return str(LEGACY_DB_PATH)
    ensure_data_dir()
    return str(DEFAULT_DB_PATH)


def resolve_service_db_path(db_path: str | Path | None = None) -> str:
    """서비스 쓰기용 DB 경로. 파일이 없어도 되며(처음 쓸 때 생성), 시세 DB를 가리키면 거부한다."""
    path = Path(db_path) if db_path else SERVICE_DB_PATH
    if path.name.startswith("upbit_data"):
        raise ValueError(f"서비스 테이블을 시세 DB에 쓰지 않는다: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)
