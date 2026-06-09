from __future__ import annotations

from pathlib import Path


DEFAULT_DB_PATH = Path("data") / "upbit_data.db"
LEGACY_DB_PATH = Path("upbit_data.db")


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
