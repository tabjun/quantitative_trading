"""서비스 DB 쓰기 공용 함수: 같은 값은 다시 쓰지 않는다.

사용자 원칙(2026-10-18): 같은 시험을 다시 돌리면 이전 기록을 그대로 쓰고, 새 값이나 바뀐 값만 넣는다(무조건 넣으면
더미만 쌓인다). `write_changed`는 열쇠 열로 기존 행과 맞춰 보고, 없는 열쇠는 넣고, 비교 열이 달라진 행만 바꾸고,
같은 행은 건너뛴다. 결과 건수를 로그와 반환값으로 남긴다.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass

import duckdb
import pandas as pd

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class WriteCount:
    inserted: int
    updated: int
    unchanged: int


def content_hash(obj: object) -> str:
    """JSON으로 바꿀 수 있는 값의 안정적인 해시(키 순서 고정)."""
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()[:16]


def add_columns(service_db: str, table: str, columns: dict[str, str]) -> None:
    """예전 스키마 표에 새 열을 더한다(표가 없으면 아무것도 안 한다)."""
    with duckdb.connect(service_db) as con:
        if con.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?", [table]).fetchone()[0]:
            for name, typ in columns.items():
                con.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {name} {typ}")


def write_changed(service_db: str, table: str, df: pd.DataFrame, keys: list[str], compare: list[str]) -> WriteCount:
    if df.empty:
        return WriteCount(0, 0, 0)
    q = lambda name: '"' + name.replace('"', '""') + '"'  # 열 이름이 SQL 예약어(at 등)여도 되게 감싼다
    on = " AND ".join(f"t.{q(k)} = n.{q(k)}" for k in keys)
    same = " AND ".join(f"t.{q(c)} IS NOT DISTINCT FROM n.{q(c)}" for c in compare) or "TRUE"  # 비교 열이 없으면 새 열쇠만 넣는다
    with duckdb.connect(service_db) as con:
        con.register("n", df)
        con.execute(f"CREATE TABLE IF NOT EXISTS {table} AS SELECT * FROM n WHERE FALSE")
        existing = con.execute(f"SELECT COUNT(*) FROM n WHERE EXISTS (SELECT 1 FROM {table} t WHERE {on})").fetchone()[0]
        unchanged = con.execute(f"SELECT COUNT(*) FROM n WHERE EXISTS (SELECT 1 FROM {table} t WHERE {on} AND {same})").fetchone()[0]
        # 바뀐 행은 지우고 다시 넣고, 없던 행은 넣는다. 같은 행은 손대지 않는다.
        con.execute(f"DELETE FROM {table} t USING n WHERE {on} AND NOT ({same})")
        cols = ", ".join(q(c) for c in df.columns)
        con.execute(f"INSERT INTO {table} ({cols}) SELECT {cols} FROM n WHERE NOT EXISTS (SELECT 1 FROM {table} t WHERE {on})")
        con.unregister("n")
    count = WriteCount(len(df) - existing, existing - unchanged, unchanged)
    log.info("%s 쓰기: 새로 %d, 바뀜 %d, 같음(건너뜀) %d", table, count.inserted, count.updated, count.unchanged)
    return count
