"""운영 기록 저장소: 파이프라인 실행·단계·항목별 실패·사용한 설정(초매개변수)을 서비스 DB에 남긴다.

표(모두 `ops_` 접두어, 서비스 DB `data/llm_agent.db`)
- `ops_runs`: 실행 한 번. 실행 번호, 시작·끝, 상태(running/ok/partial/failed), 실행 방법, 설정 JSON, 코드 버전(git), 호스트.
- `ops_step_runs`: 단계 한 번. 상태(ok/failed/skipped), 걸린 시간, 성공·실패 항목 수, 오류, 자원(RAM·부하·GPU).
- `ops_item_failures`: 단계 안에서 실패한 항목(종목·뉴스 출처)과 사유.
- `ops_param_sets`: 설정·초매개변수 묶음(구성 요소별 JSON). 내용 해시가 열쇠라 같은 설정은 한 번만 저장되고,
  `ops_step_runs.param_hash`가 그것을 가리킨다. 설정이 바뀐 실행에서만 새 행이 생긴다(더미를 쌓지 않는다).
  예: GARCH 척도, 국면 경계 버전, 규칙 경계, 지시문 버전, LLM 모델·문맥 길이.
판단 결과 자체는 `decision_log`·`baseline_decisions`에 있고, 실행 번호로 이 표들과 이어 볼 수 있다.
"""

from __future__ import annotations

import json
import logging
import socket
import subprocess
import uuid
from datetime import datetime

import duckdb

from common.db import content_hash

log = logging.getLogger(__name__)

DDL = [
    """CREATE TABLE IF NOT EXISTS ops_runs (
        run_id VARCHAR PRIMARY KEY, started_at TIMESTAMP, finished_at TIMESTAMP, status VARCHAR,
        trigger VARCHAR, config_json VARCHAR, code_version VARCHAR, host VARCHAR)""",
    """CREATE TABLE IF NOT EXISTS ops_step_runs (
        run_id VARCHAR, step VARCHAR, started_at TIMESTAMP, finished_at TIMESTAMP, status VARCHAR, param_hash VARCHAR,
        seconds DOUBLE, items_ok INTEGER, items_failed INTEGER, error VARCHAR, resources VARCHAR, summary_json VARCHAR)""",
    """CREATE TABLE IF NOT EXISTS ops_item_failures (
        run_id VARCHAR, step VARCHAR, item VARCHAR, error VARCHAR, logged_at TIMESTAMP)""",
    """CREATE TABLE IF NOT EXISTS ops_param_sets (
        param_hash VARCHAR PRIMARY KEY, component VARCHAR, params_json VARCHAR, first_seen TIMESTAMP)""",
]


def code_version() -> str:
    """git 짧은 해시. 커밋하지 않은 코드 변경이 있으면 `-dirty`를 붙인다(기록만 바뀐 경우도 포함)."""
    try:
        run = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, timeout=5, check=True).stdout.strip()
        return run("rev-parse", "--short", "HEAD") + ("-dirty" if run("status", "--porcelain", "--untracked-files=no") else "")
    except (OSError, subprocess.SubprocessError):
        return "unknown"


class OpsStore:
    def __init__(self, service_db: str):
        self.service_db = service_db
        with duckdb.connect(service_db) as con:
            for ddl in DDL:
                con.execute(ddl)

    def _exec(self, sql: str, rows: list[tuple]) -> None:
        if rows:
            with duckdb.connect(self.service_db) as con:
                con.executemany(sql, rows)

    def start_run(self, config: dict, trigger: str) -> str:
        run_id = f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
        self._exec("INSERT INTO ops_runs VALUES (?, ?, NULL, 'running', ?, ?, ?, ?)",
                   [(run_id, datetime.now(), trigger, json.dumps(config, ensure_ascii=False, default=str),
                     code_version(), socket.gethostname())])
        log.info("실행 시작 run_id=%s trigger=%s", run_id, trigger)
        return run_id

    def finish_run(self, run_id: str, status: str) -> None:
        with duckdb.connect(self.service_db) as con:
            con.execute("UPDATE ops_runs SET finished_at = ?, status = ? WHERE run_id = ?", [datetime.now(), status, run_id])
        log.info("실행 끝 run_id=%s status=%s", run_id, status)

    def record_step(self, run_id: str, step: str, started_at: datetime, status: str, result: dict, param_hash: str) -> None:
        failed = result.get("failed") or {}
        self._exec("INSERT INTO ops_step_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", [(
            run_id, step, started_at, datetime.now(), status, param_hash, result.get("seconds"),
            result.get("items_ok"), len(failed), result.get("error"), result.get("resources"),
            json.dumps({k: v for k, v in result.items() if k not in ("failed", "trace")}, ensure_ascii=False, default=str),
        )])
        self._exec("INSERT INTO ops_item_failures VALUES (?, ?, ?, ?, ?)",
                   [(run_id, step, str(item), err, datetime.now()) for item, err in failed.items()])

    def record_params(self, component: str, params: dict) -> str:
        """설정 묶음의 해시를 돌려준다. 처음 보는 설정일 때만 저장한다."""
        h = content_hash({"component": component, "params": params})
        with duckdb.connect(self.service_db) as con:
            if not con.execute("SELECT 1 FROM ops_param_sets WHERE param_hash = ?", [h]).fetchone():
                con.execute("INSERT INTO ops_param_sets VALUES (?, ?, ?, ?)",
                            [h, component, json.dumps(params, ensure_ascii=False, sort_keys=True, default=str), datetime.now()])
                log.info("새 설정 묶음 %s %s", component, h)
        return h
