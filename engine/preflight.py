"""분석 시작 시 실행 환경·데이터 준비 상태를 한 번에 점검하고, 없으면 채운다.

## 왜 이 모듈이 있는가 (2026-08-19 신설)

CUDA는 실험 시작 시 `resources.environment_report()`가 확인해 주는데, **데이터는
아무도 확인해 주지 않았다.** 그래서 다음 사고가 실제로 있었다.

- DB 파일은 `.gitignore`의 `*.db*`로 제외돼 있어 `git pull`로 오지 않는다(1.28GB DuckDB).
  다른 환경에서는 재수집해야 하는데, 그 사실이 코드가 아니라 문서에만 있었다.
- `btc_15m_advance`만 있는 상태에서 실험이 조용히 돌아갔고, 그것이 "4번부터 BTC 단일
  테이블이 기본값으로 굳어진" 문제와 겹쳐 오래 발견되지 않았다.

이 모듈은 그 확인을 **코드로 강제**한다. CUDA 점검과 같은 자리에서 같은 방식으로,
DB·테이블·종목·행수를 점검하고 **부족하면 그 자리에서 수집한다**.

## 사용법

    from engine import preflight

    # 필요한 종목만 자동 확보(권장 — 요청한 범위만 수집하므로 빠르다)
    preflight.ensure_data(table="upbit_krw_candle", tickers=["KRW-BTC"])

    # 전 종목이 필요한 실험
    preflight.ensure_data(table="upbit_krw_candle", tickers="all")

    # 점검만 하고 수집은 하지 않음(CI·조회용)
    preflight.ensure_data(table="upbit_krw_candle", tickers=["KRW-BTC"], auto_build=False)

환경변수 `QT_NO_AUTOBUILD=1`이면 어떤 경우에도 수집하지 않고 보고만 한다
(장시간 수집이 곤란한 세션 보호).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

KRW_TABLE = "upbit_krw_candle"
MIN_ROWS = 5_000          # 이 미만이면 "사실상 비어 있다"로 본다
FULL_HISTORY_HINT = 100_000


def _repo_root() -> Path:
    start = Path(__file__).resolve().parent
    for candidate in [start, *start.parents]:
        if (candidate / "pyproject.toml").exists() and (candidate / "engine").is_dir():
            return candidate
    raise RuntimeError("quantitative_trading 디렉터리를 찾지 못했다.")


ROOT = _repo_root()


def resolve_db(db_arg: str | Path | None = None) -> Path:
    return Path(db_arg) if db_arg else ROOT / "data" / "upbit_data.db"


def inspect(db_path: Path, table: str, tickers: list[str] | None) -> dict:
    """DB·테이블·종목 상태를 조회한다. 부작용 없음(읽기 전용)."""
    import duckdb

    status = {
        "db_path": str(db_path),
        "db_exists": db_path.exists(),
        "db_size_gb": db_path.stat().st_size / 1e9 if db_path.exists() else 0.0,
        "table": table,
        "table_exists": False,
        "total_rows": 0,
        "n_tickers": 0,
        "missing_tickers": list(tickers) if tickers else [],
        "thin_tickers": [],
    }
    if not status["db_exists"]:
        return status

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        tables = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
        status["all_tables"] = sorted(tables)
        if table not in tables:
            return status
        status["table_exists"] = True
        columns = [row[1] for row in con.execute(f"PRAGMA table_info('{table}')").fetchall()]
        status["has_ticker_column"] = "ticker" in columns
        status["total_rows"] = int(con.execute(f"select count(*) from {table}").fetchone()[0])

        if "ticker" not in columns:
            status["n_tickers"] = 1
            status["missing_tickers"] = []
            if status["total_rows"] < MIN_ROWS:
                status["thin_tickers"] = [f"{table}(단일 테이블, {status['total_rows']:,}행)"]
            return status

        counts = dict(con.execute(f"select ticker, count(*) from {table} group by ticker").fetchall())
        status["n_tickers"] = len(counts)
        if tickers:
            status["missing_tickers"] = [t for t in tickers if t not in counts]
            status["thin_tickers"] = [
                f"{t}({counts[t]:,}행)" for t in tickers if t in counts and counts[t] < MIN_ROWS
            ]
        else:
            status["missing_tickers"] = []
            status["thin_tickers"] = [f"{t}({n:,}행)" for t, n in counts.items() if n < MIN_ROWS]
    finally:
        con.close()
    return status


def render(status: dict) -> str:
    lines = [
        f"[preflight] DB {status['db_path']} "
        f"({'존재' if status['db_exists'] else '**없음**'}"
        + (f", {status['db_size_gb']:.2f} GB" if status["db_exists"] else "")
        + ")"
    ]
    if status["db_exists"]:
        lines.append(
            f"  테이블 `{status['table']}`: "
            f"{'존재' if status['table_exists'] else '**없음**'}"
            + (f", {status['total_rows']:,}행 / {status['n_tickers']:,}종목" if status["table_exists"] else "")
        )
    if status["missing_tickers"]:
        preview = ", ".join(status["missing_tickers"][:8])
        more = f" 외 {len(status['missing_tickers']) - 8}개" if len(status["missing_tickers"]) > 8 else ""
        lines.append(f"  **누락 종목 {len(status['missing_tickers'])}개**: {preview}{more}")
    if status["thin_tickers"]:
        lines.append(f"  **행수 부족({MIN_ROWS:,} 미만)**: {', '.join(status['thin_tickers'][:8])}")
    if status["db_exists"] and status["table_exists"] and not status["missing_tickers"] and not status["thin_tickers"]:
        lines.append("  → 준비 완료")
    return "\n".join(lines)


def _collect(db_path: Path, table: str, targets: list[str] | str, years: float, sleep: float) -> int:
    """`pipelines/rebuild_price_mart.py`를 호출해 수집한다. 반환값은 종료 코드."""
    command = [
        sys.executable,
        str(ROOT / "pipelines" / "rebuild_price_mart.py"),
        "--db", str(db_path),
        "--table", table,
        "--years", str(years),
        "--sleep", str(sleep),
        "--resume",
    ]
    command += ["--tickers", "all" if targets == "all" else ",".join(targets)]
    print(f"[preflight] 수집 시작: {' '.join(command[-4:])}", flush=True)
    return subprocess.call(command, cwd=str(ROOT))


def ensure_data(
    table: str = KRW_TABLE,
    tickers: list[str] | str | None = None,
    db: str | Path | None = None,
    auto_build: bool = True,
    years: float = 3.0,
    sleep: float = 0.11,
    strict: bool = True,
) -> dict:
    """데이터가 준비됐는지 점검하고, 부족하면 수집해 채운 뒤 최종 상태를 돌려준다.

    tickers: 종목 리스트, `"all"`(전 종목), 또는 None(점검만).
    auto_build: False면 수집하지 않고 보고만 한다. 환경변수 `QT_NO_AUTOBUILD=1`이 우선한다.
    strict: 수집 후에도 부족하면 예외를 던진다(조용한 실패 방지 — 이 프로젝트의 반복 사고 유형).
    """
    db_path = resolve_db(db)
    requested = None if tickers is None else (tickers if tickers == "all" else list(tickers))
    check_list = None if requested in (None, "all") else requested

    status = inspect(db_path, table, check_list)
    print(render(status), flush=True)

    needs_build = (
        not status["db_exists"]
        or not status["table_exists"]
        or bool(status["missing_tickers"])
        or bool(status["thin_tickers"])
        or (requested == "all" and status["n_tickers"] < 2)
    )
    if not needs_build:
        return status

    if os.environ.get("QT_NO_AUTOBUILD") == "1" or not auto_build:
        message = "[preflight] 데이터가 부족하지만 자동 수집이 비활성화됐다(QT_NO_AUTOBUILD 또는 auto_build=False)."
        print(message, flush=True)
        if strict:
            raise RuntimeError(message + " 먼저 pipelines/rebuild_price_mart.py로 수집하라.")
        return status

    db_path.parent.mkdir(parents=True, exist_ok=True)
    targets = "all" if requested == "all" else (status["missing_tickers"] + [
        entry.split("(")[0] for entry in status["thin_tickers"] if entry.startswith("KRW-")
    ] or requested or [])
    if not targets:
        targets = requested or "all"

    code = _collect(db_path, table, targets, years, sleep)
    if code != 0:
        print(f"[preflight] 수집이 종료 코드 {code}로 끝났다.", flush=True)

    status = inspect(db_path, table, check_list)
    print("[preflight] 수집 후 상태:", flush=True)
    print(render(status), flush=True)
    if strict and (not status["table_exists"] or status["missing_tickers"] or status["thin_tickers"]):
        raise RuntimeError(
            "[preflight] 수집 후에도 데이터가 부족하다 — 조용히 진행하지 않고 중단한다. "
            f"누락={status['missing_tickers']}, 부족={status['thin_tickers']}"
        )
    return status


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="데이터 준비 상태 점검(+필요시 수집)")
    parser.add_argument("--table", default=KRW_TABLE)
    parser.add_argument("--tickers", default=None, help='쉼표 구분 또는 "all"')
    parser.add_argument("--db", default=None)
    parser.add_argument("--check-only", action="store_true", help="수집하지 않고 점검만")
    args = parser.parse_args(argv)

    tickers: list[str] | str | None = None
    if args.tickers == "all":
        tickers = "all"
    elif args.tickers:
        tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]

    ensure_data(
        table=args.table, tickers=tickers, db=args.db,
        auto_build=not args.check_only, strict=not args.check_only,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
