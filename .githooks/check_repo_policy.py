"""pre-commit 저장소 정책 검사."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_SUBDIR = "quantitative_trading"
TEST_SCRIPTS_PREFIX = f"{REPO_SUBDIR}/test/scripts/"
TEST_MODELS_PREFIX = f"{REPO_SUBDIR}/test/models/"


def run_git(args: list[str], cwd: Path) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True, encoding="utf-8").strip()


def repo_root() -> Path:
    return Path(run_git(["rev-parse", "--show-toplevel"], Path.cwd()))


def staged_name_status(root: Path) -> list[tuple[str, str]]:
    output = run_git(["diff", "--cached", "--name-status", "--diff-filter=ACMRT"], root)
    rows = []
    for line in output.splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            rows.append((parts[0], parts[-1]))
    return rows


def fail(message: str, details: list[str]) -> int:
    print(f"[repo-policy] {message}", file=sys.stderr)
    for detail in details:
        print(f"  - {detail}", file=sys.stderr)
    return 1


def main() -> int:
    root = repo_root()
    staged = staged_name_status(root)
    errors: list[str] = []

    for status, path in staged:
        normalized = path.replace("\\", "/")
        if status.startswith("A") and normalized.startswith(TEST_SCRIPTS_PREFIX) and normalized.endswith(".py"):
            errors.append(
                f"{normalized}: test/scripts 아래에는 단발성 연구·워크플로우 스크립트를 추가하지 마세요. "
                "기존 스크립트, pipelines/, .githooks/, 또는 test/models/*.ipynb + *.py 미러를 사용하세요."
            )

        if normalized.startswith(TEST_MODELS_PREFIX) and normalized.endswith(".py"):
            sibling = root / Path(normalized).with_suffix(".ipynb")
            if not sibling.exists():
                errors.append(f"{normalized}: 같은 이름의 .ipynb 연구 노트북이 없습니다.")

    if errors:
        return fail("저장소 워크플로우 정책 검사 실패", errors)
    return 0


if __name__ == "__main__":
    sys.exit(main())
