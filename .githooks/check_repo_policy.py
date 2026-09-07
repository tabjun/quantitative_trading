"""pre-commit 저장소 정책 검사."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_SUBDIR = "quantitative_trading"
TEST_SCRIPTS_PREFIX = f"{REPO_SUBDIR}/test/scripts/"

# 2026-08-09: stock은 연구 브랜치라 작업 지침 파일을 포함해 전량 커밋하지만,
# develop/main은 최종 결과물만 올라가는 브랜치라 이 파일들을 제외한다
# (AGENTS.md 2.7절). git merge는 이 규칙을 모르므로, 병합 커밋이든 일반
# 커밋이든 develop/main에서 이 파일들이 스테이징되면 여기서 차단한다.
GOVERNANCE_FILES = {
    f"{REPO_SUBDIR}/AGENTS.md",
    f"{REPO_SUBDIR}/CLAUDE.md",
    f"{REPO_SUBDIR}/conversation_l2_cache.md",
    f"{REPO_SUBDIR}/history.md",
    f"{REPO_SUBDIR}/process.md",
    f"{REPO_SUBDIR}/state.md",
}
GOVERNANCE_EXCLUDED_BRANCHES = {"develop", "main"}

# 2026-08-10: AGENTS.md 2.9절은 test/scripts에 "재사용 도구"는 허용하고 "이번 한 번만 쓰는
# ad-hoc 스크립트"만 금지하는데, 이 검사는 신규 .py를 무조건 막아 문서보다 엄격했다.
# 재사용 도구를 추가할 때만 여기에 파일명을 명시적으로 올린다 — 기본값은 여전히 차단이라
# ad-hoc 스크립트가 슬쩍 들어오는 것을 막는 원래 목적은 그대로 유지된다.
TEST_SCRIPTS_ALLOWED = {
    "mcp_client.py",          # MCP 서버(arxiv/HF) 직접 호출기 — AGENTS.md 5절
    "gpu_status.py",          # NVML 깨진 환경용 GPU 상태 조회(학습 중 감시)
    "run_t8_multiseed.sh",    # t8 다중 시드 실행기(잡음 정량화) — 재현 명령 보존
    "report_header.py",       # 보고서 표준 헤더 생성기(자기완결성) — AGENTS.md 2.9g
}


def run_git(args: list[str], cwd: Path) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True, encoding="utf-8").strip()


def repo_root() -> Path:
    return Path(run_git(["rev-parse", "--show-toplevel"], Path.cwd()))


def current_branch(root: Path) -> str:
    return run_git(["rev-parse", "--abbrev-ref", "HEAD"], root)


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
    branch = current_branch(root)
    staged = staged_name_status(root)
    errors: list[str] = []
    governance_errors: list[str] = []

    for status, path in staged:
        normalized = path.replace("\\", "/")
        if (
            status.startswith("A")
            and normalized.startswith(TEST_SCRIPTS_PREFIX)
            and normalized.endswith(".py")
            and normalized[len(TEST_SCRIPTS_PREFIX):] not in TEST_SCRIPTS_ALLOWED
        ):
            errors.append(
                f"{normalized}: test/scripts 아래에는 단발성 연구·워크플로우 스크립트를 추가하지 마세요. "
                "기존 스크립트, pipelines/, .githooks/, 또는 test/models/*.ipynb + *.py 미러를 사용하세요. "
                "재사용 도구라면(AGENTS.md 2.9절) check_repo_policy.py의 TEST_SCRIPTS_ALLOWED에 "
                "파일명을 명시적으로 추가하세요."
            )

        # 2026-07-22: 동명 .ipynb 미러 강제 규칙 폐지. 연구 실험은 .py 헤드리스 드라이버가
        # 기본이고(설명은 파일 내 # %% [markdown] 셀), ipynb 미러는 더 이상 요구하지 않는다.

        if branch in GOVERNANCE_EXCLUDED_BRANCHES and normalized in GOVERNANCE_FILES and status != "D":
            governance_errors.append(
                f"{normalized}: '{branch}' 브랜치에는 작업 지침 파일을 커밋하지 않습니다(stock 전용, "
                "AGENTS.md 2.7절). git merge stock으로 다시 들어왔다면 `git restore --staged --worktree "
                "-- <파일>`로 되돌리거나 `git rm --cached <파일>`로 제외한 뒤 다시 커밋하세요."
            )

    if governance_errors:
        return fail("지침 파일은 stock 브랜치에만 존재해야 합니다", governance_errors)
    if errors:
        return fail("저장소 워크플로우 정책 검사 실패", errors)
    return 0


if __name__ == "__main__":
    sys.exit(main())
