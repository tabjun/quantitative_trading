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
    "quick_look.py",          # DB 적재 데이터 빠른 조회·시각화 확인 도구
}

# 2026-09-30: 모든 결과 산출물의 이름이 출처를 말해야 한다(AGENTS.md 2.3b). 2026-09-07
# 가정진단 작업이 `stage1_gate_20260907/`, `_stage2b_raw.md` 같은 실험번호 없는 이름으로
# 중간물 33개를 남겨, 3주 뒤 세션이 파일만 보고 어느 실험 결과인지 판단하지 못하고 커밋
# 이력을 역추적해야 했다. 종류는 상위 폴더(results/ vs images/)가 가르고, 출처는 그 바로
# 아래 이름이 가른다 — 따라서 **태그 디렉터리든 직접 놓인 파일이든 번호로 시작**해야 한다.
ARTIFACT_PREFIXES = (f"{REPO_SUBDIR}/test/results/", f"{REPO_SUBDIR}/test/images/")
# 특정 실험에 속하지 않는 교차 문서와, 번호 체계 이전(2026-07 이전) 산출물 — 소급 개명하지
# 않는다(실험 번호는 불변 식별자, known_pitfalls P6). 새 이름을 여기 추가할 때는 "이것이
# 정말 한 실험의 산출물이 아닌가"를 먼저 따진다.
ARTIFACT_ALLOWED = {
    # 여러 실험을 가로지르는 살아있는 문서
    "DATA_DICTIONARY.md", "EXPERIMENT_LOG.md", "SUMMARY_ONEPAGE_20260821.md",
    "MASTER_REPORT_20260814.md", "MASTER_REPORT_20260819.md",
    "governance_drift_rootcause_20260719.md",
    # 번호 체계 이전 산출물(2026-07~09)
    "assumption_report_20260907", "distribution_report_20260907",
    "scratch_volatility_20260907", "pipeline_report_20260908",
    "historical_flow_eval_20260723", "historical_flow_method_20260723",
    "historical_flow_methods_deep_20260723",
    "assumption_diagnosis_report_20260907.md",
    "distribution_identification_report_20260907.md",
    "scratch_volatility_report_20260907.md", "model_comparison_report_20260908.md",
    "pipeline_plan_report_20260908.md",
    "historical_flow_method_explainer_20260723.md",
    "historical_flow_metric_and_eval_20260723.md",
    "historical_flow_similarity_method_landscape_20260723.md",
    "paper_draft_nonstationary_crypto_trend_20260719.md",
    "professor_brief_publication_case_20260722.md",
    "professor_direction_recommendation_20260821.md",
    "text_context_feature_report_20260608_012237.md",
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

        if status.startswith("A"):
            for prefix in ARTIFACT_PREFIXES:
                if not normalized.startswith(prefix):
                    continue
                rest = normalized[len(prefix):]
                # ① results/ 또는 images/ 바로 아래 이름 — 태그 디렉터리일 수도, 파일일 수도 있다
                top = rest.split("/", 1)[0]
                if top not in ARTIFACT_ALLOWED and not top[:1].isdigit():
                    kind = "디렉터리" if "/" in rest else "파일"
                    errors.append(
                        f"{normalized}: 산출물 {kind} '{top}'가 실험번호로 시작하지 않습니다. "
                        "`{실험번호}_{실험약칭}_{YYYYMMDD}` 형식을 쓰세요(AGENTS.md 2.3b) — 이름만 "
                        "보고 어느 실험 산출물인지 알 수 있어야 합니다. 종류는 상위 폴더가 가르고"
                        "(results/=문서·표, images/=그림) 출처는 이 이름이 가릅니다. 한 실험의 "
                        "단계별 저장이면 태그를 새로 만들지 말고 파일명으로 가르세요. 특정 "
                        "실험에 속하지 않는 교차 문서라면 이 훅의 ARTIFACT_ALLOWED에 추가하세요."
                    )
                    break
                # ② 태그 디렉터리 안의 파일 이름도 같은 번호로 시작해야 한다. 그림·표가 메일
                #    첨부나 보고서 삽입으로 디렉터리 밖에 나가면 디렉터리 이름은 따라가지
                #    않기 때문이다(2026-09-21 포스터 첨부가 실제 사례).
                if "/" in rest and top not in ARTIFACT_ALLOWED:
                    num = top.split("_", 1)[0]          # 예: "24_maxscale_refit_20260928" → "24"
                    base = rest.rsplit("/", 1)[1]
                    if base and not base.startswith(f"{num}_"):
                        errors.append(
                            f"{normalized}: 산출물 파일 '{base}'가 실험번호 '{num}_'로 시작하지 "
                            "않습니다(AGENTS.md 2.3b). 파일이 디렉터리 밖으로 나가도 출처를 "
                            f"잃지 않도록 `{num}_{{실험약칭}}_{{부분}}_{{산출물}}.{{확장자}}` "
                            "형식을 쓰세요. 드라이버에 STEM 상수를 두고 모든 저장 파일명을 "
                            "거기서 파생시키는 방식을 권장합니다."
                        )
                break

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
