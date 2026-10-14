"""저장소 정책 검사: pre-commit, pre-merge-commit, pre-push(`--pre-push`)가 함께 쓴다."""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path


# 2026-10-15: personal_ai_project 모노레포에서 독립 저장소로 분리되어 이 폴더가 곧 저장소 루트다.
# 경로 접두어는 이제 빈 문자열이다(예전 값 "quantitative_trading/").
REPO_PREFIX = ""
TEST_SCRIPTS_PREFIX = f"{REPO_PREFIX}test/scripts/"

# 2026-08-09: stock은 연구 브랜치라 작업 지침 파일을 포함해 전량 커밋하지만,
# develop/main은 최종 결과물만 올라가는 브랜치라 이 파일들을 제외한다
# (AGENTS.md 2.7절). git merge는 이 규칙을 모르므로, 병합 커밋이든 일반
# 커밋이든 develop/main에서 이 파일들이 스테이징되면 여기서 차단한다.
# 2026-10-14 실측(격리 복제본): 충돌 없는 일반 `git merge`는 pre-commit을 실행하지 않아 develop에
# 지침 파일이 그대로 들어왔고, fast-forward 병합은 커밋 훅이 아예 없다. 그래서 자동 병합 커밋은
# pre-merge-commit이, 두 경우 모두 push 시점은 pre-push(`--pre-push`)가 같은 규칙으로 막는다.
GOVERNANCE_FILES = {
    f"{REPO_PREFIX}AGENTS.md",
    f"{REPO_PREFIX}CLAUDE.md",
    f"{REPO_PREFIX}chat_history.md",
    f"{REPO_PREFIX}conversation_l2_cache.md",  # 이전 이름 재유입도 차단
    f"{REPO_PREFIX}history.md",
    f"{REPO_PREFIX}process.md",
    f"{REPO_PREFIX}state.md",
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
ARTIFACT_PREFIXES = (f"{REPO_PREFIX}test/results/", f"{REPO_PREFIX}test/images/")
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


# 2026-10-14: 설계는 드라이버 서두 설계 셀에 먼저 쓰고 보고서 "실험 설계" 절로 옮긴다(AGENTS.md 2.9k).
# 계획서를 test/experiment_specs/에 따로 두자 보고서에 설계가 실리지 않았고, 18번 이후에는 설계가
# 드라이버 주석에만 남았다. 같은 희석이 되풀이되지 않도록 커밋 단계에서 막는다. 규칙을 만든
# 시점 이전 실험(27번까지)은 소급 차단하지 않는다.
SPECS_PREFIX = f"{REPO_PREFIX}test/experiment_specs/"
MATERIALS_PREFIX = f"{REPO_PREFIX}test/research_materials/"
MATERIALS_BLOCKED_WORDS = ("plan", "design", "spec", "report")
MODELS_PREFIX = f"{REPO_PREFIX}test/models/"
RESULTS_PREFIX = f"{REPO_PREFIX}test/results/"
DESIGN_RULE_FROM = 28


def experiment_number(name: str) -> int | None:
    """'28b_xxx' → 28. 번호로 시작하지 않으면 None."""
    digits = ""
    for ch in name:
        if not ch.isdigit():
            break
        digits += ch
    return int(digits) if digits else None


def staged_text(root: Path, path: str) -> str:
    return subprocess.check_output(["git", "show", f":{path}"], cwd=root, text=True, encoding="utf-8")


def section_body(text: str, heading: str) -> str:
    """`## 제목` 절의 본문(다음 `\n## ` 앞까지). 없으면 빈 문자열."""
    i = text.find("\n" + heading)
    if i < 0:
        return ""
    j = text.find("\n## ", i + len(heading) + 1)
    return text[i:j if j > 0 else len(text)]


def design_section_ok(text: str) -> bool:
    """제목만이 아니라 실제 설계 내용(인용 줄이 아닌 본문 5줄 이상)이 있어야 한다."""
    body = section_body(text, "## 실험 설계").splitlines()[1:]
    return sum(1 for ln in body if ln.strip() and not ln.startswith(">")) >= 5


def describe_ok(text: str) -> bool:
    """실제 describe 표(`| 종목 | count |` 머리행), 당시 노트북 출력 원문(`count` 행), 또는 describe 절의 '해당 없음'."""
    if "| 종목 | count |" in text or re.search(r"^count\s", text, flags=re.M):
        return True
    return "해당 없음" in section_body(text, "## 기초통계량(describe 표)")


def calls_design(text: str) -> bool:
    """AST로 `render_design_section`/`_design_section` 호출이 실제로 있는지 본다(주석·문자열은 불인정)."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return "render_design_section(" in text
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else ""
            if name in ("render_design_section", "_design_section"):
                return True
    return False


def head_text(root: Path, path: str) -> str:
    """HEAD에 있는 파일 내용. 새 파일이면 빈 문자열."""
    r = subprocess.run(["git", "show", f"HEAD:{path}"], cwd=root, capture_output=True, text=True, encoding="utf-8")
    return r.stdout if r.returncode == 0 else ""


def design_errors(root: Path, status: str, path: str) -> list[str]:
    """AGENTS.md 2.9k 검사: 신규 실험 드라이버의 설계 셀, 보고서의 실험 설계 절과 describe 표."""
    out: list[str] = []
    if status.startswith("A") and path.startswith(SPECS_PREFIX):
        out.append(f"{path}: test/experiment_specs/는 2026-10-14에 폐지됐습니다. 설계는 드라이버 맨 앞 "
                   "`# %% [markdown]` 셀의 `## 실험 설계`에 쓰고 보고서로 옮기세요(AGENTS.md 2.9k).")
    if status.startswith("A") and path.startswith(MATERIALS_PREFIX):
        base = path.rsplit("/", 1)[1].lower()
        if any(w in base for w in MATERIALS_BLOCKED_WORDS):
            out.append(f"{path}: research_materials/에는 설계서·계획서·보고서 이름의 파일을 두지 않습니다. 설계는 드라이버 "
                       "설계 셀과 보고서 실험 설계 절로, 결과 보고서는 test/results/로 보내세요. 문헌 정리와 "
                       "알고리즘·방법론 카탈로그, 데이터 정의는 research_materials/에 둘 수 있습니다(AGENTS.md 2.9k).")
    if status.startswith("D"):
        return out
    # 소급 보호(2026-10-14): 이미 설계 절·describe 표·설계 출력 호출을 갖춘 과거 보고서·드라이버는 번호와
    # 무관하게 이후 수정에서 그것들을 잃지 않는다. 28번 미만 파일 전체에 새로 요구하지는 않는다.
    if path.startswith(RESULTS_PREFIX) and path.endswith(".md"):
        old = head_text(root, path)
        if old:
            new = staged_text(root, path)
            if design_section_ok(old) and not design_section_ok(new):
                out.append(f"{path}: 기존에 있던 `## 실험 설계` 절의 내용이 사라졌거나 제목만 남았습니다(AGENTS.md 2.9k).")
            if describe_ok(old) and not describe_ok(new):
                out.append(f"{path}: 기존에 있던 describe 표(또는 해당 없음 표기)의 내용이 사라졌습니다. 제목만 남기면 인정하지 "
                           "않습니다(AGENTS.md 2.9k).")
    if path.startswith(MODELS_PREFIX) and path.endswith(".py"):
        if calls_design(head_text(root, path)) and not calls_design(staged_text(root, path)):
            out.append(f"{path}: 기존에 있던 `render_design_section` 호출(보고서에 설계 절을 싣는 코드)이 사라졌습니다(AGENTS.md 2.9k).")
    if path.startswith(MODELS_PREFIX) and path.endswith("_test.py"):
        base = path[len(MODELS_PREFIX):]
        num = experiment_number(base)
        if num is not None and num >= DESIGN_RULE_FROM:
            text = staged_text(root, path)
            lines = text.splitlines()
            cell = []
            for line in lines[1:]:
                if line.startswith("# %%") or not line.startswith("#"):
                    break
                cell.append(line)
            if not lines or lines[0].strip() != "# %% [markdown]" or not any(
                    l.startswith("# ## 실험 설계") for l in cell):
                out.append(f"{path}: 드라이버 맨 앞 `# %% [markdown]` 셀에 `## 실험 설계` 절이 없습니다. 실행 전에 "
                           "왜·연구 질문·비교 대상·고정 조건·테스트 케이스·판정 기준을 쓰세요(AGENTS.md 2.9k).")
            if not calls_design(text):
                out.append(f"{path}: 보고서 작성부가 `report_header.render_design_section(__file__)`를 부르지 않습니다. "
                           "설계 셀 원문을 보고서 서두에 자동으로 실어야 합니다(AGENTS.md 2.9k).")
    if path.startswith(RESULTS_PREFIX) and path.endswith("_report.md"):
        base = path[len(RESULTS_PREFIX):].rsplit("/", 1)[-1]
        num = experiment_number(base)
        if num is not None and num >= DESIGN_RULE_FROM:
            text = staged_text(root, path)
            if not design_section_ok(text):
                out.append(f"{path}: 보고서 서두에 `## 실험 설계` 절이 없거나 제목뿐입니다(AGENTS.md 2.9k).")
            if not describe_ok(text):
                out.append(f"{path}: 표준 헤더 5항의 describe 표가 없습니다. `render_standard_header`로 만드세요"
                           "(AGENTS.md 2.9k).")
    return out


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

        errors.extend(design_errors(root, status, normalized))

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


def pre_push() -> int:
    """develop/main으로 가는 push에 지침 파일이 있으면 막는다. stdin은 git이 주는 `<ref> <sha> <원격 ref> <원격 sha>`."""
    root = repo_root()
    errors: list[str] = []
    for line in sys.stdin.read().splitlines():
        parts = line.split()
        if len(parts) != 4:
            continue
        _local_ref, local_sha, remote_ref, _remote_sha = parts
        branch = remote_ref[len("refs/heads/"):] if remote_ref.startswith("refs/heads/") else remote_ref
        if branch not in GOVERNANCE_EXCLUDED_BRANCHES or set(local_sha) == {"0"}:
            continue
        for path in sorted(GOVERNANCE_FILES):
            probe = subprocess.run(["git", "cat-file", "-e", f"{local_sha}:{path}"], cwd=root,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if probe.returncode == 0:
                errors.append(
                    f"{path}: '{branch}'로 push하려는 커밋({local_sha[:7]})에 들어 있습니다. stock 전용 파일입니다"
                    "(AGENTS.md 2.7절). 병합 결과에서 `git rm --cached -- <파일>`로 제외해 새 커밋을 만든 뒤 "
                    "다시 push하세요.")
    if errors:
        return fail("지침 파일이 develop/main push에 포함됐습니다", errors)
    return 0


if __name__ == "__main__":
    sys.exit(pre_push() if "--pre-push" in sys.argv[1:] else main())
