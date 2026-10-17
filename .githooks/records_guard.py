"""기록 4종 충돌 방지: 작업 트리 세션 잠금, 잠금 안에서의 요청 번호 배정, 커밋 시 검사(AGENTS.md §1).

배경(2026-10-18): 서비스 작업 트리에서 Claude Code와 Codex 세션이 동시에 기록을 쓰면서 S192·S193·S194가
겹쳤고, 한쪽이 기록 파일을 통째로 커밋해 다른 세션의 커밋 전 행까지 자기 커밋에 넣었다. "한 작업 트리에
한 세션" 규칙이 문서에만 있고 막는 장치가 없었기 때문이다. 이 도구는 그 규칙을 커밋 단계에서 강제한다.

    python .githooks/records_guard.py claim --role service   # 세션 시작: 역할을 밝히고 이 작업 트리를 잠근다
    python .githooks/records_guard.py next --track service --summary "요청 요약"   # 잠금 안에서 번호 예약
    python .githooks/records_guard.py status           # 잠금 주인·트랙별 중복 번호
    python .githooks/records_guard.py release          # 세션 끝: 잠금 해제

세션 = 명령을 실행한 프로세스의 조상 중 `claude` 또는 `codex` 프로세스(PID). 사람이 터미널에서 직접 하는
커밋(조상에 에이전트 없음)은 잠금 검사를 하지 않는다. 잠금 주인 프로세스가 죽었으면 잠금은 무효다.

역할(2026-10-18, AGENTS.md §1 "세션 역할"): service = `llm-agent` 트리, research = `stock` 트리, supervisor = 두 트리를
읽기만 하는 점검 세션. 역할과 트리 브랜치가 맞지 않으면 claim을 거부하고, supervisor는 어떤 트리도 잠그지 못한다
(변경은 해당 트랙 세션에 넘긴다). 2026-10-17~18에 점검 세션이 서비스 기록을 직접 써서 번호가 겹쳤던 것을 막는다.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import subprocess
import sys
from collections import Counter
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

LOCK_NAME = ".records_session.lock"  # 작업 트리 루트, gitignore 대상
AGENT_COMMS = {"claude", "codex"}
TRACKS = {
    "service": {"chat": "chat_history.md", "state": "state.md", "prefix": "S"},
    "research": {"chat": "test/chat_history.md", "state": "test/state.md", "prefix": ""},
}
ROLE_BRANCH = {"service": "llm-agent", "research": "stock"}
ROLES = (*ROLE_BRANCH, "supervisor")
RECORD_FILES = {
    "chat_history.md", "state.md", "history.md", "process.md",
    "test/chat_history.md", "test/state.md", "test/history.md", "test/process.md",
}
ROW_INDEX = re.compile(r"^\| (S?\d+[a-z]?) \|", re.M)


def repo_root() -> Path:
    return Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())


def _comm(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/comm").read_text().strip()
    except OSError:
        return ""


def _ppid(pid: int) -> int:
    try:
        return int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[1])
    except (OSError, IndexError, ValueError):
        return 0


def agent_pid(start: int | None = None) -> int | None:
    """이 프로세스의 조상 중 가장 가까운 claude/codex 프로세스."""
    pid = start or os.getpid()
    for _ in range(64):
        if pid <= 1:
            return None
        if _comm(pid) in AGENT_COMMS:
            return pid
        pid = _ppid(pid)
    return None


def alive(pid: int) -> bool:
    return _comm(pid) in AGENT_COMMS


def read_lock(root: Path) -> dict | None:
    path = root / LOCK_NAME
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def lock_owner(root: Path) -> dict | None:
    """살아 있는 잠금 주인. 주인 프로세스가 죽었으면 None."""
    lk = read_lock(root)
    return lk if lk and alive(int(lk.get("pid", 0))) else None


@contextmanager
def file_lock(root: Path):
    """번호 배정·잠금 변경을 원자적으로 하기 위한 짧은 flock."""
    with open(root / f"{LOCK_NAME}.flock", "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def branch(root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=root, text=True).strip()


def claim(root: Path, role: str, force: bool = False) -> tuple[bool, str]:
    me = agent_pid()
    if me is None:
        return False, "claude/codex 세션 안에서 실행해야 한다(사람의 직접 작업은 잠금이 필요 없다)"
    if role == "supervisor":
        return False, ("점검(supervisor) 세션은 작업 트리를 잠그지 않는다. 두 트리를 읽기만 하고, 기록·코드 변경은 "
                       "해당 트랙 세션에 넘길 인계 프롬프트로 낸다(AGENTS.md §1 세션 역할).")
    want = ROLE_BRANCH[role]
    if branch(root) != want:
        return False, f"역할 {role}은 {want} 브랜치 트리에서만 잠글 수 있다(이 트리: {branch(root)}, {root})"
    with file_lock(root):
        owner = lock_owner(root)
        if owner and int(owner["pid"]) != me and not force:
            return False, (f"다른 세션이 이 작업 트리를 쓰고 있다: {owner['comm']} PID {owner['pid']} "
                           f"(시작 {owner['claimed_at']}). 그 세션을 끝내거나 다른 작업 트리에서 시작하라.")
        info = {"pid": me, "comm": _comm(me), "role": role, "claimed_at": datetime.now().isoformat(timespec="seconds"),
                "root": str(root)}
        (root / LOCK_NAME).write_text(json.dumps(info, ensure_ascii=False))
    return True, f"잠금: {role} 세션 {info['comm']} PID {me} ({root})"


def release(root: Path) -> tuple[bool, str]:
    me = agent_pid()
    with file_lock(root):
        owner = read_lock(root)
        if not owner:
            return True, "잠금 없음"
        if alive(int(owner["pid"])) and int(owner["pid"]) != me:
            return False, f"잠금 주인이 아니다(주인 {owner['comm']} PID {owner['pid']})"
        (root / LOCK_NAME).unlink(missing_ok=True)
    return True, "해제"


def indices(text: str) -> list[str]:
    return ROW_INDEX.findall(text)


def _num(idx: str) -> int:
    return int(re.sub(r"\D", "", idx) or 0)


def next_index(root: Path, track: str, summary: str, tool: str) -> tuple[bool, str]:
    """잠금 주인만 번호를 받는다. 실제 파일을 다시 읽어 마지막 번호 + 1을 정하고 두 파일에 예약 행을 바로 쓴다."""
    me = agent_pid()
    owner = lock_owner(root)
    if me is None or not owner or int(owner["pid"]) != me:
        return False, "이 세션이 작업 트리 잠금을 갖고 있지 않다. 먼저 `records_guard.py claim`"
    if owner.get("role") != track:
        return False, f"이 세션의 역할({owner.get('role')})과 다른 트랙({track}) 번호는 받지 않는다"
    cfg = TRACKS[track]
    chat, state = root / cfg["chat"], root / cfg["state"]
    with file_lock(root):
        mine = [i for i in indices(chat.read_text()) + indices(state.read_text())
                if (i.startswith("S") if cfg["prefix"] else not i.startswith("S"))]
        idx = f"{cfg['prefix']}{max((_num(i) for i in mine), default=0) + 1}"
        today = datetime.now().strftime("%Y-%m-%d")
        note = f"기록 주체: {tool}; 예약 {datetime.now():%H:%M}"
        with open(chat, "a") as fh:
            fh.write(f"| {idx} | {today} | {summary} | (원문 기록 전) | 진행중 | {note} |\n")
        with open(state, "a") as fh:
            fh.write(f"| {idx} | {summary} | [ ] | {today} | 진행중 |\n")
    return True, idx


def duplicates(text: str) -> dict[str, int]:
    return {k: v for k, v in Counter(indices(text)).items() if v > 1}


def git_text(root: Path, spec: str) -> str:
    r = subprocess.run(["git", "show", spec], cwd=root, capture_output=True, text=True, encoding="utf-8")
    return r.stdout if r.returncode == 0 else ""


def commit_errors(root: Path, staged: list[str]) -> list[str]:
    """pre-commit에서 부른다. (1) 에이전트가 기록 파일을 커밋하면 이 세션이 잠금 주인이어야 한다.
    (2) 스테이징된 chat_history·state에서 번호가 새로 겹치면 안 된다(예전부터 있던 중복은 통과)."""
    errors: list[str] = []
    rec = [p for p in staged if p in RECORD_FILES]
    if not rec:
        return errors
    me = agent_pid()
    if me is not None:
        owner = lock_owner(root)
        if not owner:
            errors.append(f"기록 파일({', '.join(rec)})을 커밋하려면 이 세션이 작업 트리 잠금을 가져야 한다: "
                          "`python .githooks/records_guard.py claim`")
        elif int(owner["pid"]) != me:
            errors.append(f"다른 세션({owner.get('role', '?')} {owner['comm']} PID {owner['pid']})이 이 작업 트리 잠금을 갖고 있다. "
                          "그 세션의 커밋 전 행까지 함께 커밋하게 되므로 막는다. 한 작업 트리에는 한 세션만 둔다.")
        else:
            # 병합 커밋은 상대 트랙이 이미 커밋한 기록을 정상적으로 가져오므로 다른 트랙 검사에서 뺀다.
            # 자동 병합 커밋(pre-merge-commit)에는 MERGE_HEAD가 아직 없어 훅이 REPO_POLICY_MERGE=1을 준다.
            git_dir = Path(subprocess.check_output(["git", "rev-parse", "--absolute-git-dir"], cwd=root, text=True).strip())
            merging = (git_dir / "MERGE_HEAD").exists() or os.environ.get("REPO_POLICY_MERGE") == "1"
            other = [] if merging else [p for p in rec if (p.startswith("test/")) != (owner.get("role") == "research")]
            if other:
                errors.append(f"{owner.get('role')} 세션이 다른 트랙 기록({', '.join(other)})을 커밋하려 한다. "
                              "한 세션은 다른 트랙의 기록 파일을 고치지 않는다(AGENTS.md §1).")
    for cfg in TRACKS.values():
        for key in ("chat", "state"):
            path = cfg[key]
            if path not in rec:
                continue
            new, old = duplicates(git_text(root, f":{path}")), duplicates(git_text(root, f"HEAD:{path}"))
            grown = {k: v for k, v in new.items() if v > old.get(k, 1)}
            if grown:
                errors.append(f"{path}: 요청 번호가 새로 겹친다 {grown}. `records_guard.py next`로 번호를 받는다.")
    return errors


def main() -> int:
    ap = argparse.ArgumentParser(description="기록 4종 충돌 방지 도구")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("claim")
    c.add_argument("--role", choices=ROLES, required=True, help="이 세션의 역할(AGENTS.md §1 세션 역할)")
    c.add_argument("--force", action="store_true", help="살아 있는 다른 주인의 잠금을 빼앗는다(사용자 지시 때만)")
    sub.add_parser("release")
    sub.add_parser("status")
    n = sub.add_parser("next")
    n.add_argument("--track", choices=sorted(TRACKS), required=True)
    n.add_argument("--summary", default="(요약 기록 전)")
    n.add_argument("--tool", default=None, help="기록 주체 표시(기본: 잠금 주인 프로세스 이름)")
    args = ap.parse_args()
    root = repo_root()
    if args.cmd == "claim":
        ok, msg = claim(root, args.role, args.force)
    elif args.cmd == "release":
        ok, msg = release(root)
    elif args.cmd == "next":
        ok, msg = next_index(root, args.track, args.summary, args.tool or _comm(agent_pid() or 0) or "unknown")
    else:
        owner = lock_owner(root)
        lines = [f"작업 트리 {root} (브랜치 {branch(root)})",
                 f"잠금 주인: {owner.get('role', '역할 미기재')} 세션 {owner['comm']} PID {owner['pid']} ({owner['claimed_at']})" if owner else "잠금 주인: 없음",
                 f"이 세션: {_comm(agent_pid() or 0) or '사람(에이전트 아님)'} PID {agent_pid()}"]
        for name, cfg in TRACKS.items():
            for key in ("chat", "state"):
                p = root / cfg[key]
                if p.exists():
                    lines.append(f"{name} {cfg[key]} 중복 번호: {duplicates(p.read_text()) or '없음'}")
        ok, msg = True, "\n".join(lines)
    print(msg, file=sys.stdout if ok else sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
