"""스테이징된 Jupyter 노트북을 같은 이름의 Python 미러로 동기화한다.

이 도우미는 연구용 스크립트가 아니라 저장소 워크플로우 플러밍이므로
`test/scripts` 밖에 둔다. 노트북 JSON을 파싱해 실행 없이 마크다운/코드 셀을
추출한다.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


HEADER = [
    "# [FOR COMMIT TRACKING ONLY - DO NOT EXECUTE]",
    "# This file is automatically mirrored from the corresponding .ipynb for git diff purposes.",
    "# Actual research execution should be performed in the Jupyter Notebook (.ipynb)",
    "# or in an approved remote/server environment.",
    "",
]


def run_git(args: list[str], cwd: Path) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True, encoding="utf-8").strip()


def repo_root() -> Path:
    return Path(run_git(["rev-parse", "--show-toplevel"], Path.cwd()))


def staged_notebooks(root: Path) -> list[Path]:
    output = run_git(["diff", "--cached", "--name-only", "--diff-filter=ACMRT"], root)
    notebooks = []
    for line in output.splitlines():
        path = Path(line)
        if path.suffix == ".ipynb" and ".ipynb_checkpoints" not in path.parts:
            notebooks.append(path)
    return notebooks


def cell_source(cell: dict) -> str:
    source = cell.get("source", "")
    if isinstance(source, list):
        return "".join(source)
    return str(source)


def markdown_to_python_comments(text: str) -> list[str]:
    lines = ["# %% [markdown]"]
    if not text:
        return lines + ["#"]
    for line in text.splitlines():
        lines.append("# " + line if line else "#")
    return lines


def code_to_python_cell(text: str) -> list[str]:
    lines = ["# %%"]
    if text:
        lines.extend(text.rstrip().splitlines())
    return lines


def notebook_to_python(notebook_path: Path) -> str:
    with notebook_path.open("r", encoding="utf-8") as handle:
        notebook = json.load(handle)

    output_lines = list(HEADER)
    for cell in notebook.get("cells", []):
        cell_type = cell.get("cell_type")
        source = cell_source(cell)
        if cell_type == "markdown":
            output_lines.extend(markdown_to_python_comments(source))
        elif cell_type == "code":
            output_lines.extend(code_to_python_cell(source))
        else:
            continue
        output_lines.append("")
    return "\n".join(output_lines).rstrip() + "\n"


def sync_notebook(root: Path, notebook_rel: Path) -> Path:
    notebook_abs = root / notebook_rel
    mirror_abs = notebook_abs.with_suffix(".py")
    mirror_abs.write_text(notebook_to_python(notebook_abs), encoding="utf-8")
    run_git(["add", str(mirror_abs.relative_to(root)).replace("\\", "/")], root)
    return mirror_abs


def main() -> int:
    root = repo_root()
    notebooks = staged_notebooks(root)
    if not notebooks:
        return 0

    synced = []
    for notebook_rel in notebooks:
        mirror_abs = sync_notebook(root, notebook_rel)
        synced.append(mirror_abs.relative_to(root))

    print("[ipynb-mirror] Python 미러를 동기화했습니다:")
    for path in synced:
        print(f"  - {path.as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
