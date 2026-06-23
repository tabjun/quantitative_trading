# Research Workflow Team Spec

## Purpose

Keep recurring research work deterministic in this repository: notebook-first analysis, mirrored Python files, short directory-level README, result reports, and history logging.

## Scope

This spec applies when a task repeats across sessions:

- research notebooks under `test/models/`
- report generation under `test/results/`
- document hygiene for `README.md`, `history.md`, `process.md`
- mirror sync between `.ipynb` and `.py`
- mail/report handoff or GitHub link preparation

## Role Topology

- Main agent: implements the change, updates the right artifact, and keeps the workflow narrow.
- No worker delegation by default.
- Add a reviewer step only when the change affects report quality, doc boundaries, or release links.

## Canonical Artifact Ownership

- `AGENTS.md`: repo-wide always-loaded rules only
- `process.md`: long-horizon roadmap and phase map
- `history.md`: completed work log and next step
- `conversation_l2_cache.md`: compressed request memory
- `test/README.md`: `test/` directory overview only
- `test/models/*.ipynb`: experiment source of truth
- `test/models/*.py`: diff-traceable mirror of the notebook
- `test/results/*.md`: experiment write-up and interpretation
- `test/experiment_specs/*.md`: design notes and rationale

## Working Rules

1. Read the always-loaded docs before editing.
2. Keep `README` short and pointer-heavy.
3. Put experiment narrative in the notebook and the result report, not in `README`.
4. Update `history.md` after meaningful completion or direction change.
5. Keep `.ipynb` and same-name `.py` together.
6. Avoid local heavy research runs; use the approved remote kernel for actual results.
7. Reuse existing scripts and docs before creating new ones.
8. If a follow-on study changes the meaning or scope of an existing numbered experiment, create a new numbered artifact set instead of overwriting the old one.
9. Write each new report as a standalone document; repeat core methodology and interpretation guidance even when earlier reports already explained it.
10. For notebook-driven experiments, prefer `plt.show()` and inline cell output. Do not call `savefig()` by default, and do not treat server-side PNG/CSV/Markdown files as the primary result unless the user explicitly asks for file artifacts.
11. Make terminology independently understandable in reports and email. On first use, explain each decision-critical term in the order `plain-language definition -> concrete example -> role in this experiment -> whether the observed value is favorable, unfavorable, or ambiguous`.
12. Never leave relative statements such as “preserved variance” or “close to baseline” unexplained. Name the comparison baseline, the direction of change, and why the result is necessary but not sufficient for model quality.
13. For any `/test` request, first evaluate whether it serves the current research question. If it does not, rewrite it into a criticism-driven or methodology-improving version before implementing.
14. When a request proposes adding a new method, objective, preprocessing path, or visualization, adversarially check whether it strengthens the research path, duplicates an existing axis, or muddies the experiment lineage.
15. A completed-notebook report automatically requires an image-output pass. Enumerate the notebook's `image/png` outputs, reuse `test/scripts/extract_notebook_images.py`, select every decision-relevant visual, and embed it without waiting for the user to repeat the request.
16. Every embedded research visual must be followed or preceded by a standalone explanation of the data/model, x/y axes, diagnostic purpose, observed shape, favorable/unfavorable interpretation, and concrete downstream decision.

## Handoff Contract

- If the work is a doc-only change, preserve the repo structure and update the smallest number of files.
- If the work changes an experiment, update notebook, mirror, and report together.
- If the work changes repository policy, update `AGENTS.md` and `history.md`.

## Failure Policy

- If a document starts absorbing another document's role, move the text out instead of expanding the overlap.
- If a task needs recurring structure but no shared spec exists, create or extend this team spec first.
- If the right artifact is unclear, prefer the notebook/report for experiment content and `history.md` for completion logging.

## Success Check

- `test/README.md` stays an overview.
- `history.md` records completion.
- experiment details live in `test/results/*.md` or the notebook.
- notebook and mirror stay synchronized.
- no one-off workflow instructions are hidden in random files.
