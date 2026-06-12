---
name: research-workflow-orchestrator
description: Orchestrate recurring notebook-first research work in quantitative_trading with strict artifact boundaries, mirror sync, and history logging.
---

# Research Workflow Orchestrator

Use this skill for recurring research tasks in this repository where the same workflow repeats across sessions.

## When to Use

- notebook-first experiment work under `test/models/`
- report or mail preparation tied to a research pass
- document cleanup where role boundaries are at risk
- mirror sync and history logging after a research change

Do not use this skill for a one-off edit that does not need a repeatable workflow.

## Required Inputs

- current task or research question
- target notebook/report files
- branch name
- whether the change is doc-only or experiment-related

## Workflow

1. Read `AGENTS.md`, `process.md`, `history.md`, `conversation_l2_cache.md`, and `test/README.md`.
2. Check `docs/harness/research-workflow/team-spec.md` for artifact ownership.
3. Decide the smallest valid artifact set.
4. Keep experiment content in the notebook and matching result report.
5. Keep `test/README.md` short and structural.
6. Update `history.md` when the task is complete or when the direction changes.
7. Preserve `.ipynb` and same-name `.py` mirrors.
8. Avoid local heavy research runs unless the user explicitly wants a light verification only.

## Output Contract

- `test/README.md` stays an overview.
- `history.md` gets a short completion record.
- `test/results/*.md` carries interpretation and conclusions.
- `test/models/*.ipynb` and `.py` stay synchronized.

## Validation Notes

- No duplicate long-form explanations across docs.
- No one-off scripts when an existing tool already fits.
- No heavy local training or analysis runs.
- No experiment narrative hidden in `README` when it belongs in a report.

## Failure Policy

- If the artifact boundary is unclear, default to notebook/report for experiment details and history for logging.
- If the task starts to recreate the same workflow repeatedly, extend `docs/harness/research-workflow/team-spec.md` instead of scattering new rules.
