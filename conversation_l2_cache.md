# Conversation L2 Cache

Purpose: compressed user-request log for fast session restore. Not full transcript. Store intent, constraints, outputs, next step.

Format:

| Date | Req | Constraints | Action/Output | Next |
| :--- | :--- | :--- | :--- | :--- |
| 2026-06-08 | Build realtime text-data env. News/report/SNS -> independent vars for quant analysis. Add analysis code. | Follow AGENTS. Use DuckDB/uv. Text as multivariate context. | Added `text_context.py`, `ingest_text_context.py`; added `text_events_raw`, `text_features_15m`; integrated `text_risk_guard` in `simulate_and_send.py`; README/process/history updated; committed `cb847a3`, pushed `origin/stock`, emailed professor. | Refresh Upbit candle mart so 2026-06-08 text timestamps overlap prices; run text-aware no-email backtest. |
| 2026-06-08 | Add L2-cache-like conversation request log. Not full contents. Like `/caveman`: compressed request summary. | Must avoid storing full raw chat. Root project file preferred. Update handoff docs. | Created `conversation_l2_cache.md` with compact table schema and current key requests. | Append one compact row after meaningful tasks; keep raw transcript out. |
| 2026-06-08 | Clarify execution actor boundary. Repo keeps automation; Codex may run light local checks but not heavy research runs. | Automation/CLI/n8n/CI/server commands remain. Forbidden: Codex local/venv execution of long-running research pipelines, training, notebook result runs, backtests unless explicitly approved. Allowed: small local calc, compile/import checks, small tests. Research notebooks need mirrored `.py`. | Updated boundary docs in `AGENTS.md`/`skills.md` language. Allowed for Codex: code, code review, light checks, docs, `.ipynb` + mirrored `.py`. Allowed for repo: runnable automation examples. | Preserve automation code; for heavy research execution, prepare code/review/docs and tell user to run on school server kernel. |
| 2026-06-08 | Korean email body mojibake. Resend readable Korean explanation. | Do not run analysis. Fix mail encoding. | Added `test/scripts/send_text_context_update_email_utf8.py`; uses UTF-8 plain text with base64 transfer encoding; resent to `nhson@ms.kmu.ac.kr`. | Use UTF-8 file-based mail script for Korean email; avoid PowerShell inline Korean here-string for SMTP body. |

## Maintenance Rule

- Append one row when user asks new meaningful task or direction changes.
- Keep `Req` short: what user wanted, not exact full text.
- Keep `Constraints` short: important style/tool/safety requirements.
- Keep `Action/Output` concrete: files, commits, links, sent mail.
- Keep `Next` actionable: next checkpoint only.
- Execution actor boundary: keep runnable automation in repo. Codex may run light local checks. Codex must not run local/venv long-running research analysis/training/backtest/notebook-result pipelines unless explicitly authorized. Research `.ipynb` changes need mirrored `.py`.
