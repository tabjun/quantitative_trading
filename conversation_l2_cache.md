# Conversation L2 Cache

Purpose: compressed user-request log for fast session restore. Not full transcript. Store intent, constraints, outputs, next step.

Format:

| Date | Req | Constraints | Action/Output | Next |
| :--- | :--- | :--- | :--- | :--- |
| 2026-06-08 | Build realtime text-data env. News/report/SNS -> independent vars for quant analysis. Add analysis code. | Follow AGENTS. Use DuckDB/uv. Text as multivariate context. | Added `text_context.py`, `ingest_text_context.py`; added `text_events_raw`, `text_features_15m`; integrated `text_risk_guard` in `simulate_and_send.py`; README/process/history updated; committed `cb847a3`, pushed `origin/stock`, emailed professor. | Refresh Upbit candle mart so 2026-06-08 text timestamps overlap prices; run text-aware no-email backtest. |
| 2026-06-08 | Add L2-cache-like conversation request log. Not full contents. Like `/caveman`: compressed request summary. | Must avoid storing full raw chat. Root project file preferred. Update handoff docs. | Created `conversation_l2_cache.md` with compact table schema and current key requests. | Append one compact row after meaningful tasks; keep raw transcript out. |

## Maintenance Rule

- Append one row when user asks new meaningful task or direction changes.
- Keep `Req` short: what user wanted, not exact full text.
- Keep `Constraints` short: important style/tool/safety requirements.
- Keep `Action/Output` concrete: files, commits, links, sent mail.
- Keep `Next` actionable: next checkpoint only.
