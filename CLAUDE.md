# CLAUDE.md - Claude Code 작업 지침서

**규칙 본문은 [`AGENTS.md`](AGENTS.md)를 source of truth로 본다.** 이 파일은 Claude Code가
세션 시작 시 자동으로 읽는 파일명(`CLAUDE.md`)만 맞추기 위한 참조 스텁이다 — Codex가 읽는
`AGENTS.md`와 규칙을 물리적으로 두 곳에 중복 관리하지 않기 위해 2026-08-09부터 이 방식으로
합쳤다(참고: https://yozm.wishket.com/magazine/detail/3874/ 의 `AGENTS.md` 단일화 패턴).

새 세션이 시작되면 이 파일 대신 `AGENTS.md` → `process.md` → `history.md` 최근 섹션 →
`conversation_l2_cache.md` 최근 항목 → `test/known_pitfalls.md` → `test/README.md` 순으로 읽는다.

Claude Code 전용 차이점(Codex와 다른 점)은 다음 두 가지뿐이며, 나머지는 전부 `AGENTS.md`를 따른다.

- MCP 설정 파일: Codex는 `.codex/config.toml`, Claude Code는 `.mcp.json`.
- 도구 전환 체크리스트에서 "이 세션"이 가리키는 주체가 Codex 대신 Claude Code다.

규칙을 고치고 싶으면 `AGENTS.md`를 고친다. 이 파일에 규칙을 직접 추가하지 않는다.
