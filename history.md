# 작업 이력 (History, 서비스 트랙)

> **서비스 트랙 기록 (2026-10-15 신설)**: 서비스 구현(루트 프레임워크: `marts/`·`contexts/`·`agents/`·`advisors/`·`pipelines/`·`engine/`)과 저장소 공통 규칙 변경의 이력이다. 연구 트랙 기록과 2026-10-15 이전 이력은 `test/` 아래 같은 이름 파일에 있다.

| 날짜 | 단계 | 수행 내용 | 산출물 | 다음 목표 | 수행환경 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 2026-10-15 | 기록 트랙 분리·도구 동일 수행 | 연구(stock)와 서비스 구현(llm-agent) 세션을 동시에 돌리면 루트 기록 4종이 병합 충돌하므로, 연구 기록을 `test/`로 옮기고(git mv, 이력 추적 가능) 루트는 서비스 트랙 기록으로 새로 시작. 요청 인덱스는 두 트랙 공유. Codex가 Claude Code와 같은 규칙을 보도록 Claude 사설 memory 16건과 fluent-korean 문장 지침을 `AGENTS.md` §8로 이관하고 2.1·§3의 Codex 전용 실행 제한을 도구 공통으로 바꿈. develop/main 차단 목록에 `test/` 기록 4종 추가. | `AGENTS.md`, 루트·`test/` 기록 4종, `.githooks/check_repo_policy.py`, README·test/README·team-spec·methodology_catalog 경로 | 서비스 세션: llm-agent worktree 생성 후 MVP 1단계 | 서버 |
