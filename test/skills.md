# Test Skills Guide

이 파일은 `test/` 아래 실험을 할 때의 작업 기준을 간단히 정리한 메모다.  
실제 운영 규칙은 루트 `AGENTS.md`를 따른다.

## 핵심 흐름
- 분석 시작 전 `history.md`와 `process.md`를 확인한다.
- 논문 조사는 `arxiv-mcp-server` 우선으로 진행한다.
- Hugging Face 관련 보조 도구가 필요하면 `.agents/skills/`를 확인한다.
- 결과 메모는 `test/research_materials/`에 남긴다.
- 리포트는 `test/results/`에 저장한다.

## 검증 기준
- 숫자는 가능한 한 원본 스케일로 확인한다.
- 시계열 평가는 `DA`와 `MASE`를 포함한다.
- 그래프 설명에는 축과 범례를 함께 적는다.
- 버그 수정은 원인 중심으로 기록한다.

## 정리 기준
- 임시 파일은 만들지 않거나 바로 지운다.
- 남은 임시 산출물은 `test/.gitignore` 규칙으로 걸러낸다.
- 노트북과 Python 미러 파일은 같이 유지한다.
