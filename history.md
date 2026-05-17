# 연구 수행 이력 (Research History / State Persistence)

이 파일은 AI 에이전트(Claude Code, Codex)가 새로운 세션으로 넘어갈 때 **'기억 상실'을 방지하기 위한 영구 저장소(Memory)**입니다.
에이전트는 작업을 마무리하기 전, 반드시 이 파일의 테이블과 요약 섹션을 업데이트해야 합니다.

## 📝 History Log

| Date | Phase / Step | Task Details (수행 내용) | Output & Result (산출물) | Next Action (다음 목표) |
| :--- | :--- | :--- | :--- | :--- |
| 2026-05-16 | Phase 0 | 프로젝트 기획안 분석 및 다변량 분석 구조 설계, 아키텍처 다이어그램 업데이트 | `README.md`, `process.md`, `skills.md`, `history.md` 재구성 완료 | Phase 1 (Data Mart 구축) 진입 대기 |
| | | | | |
| | | | | |
| | | | | |

---

## 🔄 세션 인계를 위한 핵심 요약 (Context for New Session)

새로운 AI 세션이 시작되면 이 섹션을 읽고 즉각적으로 현재 프로젝트의 스탠스와 목표를 파악하십시오.

- **프로젝트 핵심 기조:** 단일 시계열 분석을 배제하고 뉴스/매크로가 결합된 **다변량 분석(Multivariate)**을 수행. 매매 전략의 제1원칙은 **"최하방 방어(MDD 최소화)"**.
- **가장 최근에 완료된 작업:** 기반 마크다운 문서화 작업 및 시스템 아키텍처(Mermaid) 설계가 완료됨.
- **지금 당장 시작해야 할 작업 (Next Step):** `process.md`의 **Phase 1, Step 1.1** (MTS/Upbit 등에서 과거 데이터를 받아와 자체 Data Mart를 구축하기 위한 수집기 연동 로직 파악 및 코딩).
- **에이전트 주의사항:** 코드를 작성하기 전에 `/materials` 폴더 내의 배경 지식을 `npx -y @chrisryugj/kordoc`을 활용하여 필요 시 다시 확인하고, `uv` 환경에 맞춰 종속성을 관리할 것.

---
*(AI 에이전트에게: 작업을 종료할 때 이 하단 요약 내용과 위 테이블을 수정하여 다음 에이전트가 완벽히 이어받을 수 있게 하십시오.)*
