# 알려진 함정 (실행 코드 짜기 직전 이 파일만 대조 — 20줄 이내 유지)

> 원칙: 반복 재발한 함정만 1줄씩. 새 함정은 여기 1줄 추가(다른 문서에 문단 추가 금지).
> 근거·사례는 `test/results/governance_drift_rootcause_20260719.md`. 상위 규칙은 CLAUDE.md 2.12.

| # | 함정 | 탐지 신호 | 자동 대응(코드 게이트) |
|---|---|---|---|
| P1 | huber 등 평균형 손실이 큰 변동을 못 잡음(진폭 압축) | variance_ratio ≪ 1 | 챔피언 선정에서 자동 탈락. 큰변동은 pinball/student_t/tail_weighted 병행 |
| P2 | MDD(또는 수익률) 단독 랭킹 → 평탄한 예측이 방어 1등 둔갑 | active_share↓·variance_ratio↓인데 MDD 좋음 | MDD 단독 정렬 금지. variance_ratio·large_move_da·활동하한 병기 |
| P3 | BTC 단일 축을 전 종목 결론으로 일반화 | table=btc_15m_advance | 결론은 "단일 축 한정" 표기. 일반화는 upbit_krw_candle(전 종목)로 |
| P4 | 국소 회귀: 요청의 부분만 최적화 | 단일 지표 최고로 우승 선정 | 실행 전 "방향 일치성 3문" 통과(목적 3요소 대조) |
| P5 | 중간 상세 보고 남발 / 이슈가 파일에 안 남음 | 대화창에만 진단 기록 | 중간=진척만, 상세는 최종 1회 + 결과 raw md에 이슈 전량 기록 |
| P6 | 실험 번호 재사용·재정렬 | 삭제된 번호를 당겨씀 | 번호는 불변 식별자. 결번 유지, 최대번호+1만 |
| P7 | 메일에 `.md` 원본 첨부(raw 텍스트로 열려 이미지·표 깨짐) | attachments에 `*.md` 포함 | 첨부 금지, GitHub 렌더링 링크만. 그림은 inline cid 또는 PNG 첨부 |
