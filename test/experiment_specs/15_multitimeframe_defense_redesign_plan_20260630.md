# 15번 계획서 — 13번 재설계: multi-timeframe 방어 변수셋 + 비폭주 point + risk-gate 중심

작성일: 2026-06-30 · 브랜치: `stock` · 토대: 공용 엔진 `engine/`(12 fusion 결함 ①②③ 교정본)

> 13번 노트북은 저장 실패·crash 상태이고 옛 12번을 `load_module`로 끌어오는 결함 경로를 쓴다.
> 완료/동결 노트북 read-only 규칙과 "후속 연구는 새 번호" 규칙에 따라, 13번을 직접 고치지 않고
> **교정 엔진 기반 새 실험 15번**으로 재설계한다. 13번의 연구 의도(변수·알고리즘 공동 최적화)는
> 14번 교정 재실행 결과(A/B)를 반영해 아래처럼 좁힌다.

---

## 1. 왜 재설계인가 (14번 결과 반영)

14번 교정 재실행(`test/results/14_fusion_alignment_rerun_report_20260630.md`)에서:

- **A**: 정렬·활동하한 교정 후에도 `coin_multitimeframe_structure`가 fusion MDD 1위(-0.0597, 2·3위와
  격차 큼). → multi-timeframe을 1순위로 깊게 판다.
- **B**: 점예측 폭주는 `Linear` 고유(variance_ratio ≈4,722). PatchTSTLike는 ≈358로 ~13배 안정. 단
  두 모델 다 persistence 미달. → **Linear 배제, PatchTST 계열 채택. 점예측은 약한 필터로만.**
- **방어 근거**: 모든 케이스에서 risk gate가 MDD를 줄였다. → **risk gate·MDD 방어가 연구 무게중심.**

따라서 15번의 질문은 "더 정확한 점예측"이 아니라 **"어떤 multi-timeframe 하위 구조 + 어떤 비폭주
모델 + 어떤 gate 설정이 MDD를 가장 잘 줄이는가"**이다.

## 2. 실험 축 (좁힌 격자)

1. **multi-timeframe 내부 분해 / 확장 (핵심)**
   - 분해: `coin_multitimeframe_structure`를 하위 블록으로 쪼개 어느 블록이 방어를 끄는지 본다 —
     `mtf_returns`(다중 horizon 수익률), `mtf_volatility`(실현변동성·비율), `mtf_trend`(추세·위치),
     `mtf_volume_range`(거래량·range z-score). (engine.features 기준 컬럼)
   - 확장: `mtf + momentum_reversal`, `mtf + shock_event`, `mtf + volatility_regime` 조합도 비교.
2. **point 모델 (비폭주만)**: `PatchTSTLike`, `ModernTCNLike`, `TCN`, `NLinearLike`. **`Linear` 제외.**
   - 1차 게이트: variance_ratio가 일정 임계(예: <800) 이하인 모델만 통과(폭주 차단).
3. **risk gate 민감도**: `risk_allow_quantile` ∈ {0.45, 0.55, 0.65}로 gate 공격성 조절. horizon=16(4시간),
   `absolute_move` 유지.
4. **seed 안정성**: {42, 7, 123} — 결론이 seed 우연이 아닌지 확인.
5. **전처리**: `seasonal_diff16`, `winsor_025`(14번에서 부차적이었으므로 2종만 유지).

## 3. 평가·랭킹 규칙 (교정 엔진 그대로)

- 점예측·위험 신호는 **decision_timestamp inner-join**으로만 정렬(fix①). 겹침 0/중복이면 실패 처리.
- **활동 하한**(active_share ≥ 0.05, trade_count ≥ 5) 통과 케이스만 우승 후보(fix③). 비활동은 분리 표기.
- 1차 지표 = **fusion MDD**(point-only 대비 감소폭), 동일 노출(signal share) 조건에서 비교(Pareto).
- 보조 지표 = copy_risk(<persistence면 가산점), DA, risk gate의 AP/Brier-skill.
- 텍스트·cross-market·macro·on-chain·derivatives는 mart 컬럼이 실제 있을 때만 등록(fix②).

## 4. 자원·안정성 (13번 crash 회피)

13번은 shared memory 부족(`num_workers`↑) + notebook backup 실패(거대 inline output)로 끊겼다. 15번은:

- 헤드리스 `.py` 드라이버로 실행(14번 패턴), per-case 그래프 off, 결과는 leaderboard 텍스트 + 보고서용
  대표 그림만 후처리 추출.
- `num_workers` 낮게, batch 적정(256 내외, OOM 시 자동 절반 재시도는 엔진에 내장).
- **단계 실행**: (a) mtf 분해 스크리닝(작게) → (b) 모델 폭주 게이트 → (c) 확장·gate·seed 확정.
- 격자 크기를 단계마다 로그로 남겨 silent truncation 금지.

## 5. 산출물

- 실험: `test/models/15_multitimeframe_defense_test.{ipynb,py}` (engine import, 14번 드라이버 확장)
- 결과 보고서: `test/results/15_*_report_YYYYMMDD.md` (독립 문서, 14번과 같은 설명 기준 재기술)
- 보고서용 그림: `test/images/15_*/`

## 6. 성공 기준

- multi-timeframe **어느 하위 블록**이 방어를 끄는지 특정(분해 결과).
- 비폭주 point 모델 중 fusion MDD가 가장 안정적으로 낮은 후보 확정(seed 가로질러).
- gate 공격성-노출-MDD의 트레이드오프 곡선 확보 → 데이터마트 정식 스키마 승격 변수셋·모델·gate 설정 1세트 확정.
