# 17번 계획서 — 전면 EDA: 데이터 특성·상관·인과 구조를 모델링 전에 먼저 본다

작성일: 2026-07-19 · 브랜치: `stock` · 근거: `test/results/17_direction_predictability_
literature_review_20260719.md`, `governance_drift_rootcause_20260719.md`

> 사용자 지적: "데이터 전처리도 좀 수행해야 하는 거 아니야? 단순히 스케일링만 하는 게 아니라
> 파생변수 생성(이미 했지?) 데이터 분석도 좀 수행해서 애초에 데이터 자체의 특성과 상관, 인과
> 등을 시각화나 분석으로 먼저 파악하고 수행해야 하는 것 같다." — 8~16번 내내 파생변수는
> 만들었지만(`engine/features.py`, 60여개) 그 변수들과 target의 관계를 직접 분석한 적이 없다.

## 1. 왜 지금 이 실험인가

16번까지 확인된 사실: 순수 가격 15분봉에서 방향 신호(Pearson≤0.07, R²≈0.5%)는 문헌상 정상
범위이자 사실상 한계다. 다음 스텝(외생변수 도입 여부)을 **추측이 아니라 데이터로** 결정하려면
"가격 파생변수 중 정말 방향 신호를 가진 게 하나도 없는지"를 먼저 전수 확인해야 한다.

## 2. 연구 질문

- Q1: 각 파생변수(60여개)와 target(h-step 방향/수익률) 간 상관·상호정보량(MI)이 얼마나 되나?
  선형(Pearson)으로 안 보이는 비선형 관계(MI)가 있나?
- Q2: 변수 간 다중공선성이 심한가? (모델이 중복 정보를 여러 채널로 받아 오히려 방해될 수 있음)
- Q3: 변수·target의 분포·꼬리·이상치 구조는 어떤가? (모델 실패 원인의 다른 단서)
- Q4: 변동성 regime(고변동/저변동 국면)별로 신호가 다른가? (평균으로 뭉개진 약한 신호가
  특정 regime에서는 강할 수 있음)
- Q5: lead-lag — 어떤 변수가 미래 수익률/방향을 "선행"하는 조짐이 있나? (인과 방향의 단서,
  단 인과 증명은 아니고 예비 탐색)

## 3. 데이터·범위

- 원천: `upbit_krw_candle`(KRW 전 종목, 269종목) + 단일 종목 진단용 `btc_15m_advance`.
- 1차는 BTC로 전체 분석 틀을 검증하고, 2차는 유동성 상위 10~20종목으로 확장해 결론이
  BTC 특정인지 일반적인지 본다(known_pitfalls P3).
- target: h=16(4시간) 방향(부호)과 수익률 크기 둘 다. 15~16번과 동일 정의로 비교 가능하게.

## 4. 분석 축 (suite)

| suite | 내용 |
|---|---|
| t1_stationarity | 전 파생변수에 ADF/KPSS, 결과를 표+히트맵으로 |
| t2_correlation | 변수×target Pearson + Spearman(비선형 단조관계) + 상호정보량(MI, 비선형 전반) |
| t3_multicollinearity | 변수 간 상관행렬 + VIF(분산팽창지수) |
| t4_distribution | 분포(히스토그램/QQ), 첨도·왜도, 이상치(IQR/z-score 비율) |
| t5_regime | 변동성 상위/하위 25% 구간을 나눠 t2 상관을 재계산 — regime별 신호 차이 |
| t6_leadlag | 변수를 -h~+h만큼 시프트해 target과의 상관 곡선(선행/동행/후행 판정) |

## 5. 산출물

- 드라이버: `test/models/17_eda_direction_signal_test.py`(`.py`만, `#%%`+마크다운, CLAUDE.md 2.3)
- 결과: `test/results/17_eda_direction_signal_20260719/<suite>_raw.md` + csv
- 이미지: `test/images/17_eda_direction_signal_20260719/<suite>/`
- 종합 보고서: `test/results/17_eda_direction_signal_report_20260719.md`

## 6. 성공 기준 / 다음 반영

- 방향 신호(|MI| 또는 |상관|)가 유의하게 큰 변수가 하나라도 나오면 → 그 변수를 16번 손실·
  cross-sectional 실험에 우선 반영해 재검증.
- 전 변수가 여전히 무신호로 확인되면 → "가격 파생변수만으로는 방향 신호 없음"이 최종 확정되고,
  다음은 외생변수(텍스트·cross-asset·과거 유사국면) 도입으로 넘어간다(process.md 순서 확정).
- MDD 단독 랭킹 금지·진폭 게이트 등 known_pitfalls는 EDA에는 해당 없음(정책 평가 아님).
