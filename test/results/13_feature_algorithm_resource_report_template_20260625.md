# 13번 feature·algorithm·resource 결과 보고서 템플릿

## 결론 요약

| 순위 | 변수셋 의미 | 모델/전처리/risk gate | 핵심 수치 | 최종 판단 |
|---|---|---|---|---|
| 1 | TBD | TBD | fusion MDD, fusion return, copy-risk, AP lift | TBD |
| 2 | TBD | TBD | TBD | TBD |
| 3 | TBD | TBD | TBD | TBD |

## 실험 목적

12번에서 가장 좋았던 multi-timeframe 구조가 특정 모델·seed의 우연인지, 아니면 변수셋 자체가 여러 알고리즘과 risk gate 조건에서도 버티는지 확인한다.

## 데이터와 변수

### RAW 변수

| RAW 변수 | 의미 | 사용 방식 |
|---|---|---|
| timestamp | 15분 봉 시각 | 시간 정렬, 시간대 파생변수, 시간 기반 split |
| open/high/low/close | 15분 봉 가격 | 수익률, range, candle body/wick, proxy candle 평가 |
| volume | 15분 거래량 | 거래량 z-score, shock, signed volume |
| value | 15분 거래대금 | 거래대금 z-score, liquidity, attention proxy |
| optional mart columns | text/macro/on-chain/derivatives/social | 컬럼이 있을 때만 optional group으로 자동 결합 |

### 파생 변수군

| 변수군 | 예시 | 생성 방식 |
|---|---|---|
| multi-timeframe return | 15분, 1시간, 4시간, 16시간, 2일 수익률 | log return rolling sum |
| volatility | 4시간, 16시간, 2일 변동성 | return rolling std |
| trend/position | price z-score, trend strength | rolling z-score, rolling mean 대비 괴리 |
| volume/range | volume/value z-score, range mean | log volume/value와 high-low range 기반 |
| order-flow proxy | candle body, wick, signed volume/value | 실제 호가창이 없을 때 캔들 구조로 압력 근사 |
| risk/attention proxy | abs return shock, volume shock, attention pressure | 급변·거래 폭증·관심도 대리 지표 |

## Case 수

실행 노트북의 `[plan] executable_cases`와 `case-count-by-feature`, `case-count-by-model` 출력을 여기에 옮긴다.

## 지표 해석

- `fusion MDD`: risk gate와 점예측을 결합한 정책의 최대 낙폭이다. 0에 가까울수록 하방 방어가 좋다.
- `fusion return after cost`: 수수료와 슬리피지 proxy를 뺀 누적 수익률이다. 양수면 비용 차감 후에도 이익을 낸 것이다.
- `copy-risk ratio`: 모델 MAE를 직전가 유지 기준 MAE로 나눈 값이다. 1보다 낮으면 persistence보다 나은 점예측이다.
- `AP lift`: 위험 이벤트 확률 예측의 average precision을 단순 event rate로 나눈 값이다. 1보다 높으면 무작위/기준선보다 이벤트 선별력이 있다.
- `low_trade_warning`: 거래 횟수나 active share가 너무 낮아 MDD만 좋아 보이는 착시 가능성을 표시한다.

## 대표 그래프

실행 완료 후 `test/scripts/extract_notebook_images.py`로 원본 노트북을 변경하지 않고 이미지를 추출한다.

각 그림 설명은 다음 순서를 따른다.

1. 데이터/대상 모델
2. x축과 y축
3. 진단 목적
4. 실제 관찰
5. 좋은지 나쁜지
6. 다음 실험에 반영할 결정

## 전체 결과표

실행 완료 후 top 30 leaderboard와 전체 case 표를 붙인다. 단 하나의 case도 누락하지 않는다.

## 다음 단계

- 최종 후보 feature group을 데이터마트 정식 스키마로 승격할지 결정한다.
- optional mart 컬럼이 빠져 제외된 group은 실제 컬럼 overlap 확보 후 재검증한다.
- seed 안정성이 부족한 후보는 모델 확장보다 변수 정의를 먼저 재검토한다.
