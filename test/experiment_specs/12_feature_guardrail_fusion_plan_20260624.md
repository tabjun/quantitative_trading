# 12번 feature guardrail fusion 실험 계획

## 1. 결론부터

12번은 알고리즘을 최고 수준으로 새로 설계하는 실험이 아니다. 10번과 11번에서 이미 확인한 결과를 토대로, 다음 최적화 단계에서 **어떤 코인 독립변수 조합을 데이터마트 후보로 올릴지** 고르는 실험이다.

역할은 다음처럼 고정한다.

| 번호 | 역할 | 12번에서의 사용 방식 |
|---|---|---|
| 10번 | `balanced_composite` objective 기반 점예측 | 다음 15분 수익률의 방향과 크기를 내는 기본 branch |
| 11번 | `absolute_move` 위험 확률 | 큰 변동 위험이 높은 구간에서 진입을 막는 guardrail |
| 12번 | feature group별 결합 검증 | 독립변수 조합이 collapse, persistence 미달, MDD를 얼마나 줄이는지 확인 |

따라서 12번의 핵심 질문은 다음이다.

> 10번의 점예측 branch가 전체 구간에서는 persistence를 넘지 못했더라도, 코인에 맞는 독립변수와 11번 위험 gate를 결합하면 collapse를 줄이고 MDD 방어에 쓸 수 있는가?

## 2. 왜 이 방향이 맞는가

10번 결과에서 `huber`는 loss를 안정적으로 낮췄지만 예측값이 0수익률 근처로 평평해지는 쉬운 해를 택할 위험이 컸다. 반면 `balanced_composite`는 방향성, 분산, 상관, tail, regime, anti-collapse 항을 같이 보면서 완전한 평탄화를 줄였다. 아직 persistence를 넘지는 못했지만, “최적화가 쉬운 해로 바로 무너지는 문제”는 이전보다 완화되었다.

11번 결과에서는 가격을 정확히 맞히는 대신 향후 4시간의 큰 움직임을 확률로 예측했다. `PatchTSTLike + absolute_move` 계열은 AP lift와 Brier skill에서 기준선을 넘었다. 즉 다음 가격을 바로 맞히는 것은 약하지만, 위험한 구간을 걸러내는 정보는 있었다.

따라서 지금 모델 구조를 다시 늘리는 것보다, 10번의 점예측을 유지하고 11번 위험 확률을 가드레일로 붙인 뒤, 어떤 독립변수 묶음이 가장 도움이 되는지 확인하는 편이 연구 계보에 맞다.

## 3. 독립변수 후보군

코인 예측 문헌과 실무 자료는 보통 다음 입력군을 함께 다룬다. 이번 수정에서는 논문뿐 아니라 YouTube/블로그/실무형 자료에서 반복적으로 강조되는 order flow, liquidity sweep, multi-timeframe, volume shock 관점도 feature group으로 분리했다.

- 기술지표와 OHLCV: 수익률, 이동수익률, EMA gap, RSI, MACD, range
- 유동성·거래대금: volume, value, turnover, range-volume interaction, illiquidity proxy
- 변동성·레짐: realized volatility, downside volatility, volatility ratio, drawdown
- order flow proxy: candle body, upper/lower wick, close location value, signed volume/value, volume pressure
- multi-timeframe structure: 15분, 1시간, 4시간, 16시간, 2일 근처 수익률·변동성·추세 구조
- shock/event: return shock, volume shock, value shock, range shock, jump-reversal
- attention proxy: 검색량·소셜 데이터가 없을 때 거래량/거래대금/range shock으로 만든 관심도 proxy
- 시간대 효과: 코인은 24시간 거래되므로 한국/미국 시간대와 요일 주기
- cross-market: ETH, XRP, SOL 등 주요 코인의 동조·선행 관계
- 텍스트·관심도: 뉴스, SNS, 검색, sentiment shock
- macro/on-chain/orderflow: 달러, VIX, 금리, 온체인 활동, 오더북 imbalance

현재 저장소에서 바로 쓸 수 있는 것은 OHLCV 기반 proxy, 선택적 `text_features_15m`, 선택적 다중 ticker cross return이다. on-chain, funding/open interest, order book, Google Trends, YouTube/SNS, GitHub activity는 아직 mart가 없을 수 있으므로 12번 코드에서는 컬럼이 있을 때만 자동 feature group으로 등록한다. 컬럼이 없으면 실험이 깨지지 않고 해당 group만 건너뛴다.

## 4. 12번 feature group

| feature group | 의미 | 12번에서 보는 것 |
|---|---|---|
| `ohlcv_core` | 기존 가격·거래량 기준선 | 독립변수 추가 전 최소 기준 |
| `coin_liquidity_micro` | 거래대금, turnover, range, Amihud-style illiquidity proxy | 코인 특유의 유동성 변화가 collapse를 줄이는지 |
| `coin_volatility_regime` | 변동성 비율, downside/upside vol, drawdown | 11번 risk gate와 잘 맞는지 |
| `coin_momentum_reversal` | RSI, MACD, EMA gap, trend/reversal | 10번 점예측 direction이 개선되는지 |
| `coin_orderflow_proxy` | candle body, wick, close location, signed volume/value | order book 없이도 단기 매수·매도 압력 proxy가 도움이 되는지 |
| `coin_multitimeframe_structure` | 4/16/64/192 window return, vol, z-score, trend | multi-timeframe 구조가 단일 15분 신호보다 안정적인지 |
| `coin_shock_event` | return/volume/value/range shock, tail, jump-reversal | 이벤트성 급변과 되돌림을 위험 gate가 잡는지 |
| `coin_attention_proxy` | volume/value/range shock 기반 관심도 proxy | 검색·소셜 데이터 전 단계에서 attention 신호가 유효한지 |
| `coin_calendar_cycle` | hour/day cycle, 한국/미국 시간대 proxy | 24시간 시장의 유동성 시간대가 도움이 되는지 |
| `coin_text_context` | 텍스트 mart가 있을 때 sentiment/shock/topic | 텍스트가 있으면 위험 gate에 보조 정보가 되는지 |
| `coin_cross_market` | 다중 ticker 테이블이 있을 때 주요 코인 return | 시장 전체 흐름이 BTC 예측에 도움이 되는지 |
| `coin_macro_proxy` | DXY/VIX/금리/주가지수/환율 등 컬럼이 있을 때 | 위험자산 레짐 설명력이 있는지 |
| `coin_onchain_proxy` | active address, exchange flow, SOPR, MVRV 등 컬럼이 있을 때 | 코인 고유 수급·네트워크 활동이 도움이 되는지 |
| `coin_derivatives_proxy` | funding, open interest, basis, liquidation 등 컬럼이 있을 때 | 레버리지·청산 압력 정보가 급변 예측에 도움이 되는지 |
| `coin_search_social_dev` | Google Trends, YouTube, Twitter/X, Reddit, GitHub 등 컬럼이 있을 때 | 관심도·커뮤니티·개발자 활동이 선행 정보인지 |
| `coin_full_available` | 현재 사용 가능한 모든 feature | feature 과다 결합이 오히려 노이즈를 키우는지 |

## 5. 실험 구조

12번은 각 case에서 두 branch를 모두 학습한다.

```text
feature group
  -> 10번 point branch: balanced_composite objective
  -> 11번 risk branch: absolute_move probability
  -> validation risk cutoff
  -> point-only / risk-only / point+risk gate 정책 비교
```

기본 모델은 다음으로 제한한다.

- point branch: `Linear`, `PatchTSTLike`
- risk branch: `PatchTSTLike`
- preprocessing: `seasonal_diff16`, `winsor_025`
- seeds: `42`, `2026`
- objective: `balanced_composite`
- risk event: `absolute_move`

기본 case 수는 사용 가능한 feature group 수에 따라 늘어난다. 현재 OHLCV만 있어도 기존보다 넓은 proxy group이 자동 생성되며, text/cross/macro/on-chain/derivatives/search-social-dev 컬럼이 있으면 추가 group이 더 붙는다. 각 case는 point model과 risk model을 모두 학습하므로 실제 학습 model 수는 그 두 배가 된다. 사용자는 12번만 실행할 예정이고 RTX 4090 서버 자원이 있으므로, 이번 단계는 넓게 돌려 feature 후보를 많이 탈락시키는 방향으로 간다.

## 6. 주요 지표

### 6.1 Point branch

- `copy_risk_ratio`: 모델 MAE / persistence MAE
- `direction_accuracy`: 실제 수익률 부호와 예측 부호가 같은 비율
- `variance_ratio`: 예측 수익률 분산 / 실제 수익률 분산

좋은 결과는 copy-risk가 1에 가까워지거나 1보다 작아지고, 방향 정확도가 50%보다 올라가며, variance ratio가 0에 붙지 않는 것이다.

### 6.2 Risk branch

- `average_precision`
- `average_precision_lift`
- `brier_skill_score`
- `expected_calibration_error`

좋은 결과는 AP lift가 1보다 크고, Brier skill이 0보다 크며, ECE가 너무 크지 않은 것이다.

### 6.3 Decision policy

- `point_only_mdd`
- `point_plus_risk_gate_mdd`
- `point_only_cumulative_return`
- `point_plus_risk_gate_cumulative_return`
- `fusion_signal_share`
- `trade_count`

좋은 결과는 risk gate를 붙였을 때 MDD가 줄면서도 거래가 거의 0으로 사라지지 않는 것이다. 거래를 거의 하지 않아서 MDD만 좋아지는 결과는 성공으로 보지 않는다.

## 7. 그래프 읽는 법

12번은 case별로 다음 그래프를 inline 출력한다.

1. Point forecast graph
   - x축: test time index
   - y축: 다음 15분 로그수익률
   - 목적: 10번 branch가 실제 수익률을 따라가는지, 0 근처로 납작해지는지 확인

2. Risk guardrail graph
   - x축: test time index
   - y축: 향후 4시간 absolute-move 위험 확률
   - 목적: 11번 branch가 위험 구간에서 확률을 높이는지 확인

3. Decision mask graph
   - x축: test time index
   - y축: position 여부
   - 목적: point-only 진입과 risk-gated 진입이 어떻게 달라지는지 확인

4. Policy cumulative return proxy
   - x축: test time index
   - y축: 거래비용 반영 누적수익률 proxy
   - 목적: point-only, risk-only, point+risk gate 정책을 비교

5. Risk probability vs realized movement
   - x축: 예측 risk probability
   - y축: 실제 절대수익률
   - 목적: risk probability가 실제 움직임과 단조롭게 관련되는지 확인

6. MDD by policy
   - x축: MDD
   - y축: 정책 이름
   - 목적: 하방 방어 개선 여부를 직접 비교

## 8. 실행 명령

서버 Jupyter 노트북에서는 기본 셀을 그대로 실행한다.

```python
%run test/models/12_feature_guardrail_fusion_test.py --suite feature_guardrail_matrix
```

터미널 소형 점검은 다음처럼 한다.

```bash
python test/models/12_feature_guardrail_fusion_test.py \
  --suite feature_ablation_quick \
  --epochs 3 \
  --max-windows 1024 \
  --max-cases 12 \
  --continue-on-failure
```

본실험은 다음처럼 한다.

```bash
python test/models/12_feature_guardrail_fusion_test.py \
  --suite feature_guardrail_matrix \
  --epochs 12 \
  --max-windows 4096 \
  --max-cases 0 \
  --continue-on-failure
```

## 9. 성공 판단

12번이 성공하려면 다음 중 최소 하나는 보여야 한다.

1. 특정 feature group에서 point copy-risk가 10번보다 낮아진다.
2. 특정 feature group에서 direction accuracy가 안정적으로 50%를 넘고 seed 간 순위가 유지된다.
3. risk gate 결합 후 point-only보다 MDD가 줄어든다.
4. MDD 개선이 거래 회피 착시가 아니라 충분한 signal share와 함께 나타난다.
5. volatility/liquidity/momentum/text/cross-market 중 데이터마트로 승격할 feature group 우선순위가 나온다.

이 조건을 만족하면 다음 단계는 데이터마트 정식 구축이다. 만족하지 못하면 모델 구조가 아니라 target horizon, risk label, 거래 정책을 먼저 다시 봐야 한다.

## 10. 참고 문헌·자료 반영

- `From On-chain to Macro` 계열 연구는 기술지표, 온체인, sentiment/search, 전통시장, macro data source를 분리해 성능 기여를 보아야 한다는 점을 강조한다.
- `Decoding Bitcoin` 계열 연구는 micro/macro factor를 함께 쓰는 방향이 단일 가격 입력보다 타당하다는 흐름을 제공한다.
- 코인 price prediction survey들은 OHLCV, order book, social sentiment, on-chain, macro 변수가 넓게 쓰인다고 정리한다.
- 금융 uncertainty quantification 연구는 점예측만 보는 대신 확률·예측구간·위험 gate를 함께 보는 방향을 지지한다.
- multi-timeframe feature engineering 연구와 실무 자료는 단일 15분봉만 보지 않고 여러 시간축의 수익률, 변동성, 거래량 구조를 함께 보는 쪽을 강조한다.
- order flow 실무 자료는 실제 order book이 없더라도 거래량 폭증, candle body/wick, close location, signed volume 같은 proxy를 먼저 검증할 수 있다는 힌트를 준다.

12번은 이 문헌 흐름을 전부 구현하는 최종 데이터마트가 아니라, 어떤 입력군을 정식 데이터마트로 올릴지 고르는 선별 실험이다.
